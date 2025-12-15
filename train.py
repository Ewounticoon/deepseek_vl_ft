import os
import json
import math
import random
import argparse
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_image_path(image_root: str, image_field: str, strip_prefix: str | None):
    if strip_prefix and image_field.startswith(strip_prefix):
        image_field = image_field[len(strip_prefix):]
    return os.path.join(image_root, image_field)


def user_assistant_from_conversations(convs):
    """
    Supporte les formats fréquents:
      - [{"from":"human","value":"..."}, {"from":"gpt","value":"..."}]
      - [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
    Retourne (user_text, assistant_text)
    """
    user_parts, assistant_parts = [], []
    for t in convs:
        if "from" in t:
            role = t.get("from")
            text = t.get("value", "")
            if role in ("human", "user"):
                user_parts.append(text)
            elif role in ("gpt", "assistant"):
                assistant_parts.append(text)
        else:
            role = t.get("role")
            text = t.get("content", "")
            if role == "user":
                user_parts.append(text)
            elif role == "assistant":
                assistant_parts.append(text)

    user_text = "\n\n".join([x for x in user_parts if x])
    assistant_text = assistant_parts[-1] if assistant_parts else ""
    return user_text, assistant_text


def to_deepseek_conversation(sample, image_root, strip_prefix, system_prompt=None):
    user_text, target_md = user_assistant_from_conversations(sample["conversations"])
    img_path = resolve_image_path(image_root, sample["image"], strip_prefix)

    # 1) Si ton JSONL contient déjà <image>, on le remplace par <image_placeholder>
    if "<image>" in user_text:
        user_content = user_text.replace("<image>", "<image_placeholder>")
    else:
        user_content = "<image_placeholder>\n" + user_text

    # 2) Optionnel: injecter un system prompt
    if system_prompt:
        user_content = system_prompt.strip() + "\n\n" + user_content

    return [
        {"role": "User", "content": user_content, "images": [img_path]},
        {"role": "Assistant", "content": target_md},
    ]




# ----------------------------
# Dataset
# ----------------------------
class JsonlVLDataset(Dataset):
    def __init__(self, jsonl_path, processor, image_root, strip_prefix=None, system_prompt=None):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

        self.processor = processor
        self.image_root = image_root
        self.strip_prefix = strip_prefix
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        conv = to_deepseek_conversation(
            sample,
            image_root=self.image_root,
            strip_prefix=self.strip_prefix,
            system_prompt=self.system_prompt,
        )
        # PIL images
        pil_images = load_pil_images(conv)

        # Processor output (includes input_ids, attention_mask, etc.)
        inputs = self.processor(
            conversations=conv,
            images=pil_images,
            force_batchify=True,
        )
        return inputs


def collate_fn(batch):
    # batch_size=1 recommandé. Si >1, il faudrait pad/stack; DeepSeek-VL processor renvoie déjà batchified.
    assert len(batch) == 1, "Pour commencer, utilise batch_size=1 (plus simple, moins de bugs)."
    return batch[0]


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--max_steps", type=int, default=None)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["training"]["seed"]))

    model_name = cfg["model"]["name"]
    torch_dtype = torch.float16 if cfg["model"]["torch_dtype"] == "float16" else torch.float32

    jsonl_path = cfg["dataset"]["jsonl_path"]
    image_root = cfg["dataset"]["image_root"]
    strip_prefix = cfg["dataset"].get("strip_image_prefix", None)

    system_prompt = None
    if cfg["dataset"].get("add_system_prompt", False):
        system_prompt = cfg["dataset"].get("system_prompt", None)

    out_dir = cfg["training"]["output_dir"]

    loss_log_path = os.path.join(out_dir, "loss.csv")
    with open(loss_log_path, "w") as f:
        f.write("step,loss\n")

    os.makedirs(out_dir, exist_ok=True)

    # Processor
    processor = VLChatProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    # Model (FP16 on 3090)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    if cfg["model"].get("gradient_checkpointing", False) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.train()

    # Apply LoRA on language model only
    lora_cfg = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg["lora"]["target_modules"]),
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)
    model.language_model.print_trainable_parameters()

    # Dataset / Loader
    ds = JsonlVLDataset(
        jsonl_path=jsonl_path,
        processor=processor,
        image_root=image_root,
        strip_prefix=strip_prefix,
        system_prompt=system_prompt,
    )
    dl = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Optim
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"]["weight_decay"])
    optimizer = torch.optim.AdamW(model.language_model.parameters(), lr=lr, weight_decay=wd)

    grad_accum = int(cfg["training"]["grad_accum"])
    max_steps = int(cfg["training"]["max_steps"])
    if args.max_steps is not None:
        max_steps = args.max_steps
    log_every = int(cfg["training"]["log_every"])
    save_every = int(cfg["training"]["save_every"])

    # Smoke test generation (optional but handy)
    # We do a tiny forward first to catch issues
    first = ds[0].to(model.device)
    with torch.no_grad():
        _ = model.prepare_inputs_embeds(**first)

    step = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=max_steps, desc="train")
    dl_iter = iter(dl)

    while step < max_steps:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        batch = batch.to(model.device)

        # multimodal embeds
        inputs_embeds = model.prepare_inputs_embeds(**batch)

        # labels: train on all tokens (simple baseline)
        labels = batch.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        # --- MASK USER PART (train only assistant) ---
        assistant_token_ids = tokenizer.encode("Assistant:", add_special_tokens=False)

        ids = batch.input_ids[0].tolist()
        start = -1
        for i in range(len(ids) - len(assistant_token_ids)):
            if ids[i:i+len(assistant_token_ids)] == assistant_token_ids:
                start = i + len(assistant_token_ids)
                break
            
        if start != -1:
            labels[0, :start] = -100
        # --------------------------------------------


        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.attention_mask,
            labels=labels,
        )

        loss = outputs.loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % log_every == 0:
            pbar.set_postfix({"loss": round(float(loss.detach().cpu()), 4)})
            with open(loss_log_path, "a") as f:
                f.write(f"{step},{float(loss.detach().cpu())}\n")


        if (step + 1) % save_every == 0:
            save_path = os.path.join(out_dir, f"step_{step+1}")
            os.makedirs(save_path, exist_ok=True)
            model.language_model.save_pretrained(save_path)

        step += 1
        pbar.update(1)

    pbar.close()

    # Final save
    final_path = os.path.join(out_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.language_model.save_pretrained(final_path)
    print(f"✅ LoRA saved to: {final_path}")


if __name__ == "__main__":
    main()
