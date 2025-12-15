import os
import json
import random
import argparse
from typing import List, Tuple, Optional

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


def resolve_image_path(image_root: str, image_field: str, strip_prefix: Optional[str]):
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

    # Si déjà <image>, remplacer; sinon, préfixer.
    if "<image>" in user_text:
        user_content = user_text.replace("<image>", "<image_placeholder>")
    else:
        user_content = "<image_placeholder>\n" + user_text

    # Optionnel: system prompt injecté au début
    if system_prompt:
        user_content = system_prompt.strip() + "\n\n" + user_content

    return [
        {"role": "User", "content": user_content, "images": [img_path]},
        {"role": "Assistant", "content": target_md},
    ]


def force_fp16_vision_inputs(batch):
    """
    Fix dtype: certains pipelines produisent du bf16 côté vision, alors que le
    vision tower (conv) est fp16. On force donc tout bf16 -> fp16.
    """
    # dict-like BatchEncoding
    if hasattr(batch, "items"):
        for k, v in list(batch.items()):
            if torch.is_tensor(v) and v.dtype == torch.bfloat16:
                batch[k] = v.to(torch.float16)

    # attributs possibles
    for name in ["pixel_values", "images", "high_images", "low_images"]:
        if hasattr(batch, name):
            v = getattr(batch, name)
            if torch.is_tensor(v) and v.dtype == torch.bfloat16:
                setattr(batch, name, v.to(torch.float16))
    return batch


def freeze_module(mod: torch.nn.Module):
    mod.eval()
    for p in mod.parameters():
        p.requires_grad_(False)


def safe_enable_gradient_checkpointing(mod, name="module"):
    fn = getattr(mod, "gradient_checkpointing_enable", None)
    if fn is None:
        print(f"[ckpt] {name}: no gradient_checkpointing_enable()")
        return False
    try:
        fn()
        print(f"[ckpt] {name}: enabled ✅")
        return True
    except Exception as e:
        print(f"[ckpt] {name}: not supported ({type(e).__name__}: {e})")
        return False


def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_ratio)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ----------------------------
# Dataset
# ----------------------------
class JsonlVLDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        processor,
        image_root: str,
        strip_prefix: Optional[str] = None,
        system_prompt: Optional[str] = None,
        indices: Optional[List[int]] = None,
    ):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

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
        pil_images = load_pil_images(conv)

        inputs = self.processor(
            conversations=conv,
            images=pil_images,
            force_batchify=True,
        )
        return inputs


def collate_fn(batch):
    # Simple: batch_size=1
    assert len(batch) == 1, "Utilise batch_size=1 (recommandé pour commencer)."
    return batch[0]


# ----------------------------
# Masking helpers
# ----------------------------
def mask_labels_to_train_only_assistant(labels, input_ids, tokenizer):
    """
    Masque tout ce qui précède le début de la réponse assistant.

    Par défaut, on cherche le tag "Assistant:" (comme dans ton premier script).
    Si ton format n’inclut pas "Assistant:", la loss peut devenir bizarre.
    """
    assistant_token_ids = tokenizer.encode("Assistant:", add_special_tokens=False)
    ids = input_ids[0].tolist()
    start = -1
    for i in range(len(ids) - len(assistant_token_ids)):
        if ids[i : i + len(assistant_token_ids)] == assistant_token_ids:
            start = i + len(assistant_token_ids)
            break
    if start != -1:
        labels[0, :start] = -100
    return labels


# ----------------------------
# Eval
# ----------------------------
@torch.no_grad()
def evaluate_loss(
    model,
    tokenizer,
    dataloader,
    device,
    max_batches: int = 50,
):
    """
    Eval loss moyenne sur `max_batches` batches de val (batch_size=1).
    On garde no_grad + pas de graph vision.
    """
    model.language_model.eval()

    losses = []
    it = iter(dataloader)
    for _ in range(max_batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        batch = batch.to(device)
        batch = force_fp16_vision_inputs(batch)

        # vision / multimodal embeds sans grad
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                inputs_embeds = model.prepare_inputs_embeds(**batch)

        labels = batch.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels = mask_labels_to_train_only_assistant(labels, batch.input_ids, tokenizer)

        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.attention_mask,
            labels=labels,
        )
        losses.append(float(outputs.loss.detach().cpu()))

    model.language_model.train()
    if not losses:
        return None
    return sum(losses) / len(losses)


# ----------------------------
# Main
# ----------------------------
def main():
    # Evite les warnings tokenizers fork
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Réduit fragmentation mémoire
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # TF32 ok (3090)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["training"]["seed"])
    set_seed(seed)

    model_name = cfg["model"]["name"]
    torch_dtype = torch.float16 if cfg["model"].get("torch_dtype", "float16") == "float16" else torch.float32

    jsonl_path = cfg["dataset"]["jsonl_path"]
    image_root = cfg["dataset"]["image_root"]
    strip_prefix = cfg["dataset"].get("strip_image_prefix", None)

    system_prompt = None
    if cfg["dataset"].get("add_system_prompt", False):
        system_prompt = cfg["dataset"].get("system_prompt", None)

    out_dir = cfg["training"]["output_dir"]
    ensure_dir(out_dir)

    loss_log_path = os.path.join(out_dir, "loss.csv")
    with open(loss_log_path, "w", encoding="utf-8") as f:
        f.write("step,train_loss,val_loss\n")

    # ----------------------------
    # Processor & model
    # ----------------------------
    processor = VLChatProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.train()

    # Gradient checkpointing : seulement sur le LM interne si possible
    if cfg["model"].get("gradient_checkpointing", True):
        if hasattr(model, "language_model"):
            safe_enable_gradient_checkpointing(model.language_model, "model.language_model")

    # Checkpointing & cache
    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        model.language_model.config.use_cache = False

    # Freeze vision + aligner (LoRA uniquement sur LM)
    if hasattr(model, "vision_model"):
        freeze_module(model.vision_model)
    if hasattr(model, "aligner"):
        freeze_module(model.aligner)

    # LoRA
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

    # ----------------------------
    # Dataset split train/val
    # ----------------------------
    # Valeurs par défaut si pas dans config
    val_ratio = float(cfg["dataset"].get("val_ratio", 0.1))  # 10% -> 900/100 si 1000
    split_seed = int(cfg["dataset"].get("split_seed", seed))

    # On charge une fois pour connaître N, puis on split par indices
    tmp_ds = JsonlVLDataset(
        jsonl_path=jsonl_path,
        processor=processor,
        image_root=image_root,
        strip_prefix=strip_prefix,
        system_prompt=system_prompt,
        indices=None,
    )
    n_total = len(tmp_ds)
    train_idx, val_idx = split_indices(n_total, val_ratio=val_ratio, seed=split_seed)
    print(f"[data] total={n_total} train={len(train_idx)} val={len(val_idx)} (val_ratio={val_ratio})")

    ds_train = JsonlVLDataset(
        jsonl_path=jsonl_path,
        processor=processor,
        image_root=image_root,
        strip_prefix=strip_prefix,
        system_prompt=system_prompt,
        indices=train_idx,
    )
    ds_val = JsonlVLDataset(
        jsonl_path=jsonl_path,
        processor=processor,
        image_root=image_root,
        strip_prefix=strip_prefix,
        system_prompt=system_prompt,
        indices=val_idx,
    )

    # Loaders
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"].get("num_workers", 0))

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ----------------------------
    # Optim & schedule
    # ----------------------------
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"]["weight_decay"])
    optimizer = torch.optim.AdamW(model.language_model.parameters(), lr=lr, weight_decay=wd)

    grad_accum = int(cfg["training"]["grad_accum"])
    max_steps = int(cfg["training"]["max_steps"])
    if args.max_steps is not None:
        max_steps = args.max_steps

    log_every = int(cfg["training"]["log_every"])
    save_every = int(cfg["training"]["save_every"])

    # Eval config
    eval_every = int(cfg["training"].get("eval_every", 200))  # toutes les 200 steps
    val_steps = int(cfg["training"].get("val_steps", 50))     # max 50 batches val à chaque eval

    # ----------------------------
    # Smoke test forward (no grad)
    # ----------------------------
    first = ds_train[0].to(model.device)
    first = force_fp16_vision_inputs(first)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            _ = model.prepare_inputs_embeds(**first)

    # ----------------------------
    # Train loop
    # ----------------------------
    step = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=max_steps, desc="train")
    dl_iter = iter(dl_train)

    while step < max_steps:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl_train)
            batch = next(dl_iter)

        batch = batch.to(model.device)
        batch = force_fp16_vision_inputs(batch)

        # multimodal embeds SANS graph (grosse économie VRAM)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                inputs_embeds = model.prepare_inputs_embeds(**batch)

        # labels
        labels = batch.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels = mask_labels_to_train_only_assistant(labels, batch.input_ids, tokenizer)

        # forward LM
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

        # Logging + debug tokens
        val_loss = ""
        if step % log_every == 0:
            trained_tokens = int((labels != -100).sum().item())
            total_tokens = int(labels.numel())
            pbar.set_postfix({"loss": round(float(loss.detach().cpu()), 4)})
            print(f"[dbg] step={step} trained_tokens={trained_tokens}/{total_tokens}")

        # Eval val
        if eval_every > 0 and (step + 1) % eval_every == 0:
            v = evaluate_loss(
                model=model,
                tokenizer=tokenizer,
                dataloader=dl_val,
                device=model.device,
                max_batches=val_steps,
            )
            if v is not None:
                val_loss = f"{v:.6f}"
                print(f"[val] step={step+1} val_loss={val_loss}")

        # CSV log
        if step % log_every == 0:
            with open(loss_log_path, "a", encoding="utf-8") as f:
                f.write(f"{step},{float(loss.detach().cpu())},{val_loss}\n")

        # Save LoRA
        if (step + 1) % save_every == 0:
            save_path = os.path.join(out_dir, f"step_{step+1}")
            ensure_dir(save_path)
            model.language_model.save_pretrained(save_path)
            # (optionnel mais propre)
            try:
                processor.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
            except Exception:
                pass

        step += 1
        pbar.update(1)

    pbar.close()

    # Final save
    final_path = os.path.join(out_dir, "final")
    ensure_dir(final_path)
    model.language_model.save_pretrained(final_path)
    try:
        processor.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
    except Exception:
        pass

    print(f"✅ LoRA saved to: {final_path}")


if __name__ == "__main__":
    main()
