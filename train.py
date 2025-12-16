# train.py
import argparse, yaml, json
from pathlib import Path
import torch

from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import copy

def sanitize_messages_strict(messages):
    m = copy.deepcopy(messages)
    for msg in m:
        if "content" not in msg:
            continue
        new_content = []
        for c in msg["content"]:
            t = c.get("type")
            if t == "image":
                if c.get("image") is None:
                    raise ValueError("Found an image block with image=None")
                new_content.append({"type": "image", "image": c["image"]})
            elif t == "text":
                if c.get("text") is None:
                    continue
                new_content.append({"type": "text", "text": c["text"]})
            else:
                new_content.append(c)
        msg["content"] = new_content
    return m

def format_example(processor, example):
    # messages already sanitized
    messages = example["messages"]

    # Full conversation (user + assistant) so the model learns the assistant part
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Extract the image(s) from the messages
    image_inputs, video_inputs = process_vision_info(messages)

    if len(image_inputs) != 1:
        # In your dataset, it should be exactly 1 image per example
        raise ValueError(f"Expected 1 image, got {len(image_inputs)} for id={example.get('id')}")

    return {
        "text": text,
        "image_inputs": image_inputs,   # list length 1
        "video_inputs": video_inputs,
    }


def collate_fn(processor, batch):
    texts = [b["text"] for b in batch]
    images = [b["image_inputs"] for b in batch]  # list of list-of-images
    videos = [b["video_inputs"] for b in batch]

    inputs = processor(
        text=texts,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )

    # Standard SFT: labels = input_ids (masking is handled by chat template structure)
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(cfg_path):
    cfg = load_cfg(cfg_path)
    torch.manual_seed(cfg.get("seed", 42))

    ds = load_from_disk(cfg["data"]["hf_dataset_dir"])
    train_ds = ds[cfg["data"]["train_split"]]
    eval_ds  = ds[cfg["data"]["eval_split"]]

    def sanitize_example(ex):
        ex["messages"] = sanitize_messages_strict(ex["messages"])
        return ex

    train_ds = train_ds.map(sanitize_example)
    eval_ds  = eval_ds.map(sanitize_example)

    train_fmt = train_ds.map(
        lambda ex: format_example(processor, ex),
        remove_columns=train_ds.column_names,
    )

    eval_fmt = eval_ds.map(
        lambda ex: format_example(processor, ex),
        remove_columns=eval_ds.column_names,
    )

    model_id = cfg["model"]["base_model"]
    out_dir = cfg["train"]["output_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        load_in_4bit=True,  # QLoRA style
    )
    processor = AutoProcessor.from_pretrained(model_id)

    lora = cfg["lora"]
    lora_cfg = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    tcfg = cfg["train"]
    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        max_steps=tcfg["max_steps"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        warmup_steps=tcfg["warmup_steps"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        eval_steps=tcfg["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        report_to=tcfg.get("report_to", ["tensorboard"]),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and (not torch.cuda.is_bf16_supported()),
    
        remove_unused_columns=False,
        max_length=tcfg["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=lambda batch: collate_fn(processor, batch),
    )
    trainer.train()

    # save adapter + processor
    trainer.model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

    # save loss history for plotting
    (Path(out_dir) / "trainer_log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Saved:", out_dir)
    print("TensorBoard:", f"tensorboard --logdir {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
