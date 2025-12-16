# train.py
import argparse, yaml, json
from pathlib import Path
import copy

import torch
from datasets import load_from_disk, Features, Sequence, Image
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sanitize_messages_for_trl(messages):
    """
    TRL expects:
      - structured blocks in messages
      - image blocks are placeholders: {"type": "image"} (NO image=... here)
    """
    m = copy.deepcopy(messages)
    for msg in m:
        if "content" not in msg:
            continue
        new_content = []
        for c in msg["content"]:
            t = c.get("type")
            if t == "image":
                # force placeholder only
                new_content.append({"type": "image"})
            elif t == "text":
                txt = c.get("text")
                if txt is None:
                    continue
                new_content.append({"type": "text", "text": txt})
            else:
                new_content.append(c)
        msg["content"] = new_content
    return m


def add_images_column(example, images_dir):
    msgs = sanitize_messages_for_trl(example["messages"])

    def count_image_placeholders(messages):
        n = 0
        for msg in messages:
            for c in msg.get("content", []):
                if c.get("type") == "image":
                    n += 1
        return n


    img_path = example["image_path"]  # ex: "images/doc_0615_p0001.png" OU "data/images/..."
    img_path = str(img_path)

    # Si ça commence par file://, on enlève
    if img_path.startswith("file://"):
        img_path = img_path.replace("file://", "")

    # Si c'est un chemin relatif "images/xxx.png"
    # on le remappe vers images_dir/xxx.png
    p = Path(img_path)

    if not p.is_absolute():
        # Si le dataset contient "images/xxx.png", on prend juste le filename relatif sous images/
        # -> images_dir / doc_0615_p0001.png
        if p.parts and p.parts[0] == "images":
            p = images_dir / Path(*p.parts[1:])
        else:
            # fallback: relatif au dossier images_dir
            p = images_dir / p

    p = p.resolve()

    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p} (from image_path={example.get('image_path')})")

    return {
        "id": example.get("id"),
        "messages": msgs,
        "images": [str(p)],
    }

def main(cfg_path):
    cfg = load_cfg(cfg_path)
    torch.manual_seed(cfg.get("seed", 42))

    data_root = Path(cfg["data"]["data_root"]).resolve()   # -> /workspace/deepseek_vl_ft/data
    images_dir = data_root / "images"                      # -> /workspace/deepseek_vl_ft/data/images
    print("[INFO] data_root =", data_root)
    print("[INFO] images_dir =", images_dir)

    # --- load dataset ---
    ds = load_from_disk(cfg["data"]["hf_dataset_dir"])
    train_ds = ds[cfg["data"]["train_split"]]
    eval_ds  = ds[cfg["data"]["eval_split"]]

    # --- convert to TRL VLM dataset format ---
    train_ds = train_ds.map(
        add_images_column,
        fn_kwargs={"images_dir": images_dir},
        remove_columns=train_ds.column_names
    )
    eval_ds = eval_ds.map(
        add_images_column,
        fn_kwargs={"images_dir": images_dir},
        remove_columns=eval_ds.column_names
    )

    # cast images column so it loads as images (PIL) when accessed
    features = Features({
        "id": train_ds.features["id"],
        "messages": train_ds.features["messages"],
        "images": Sequence(Image()),
    })
    train_ds = train_ds.cast(features)
    eval_ds  = eval_ds.cast(features)

    # --- model / processor ---
    model_id = cfg["model"]["base_model"]
    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # QLoRA config (preferred over load_in_4bit arg)
    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=qcfg,
    )

    processor = AutoProcessor.from_pretrained(model_id)

    # --- LoRA ---
    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    # --- training args ---
    t = cfg["train"]
    t["learning_rate"] = float(t["learning_rate"])
    t["max_steps"] = int(t["max_steps"])
    t["per_device_train_batch_size"] = int(t["per_device_train_batch_size"])
    t["gradient_accumulation_steps"] = int(t["gradient_accumulation_steps"])
    t["warmup_steps"] = int(t["warmup_steps"])
    t["logging_steps"] = int(t["logging_steps"])
    t["save_steps"] = int(t["save_steps"])
    t["eval_steps"] = int(t["eval_steps"])
    t["max_seq_length"] = int(t["max_seq_length"])
    print("[INFO] learning_rate:", t["learning_rate"], type(t["learning_rate"]))

    sft_args = SFTConfig(
        output_dir=str(out_dir),
        max_steps=t["max_steps"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_steps=t.get("warmup_steps", 0),
        logging_steps=t.get("logging_steps", 10),
        save_steps=t.get("save_steps", 200),
        eval_steps=t.get("eval_steps", 200),
        eval_strategy="steps",
        save_strategy="steps",
        report_to=t.get("report_to", ["tensorboard"]),
        remove_unused_columns=False,  # IMPORTANT for multimodal
        max_length=t.get("max_seq_length", None),  # ok in TRL (max_length)
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and (not torch.cuda.is_bf16_supported()),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=processor,  # tokenizer -> processing_class in new TRL
    )

    trainer.train()

    # Save adapter + processor
    trainer.model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

    # Save trainer logs for plotting loss later
    (out_dir / "trainer_log_history.json").write_text(
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
