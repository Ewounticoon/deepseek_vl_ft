# train.py
import argparse
import yaml
import json
from pathlib import Path
import copy

import torch
from PIL import Image
from datasets import load_from_disk
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info


# -------------------------
# Utils
# -------------------------
def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def total_token_len(processor, messages):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=[image_inputs],
        videos=[video_inputs] if video_inputs else None,
        truncation=False,
        padding=False,
        return_tensors="pt",
    )

    return inputs["input_ids"].shape[1]
def add_total_len(example, processor):
    example["total_len"] = total_token_len(processor, example["messages"])
    return example


def text_token_len(processor, messages):
    """
    Mesure le nombre de tokens (texte) du prompt complet (user+assistant),
    sans toucher aux images.
    """
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    # tokenizer pur, pas de truncation ici -> on veut détecter les outliers
    ids = processor.tokenizer(
        text,
        add_special_tokens=False,
        truncation=False
    ).input_ids
    return len(ids)


def add_text_len(example, processor):
    # exemple doit déjà avoir messages “sanitisés”
    example["text_len"] = text_token_len(processor, example["messages"])
    return example


def sanitize_messages_strict(messages):
    """
    EXACTEMENT la même logique que eval.py
    """
    m = copy.deepcopy(messages)
    for msg in m:
        if "content" not in msg:
            continue
        new_content = []
        for c in msg["content"]:
            t = c.get("type")
            if t == "image":
                if c.get("image") is None:
                    raise ValueError("Found image block with image=None")
                new_content.append({"type": "image", "image": c["image"]})
            elif t == "text":
                if c.get("text") is None:
                    continue
                new_content.append({"type": "text", "text": c["text"]})
            else:
                new_content.append(c)
        msg["content"] = new_content
    return m


def resolve_image_path(image_path, data_root: Path):
    s = str(image_path)
    if s.startswith("file://"):
        s = s.replace("file://", "")
    p = Path(s)
    if not p.is_absolute():
        p = data_root / p
    return p.resolve()


def prepare_example(example, data_root_str):
    data_root = Path(data_root_str).resolve()
    msgs = sanitize_messages_strict(example["messages"])

    # résoudre le chemin image
    img_path = resolve_image_path(example["image_path"], data_root)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # injecter le vrai chemin image dans le bloc image
    for msg in msgs:
        for c in msg["content"]:
            if c["type"] == "image":
                c["image"] = str(img_path)

    return {
        "id": example.get("id"),
        "messages": msgs,
    }


# -------------------------
# Collator (copie d’eval.py)
# -------------------------
def build_collate_fn(processor):
    def collate_fn(examples):
        texts = []
        image_inputs_all = []
        video_inputs_all = []

        for ex in examples:
            msgs = sanitize_messages_strict(ex["messages"])

            text = processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )

            image_inputs, video_inputs = process_vision_info(msgs)

            texts.append(text)
            image_inputs_all.append(image_inputs)
            video_inputs_all.append(video_inputs)

        kwargs = dict(
            text=texts,
            images=image_inputs_all,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Ne passer videos que si au moins un exemple en contient vraiment
        has_any_video = any(v is not None and len(v) > 0 for v in video_inputs_all)
        if has_any_video:
            kwargs["videos"] = video_inputs_all

        batch = processor(**kwargs)

        labels = batch["input_ids"].clone()
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # tokens image à ignorer
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


# -------------------------
# Main
# -------------------------
def main(cfg_path, resume_from_checkpoint=None):
    cfg = load_cfg(cfg_path)
    torch.manual_seed(int(cfg.get("seed", 42)))

    data_root = Path(cfg["data"]["data_root"]).resolve()
    ds = load_from_disk(cfg["data"]["hf_dataset_dir"])

    train_ds = ds[cfg["data"]["train_split"]]
    eval_ds  = ds[cfg["data"]["eval_split"]]

    train_ds = train_ds.map(
        prepare_example,
        fn_kwargs={"data_root_str": str(data_root)},
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        prepare_example,
        fn_kwargs={"data_root_str": str(data_root)},
        remove_columns=eval_ds.column_names,
    )

    # -------------------------
    # (1) model_id + processor tôt
    # -------------------------
    model_id = cfg["model"]["base_model"]
    processor = AutoProcessor.from_pretrained(model_id)

    # (optionnel mais bien pour SFT)
    processor.tokenizer.padding_side = "right"

    # -------------------------
    # (2) anti-outlier: ajouter text_len
    # -------------------------
    train_ds = train_ds.map(lambda ex: add_text_len(ex, processor))
    eval_ds  = eval_ds.map(lambda ex: add_text_len(ex, processor))

    # -------------------------
    # (3) anti-outlier: filtrer les très longs
    # -------------------------
    max_text_tokens = int(cfg["train"].get("max_text_tokens", 2800))
    print(f"[INFO] Filtering examples with text_len > {max_text_tokens}")

    before = len(train_ds)
    train_ds = train_ds.filter(lambda ex: ex["text_len"] <= max_text_tokens)
    train_ds = train_ds.filter(lambda ex: ex["id"] != "doc_0651_p0012")
    after = len(train_ds)
    print(f"[INFO] Train filtered: {before} -> {after} (removed {before-after})")

    before = len(eval_ds)
    eval_ds = eval_ds.filter(lambda ex: ex["text_len"] <= max_text_tokens)
    after = len(eval_ds)
    print(f"[INFO] Eval filtered: {before} -> {after} (removed {before-after})")

    # (optionnel debug) top 5 plus longs restants
    # longest = train_ds.sort("text_len", reverse=True).select(range(min(5, len(train_ds))))
    # for ex in longest:
    #     print("[LONG]", ex["text_len"], ex["id"])

    # -------------------------
    # le reste inchangé: model / lora / trainer
    # -------------------------
    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

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

    peft = cfg["lora"]
    peft_config = LoraConfig(
        r=int(peft["r"]),
        lora_alpha=int(peft["alpha"]),
        lora_dropout=float(peft["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    t = cfg["train"]
    training_args = SFTConfig(
        output_dir=str(out_dir),
        max_steps=int(t["max_steps"]),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        learning_rate=float(t["learning_rate"]),
        logging_steps=int(t.get("logging_steps", 10)),
        save_steps=int(t.get("save_steps", 200)),
        eval_steps=int(t.get("eval_steps", 200)),
        eval_strategy="no",
        save_strategy="steps",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=t.get("report_to", ["tensorboard"]),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and (not torch.cuda.is_bf16_supported()),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=build_collate_fn(processor),
        processing_class=processor,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)

    (out_dir / "trainer_log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[OK] Training finished")
    print("[OK] Saved to:", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = ap.parse_args()
    main(args.config, args.resume_from_checkpoint)
