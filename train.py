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

def sanitize_messages_ultra(messages):
    m = copy.deepcopy(messages)
    out = []
    for msg in m:
        role = msg.get("role")
        content = msg.get("content", [])
        new_content = []
        for c in content:
            t = c.get("type")

            # If type missing, try to infer
            if t is None:
                if "image" in c:
                    t = "image"
                elif "text" in c:
                    t = "text"

            if t == "image":
                img = c.get("image", None)
                if not isinstance(img, str) or len(img.strip()) == 0:
                    # keep it explicit so we catch bad rows early
                    raise ValueError("Found image block with invalid `image` value")
                new_content.append({"type": "image", "image": img})

            elif t == "text":
                txt = c.get("text", None)
                if not isinstance(txt, str) or len(txt.strip()) == 0:
                    continue
                new_content.append({"type": "text", "text": txt})

            else:
                # Drop unknown multimodal types to avoid surprising qwen_vl_utils
                # or keep them if you need later
                pass

        out.append({"role": role, "content": new_content})
    return out

def format_example(example, processor):
    messages = example["messages"]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # IMPORTANT: ne pas toucher aux images ici
    return {
        "text": text,
        "messages": messages
    }


def collate_fn(processor, batch):
    texts = []
    images = []
    videos = []

    for b in batch:
        texts.append(b["text"])
        img_in, vid_in = process_vision_info(b["messages"])
        images.append(img_in)
        videos.append(vid_in)

    inputs = processor(
        text=texts,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )

    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(cfg_path):
    cfg = load_cfg(cfg_path)
    torch.manual_seed(cfg.get("seed", 42))

    model_id = cfg["model"]["base_model"]
    out_dir = cfg["train"]["output_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1) Load model + processor FIRST (processor is needed for apply_chat_template and vision processing)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        load_in_4bit=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # 2) Load dataset
    ds = load_from_disk(cfg["data"]["hf_dataset_dir"])
    train_ds = ds[cfg["data"]["train_split"]]
    eval_ds  = ds[cfg["data"]["eval_split"]]

    # 3) Sanitize messages once
    def sanitize_example(ex):
        try:
            ex["messages"] = sanitize_messages_ultra(ex["messages"])
            return ex
        except Exception as e:
            raise RuntimeError(f"Sanitize failed for id={ex.get('id')} image_path={ex.get('image_path')} -> {e}")
    
    train_ds = train_ds.map(sanitize_example)
    eval_ds  = eval_ds.map(sanitize_example)

    # 4) Format dataset to explicit multimodal inputs (text + images/videos)
    train_fmt = train_ds.map(
        format_example,
        fn_kwargs={"processor": processor},
        remove_columns=train_ds.column_names,
    )

    eval_fmt = eval_ds.map(
        format_example,
        fn_kwargs={"processor": processor},
        remove_columns=eval_ds.column_names,
    )

    # 5) Add LoRA
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

    # 6) Trainer config
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

    # 7) Train (IMPORTANT: use train_fmt/eval_fmt)
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_fmt,
        eval_dataset=eval_fmt,
        data_collator=lambda batch: collate_fn(processor, batch),
    )

    trainer.train()

    # 8) Save adapter + processor + logs
    trainer.model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

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
