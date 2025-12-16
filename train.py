# train.py
import argparse, yaml, json
from pathlib import Path
import torch

from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(cfg_path):
    cfg = load_cfg(cfg_path)
    torch.manual_seed(cfg.get("seed", 42))

    ds = load_from_disk(cfg["data"]["hf_dataset_dir"])
    train_ds = ds[cfg["data"]["train_split"]]
    eval_ds  = ds[cfg["data"]["eval_split"]]

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
        output_dir=out_dir,
        max_steps=tcfg["max_steps"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        warmup_steps=tcfg["warmup_steps"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        eval_steps=tcfg["eval_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        max_seq_length=tcfg["max_seq_length"],
        report_to=tcfg.get("report_to", ["tensorboard"]),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and (not torch.cuda.is_bf16_supported()),
        remove_unused_columns=False,  # important multimodal
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor,
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
