import os
import json
import argparse
from difflib import SequenceMatcher

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from PIL import Image

from deepseek_vl.models import VLChatProcessor


# ============================================================
# Metrics
# ============================================================

def cer(pred, ref):
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return 1.0 - SequenceMatcher(None, pred, ref).ratio()


def line_em(pred, ref):
    p = [l.strip() for l in pred.splitlines() if l.strip()]
    r = [l.strip() for l in ref.splitlines() if l.strip()]
    if not r:
        return 1.0 if not p else 0.0
    s = set(r)
    return sum(1 for l in p if l in s) / len(r)


def table_pipe_score(pred, ref):
    p_lines = [l for l in pred.splitlines() if "|" in l]
    r_lines = [l for l in ref.splitlines() if "|" in l]
    if not r_lines:
        return 1.0 if not p_lines else 0.0
    n = min(len(p_lines), len(r_lines))
    if n == 0:
        return 0.0
    ok = 0
    for i in range(n):
        if p_lines[i].count("|") == r_lines[i].count("|"):
            ok += 1
    return ok / n


# ============================================================
# Helpers
# ============================================================

def resolve_image_path(image_root, image_field, strip_prefix):
    if strip_prefix and image_field.startswith(strip_prefix):
        image_field = image_field[len(strip_prefix):]
    return os.path.join(image_root, image_field)


def extract_user_and_ref(convs):
    user_msgs = []
    ref = ""
    for c in convs:
        if c.get("from") in ("user", "human"):
            user_msgs.append(c["value"])
        elif c.get("from") in ("assistant", "gpt"):
            ref = c["value"]
    return "\n".join(user_msgs), ref


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/deepseek-vl-7b-base")
    ap.add_argument("--eval_jsonl", default="data/eval.jsonl")
    ap.add_argument("--image_root", default="data/images")
    ap.add_argument("--strip_prefix", default="images/")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--out_dir", default="reports/fixed")
    ap.add_argument("--max_samples", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--system_prompt", default=None, help="Optional instruction prefix")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = "cuda"

    # Processor / tokenizer
    processor = VLChatProcessor.from_pretrained(args.model)
    tok = processor.tokenizer

    # Model (MultiModalityCausalLM sous le capot)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    ).eval()

    model = base
    if args.lora:
        model = PeftModel.from_pretrained(base, args.lora).eval()

    # Load dataset
    samples = []
    with open(args.eval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    samples = samples[: args.max_samples]

    metrics = {"cer": [], "line_em": [], "table_pipe": []}

    report_path = os.path.join(args.out_dir, "samples.md")
    with open(report_path, "w", encoding="utf-8") as rep:
        rep.write("# Eval report (FIXED – MultiModalityCausalLM)\n\n")
        rep.write(f"- model: {args.model}\n")
        rep.write(f"- lora: {args.lora}\n")
        rep.write(f"- samples: {len(samples)}\n\n")

        for s in tqdm(samples, desc="eval"):
            user_text, ref = extract_user_and_ref(s["conversations"])

            # Ensure <image> tag exists
            if "<image>" not in user_text:
                user_text = "<image>\n" + user_text

            # Optional system prompt (prefix inside user message)
            if args.system_prompt:
                user_text = args.system_prompt.strip() + "\n\n" + user_text

            img_path = resolve_image_path(args.image_root, s["image"], args.strip_prefix)
            assert os.path.exists(img_path), img_path

            # Load PIL image (CRITICAL)
            img = Image.open(img_path).convert("RGB")

            # DeepSeek-VL conversation format
            conv = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": ""}  # trigger generation
            ]

            # Build batch
            batch = processor(
                conversations=conv,
                images=[img],
                force_batchify=True,
            ).to(device)

            # Prepare multimodal embeddings
            with torch.no_grad():
                inputs_embeds = model.prepare_inputs_embeds(**batch)

                # Generate from language_model (Transformers generation works here)
                out = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch.attention_mask,
                    pad_token_id=tok.eos_token_id,
                    bos_token_id=tok.bos_token_id,
                    eos_token_id=tok.eos_token_id,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=4,
                )

            # IMPORTANT: remove the "prompt" part from generation if present
            # out[0] includes prompt tokens in many cases; safest: cut by input length if available.
            if hasattr(batch, "input_ids") and batch.input_ids is not None:
                prompt_len = batch.input_ids.shape[1]
                gen_tokens = out[0][prompt_len:]
                pred = tok.decode(gen_tokens.detach().cpu().tolist(), skip_special_tokens=True)
            else:
                pred = tok.decode(out[0].detach().cpu().tolist(), skip_special_tokens=True)

            pred = pred.strip()

            # Metrics
            metrics["cer"].append(cer(pred, ref))
            metrics["line_em"].append(line_em(pred, ref))
            metrics["table_pipe"].append(table_pipe_score(pred, ref))

            # Report
            rep.write(f"## {s.get('id','')}\n\n")
            rep.write(f"**Image:** `{s['image']}`\n\n")
            rep.write("### Prediction\n\n```markdown\n")
            rep.write(pred[:8000] + "\n```\n\n")
            rep.write("### Reference\n\n```markdown\n")
            rep.write(ref[:8000] + "\n```\n\n---\n\n")

    summary = {
        "n": len(samples),
        "cer_mean": float(sum(metrics["cer"]) / max(1, len(metrics["cer"]))),
        "line_em_mean": float(sum(metrics["line_em"]) / max(1, len(metrics["line_em"]))),
        "table_pipe_mean": float(sum(metrics["table_pipe"]) / max(1, len(metrics["table_pipe"]))),
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("✅ FIXED eval done (MultiModalityCausalLM path)")
    print(summary)


if __name__ == "__main__":
    main()

