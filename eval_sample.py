import os, json, argparse, re
from difflib import SequenceMatcher

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

# ---------- helpers ----------
def resolve_image_path(image_root, image_field, strip_prefix=None):
    if strip_prefix and image_field.startswith(strip_prefix):
        image_field = image_field[len(strip_prefix):]
    return os.path.join(image_root, image_field)

def user_assistant_from_conversations(convs):
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

def to_deepseek_conversation(sample, image_root, strip_prefix=None, system_prompt=None):
    user_text, target_md = user_assistant_from_conversations(sample["conversations"])
    user_text = user_text.replace("<image>", "<image_placeholder>")
    if "<image_placeholder>" not in user_text:
        user_text = "<image_placeholder>\n" + user_text

    if system_prompt:
        user_text = system_prompt.strip() + "\n\n" + user_text

    img_path = resolve_image_path(image_root, sample["image"], strip_prefix)
    return [
        {"role": "User", "content": user_text, "images": [img_path]},
        {"role": "Assistant", "content": target_md},
    ], target_md

def cer(pred, ref):
    # char error rate (1 - similarity)
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    sm = SequenceMatcher(None, pred, ref)
    return 1.0 - sm.ratio()

def line_em(pred, ref):
    p = [l.strip() for l in pred.strip().splitlines() if l.strip()]
    r = [l.strip() for l in ref.strip().splitlines() if l.strip()]
    if not r:
        return 1.0 if not p else 0.0
    s = set(r)
    ok = sum(1 for l in p if l in s)
    return ok / max(1, len(r))

def table_pipe_score(pred, ref):
    # % de lignes "table" avec même nombre de pipes que ref
    p_lines = [l for l in pred.splitlines() if "|" in l]
    r_lines = [l for l in ref.splitlines() if "|" in l]
    if not r_lines:
        return 1.0 if not p_lines else 0.0
    # compare line-by-line up to min length
    n = min(len(p_lines), len(r_lines))
    if n == 0:
        return 0.0
    ok = 0
    for i in range(n):
        if p_lines[i].count("|") == r_lines[i].count("|"):
            ok += 1
    return ok / n

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/deepseek-vl-7b-base")
    ap.add_argument("--eval_jsonl", default="data/eval.jsonl")
    ap.add_argument("--image_root", default="data/images")
    ap.add_argument("--strip_prefix", default="images/")
    ap.add_argument("--lora", default=None, help="path to LoRA adapters (optional)")
    ap.add_argument("--out_dir", default="reports/before")
    ap.add_argument("--max_samples", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Run this eval on a GPU machine (RunPod)."

    processor = VLChatProcessor.from_pretrained(args.model)
    tok = processor.tokenizer

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

    # load samples
    samples = []
    with open(args.eval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    samples = samples[: args.max_samples]

    preds = []
    metrics = {"cer": [], "line_em": [], "table_pipe": []}

    md_report_path = os.path.join(args.out_dir, "samples.md")
    with open(md_report_path, "w", encoding="utf-8") as rep:
        rep.write(f"# Eval report\n\n")
        rep.write(f"- model: {args.model}\n")
        rep.write(f"- lora: {args.lora}\n")
        rep.write(f"- samples: {len(samples)}\n\n")

        for s in tqdm(samples, desc="eval"):
            conv, ref = to_deepseek_conversation(s, args.image_root, args.strip_prefix, system_prompt=None)
            pil_images = load_pil_images(conv)
            batch = processor(conversations=conv, images=pil_images, force_batchify=True).to("cuda")

            inputs_embeds = model.prepare_inputs_embeds(**batch)
            out = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=batch.attention_mask,
                pad_token_id=tok.eos_token_id,
                bos_token_id=tok.bos_token_id,
                eos_token_id=tok.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            pred = tok.decode(out[0].detach().cpu().tolist(), skip_special_tokens=True)

            # metrics
            metrics["cer"].append(cer(pred, ref))
            metrics["line_em"].append(line_em(pred, ref))
            metrics["table_pipe"].append(table_pipe_score(pred, ref))

            # report (evidence)
            img_rel = s["image"]
            rep.write(f"## {s.get('id','')}\n\n")
            rep.write(f"**Image:** `{img_rel}`\n\n")
            rep.write("### Prediction\n\n```markdown\n")
            rep.write(pred.strip()[:8000] + "\n```\n\n")
            rep.write("### Reference\n\n```markdown\n")
            rep.write(ref.strip()[:8000] + "\n```\n\n---\n\n")

    # aggregate metrics
    summary = {
        "n": len(samples),
        "cer_mean": float(sum(metrics["cer"]) / len(metrics["cer"])),
        "line_em_mean": float(sum(metrics["line_em"]) / len(metrics["line_em"])),
        "table_pipe_mean": float(sum(metrics["table_pipe"]) / len(metrics["table_pipe"])),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("✅ Saved:", args.out_dir)
    print(summary)

if __name__ == "__main__":
    main()
