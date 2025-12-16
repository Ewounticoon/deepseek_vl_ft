# eval.py
import argparse, json, random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch

from datasets import load_from_disk
from jiwer import wer
from rapidfuzz.distance import Levenshtein
from sacrebleu.metrics import CHRF
from rouge_score import rouge_scorer

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
                    # optional: skip empty text blocks
                    continue
                new_content.append({"type": "text", "text": c["text"]})
            else:
                # keep unknown types as-is (rare)
                new_content.append(c)
        msg["content"] = new_content
    return m


def norm(s):
    return "\n".join([l.rstrip() for l in s.replace("\r\n","\n").replace("\r","\n").split("\n")]).strip()

def cer(ref, hyp):
    ref, hyp = norm(ref), norm(hyp)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


@torch.no_grad()
def generate(model, processor, user_messages, max_new_tokens):
    user_messages = sanitize_messages_strict(user_messages)

    text = processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)

    # Debug (temporaire) : combien de slots image dans le texte ?
    # print("image placeholders in prompt:", text.count("<image>"))

    image_inputs, video_inputs = process_vision_info(user_messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)      # base model id OR outputs/finetuned
    ap.add_argument("--dataset", required=True)    # data/hf_dataset
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(args.dataset)
    eval_ds = ds["eval"]
    idxs = list(range(len(eval_ds)))
    random.shuffle(idxs)
    idxs = idxs[: min(args.n_samples, len(idxs))]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, device_map="auto", torch_dtype="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model)

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    chrf = CHRF()

    preds_path = out_dir / "predictions.jsonl"
    fout = preds_path.open("w", encoding="utf-8")

    all_c, all_w, all_r, all_cf = [], [], [], []

    for i in tqdm(idxs):
        sample = eval_ds[i]
        # user turn is messages[0], gt is messages[1]
        user_msgs = [sample["messages"][0]]
        gt = sample["messages"][1]["content"][0]["text"]
        pred = generate(model, processor, user_msgs, args.max_new_tokens)

        gt_n, pr_n = norm(gt), norm(pred)
        c = cer(gt_n, pr_n)
        w = wer(gt_n, pr_n)
        r = rouge.score(gt_n, pr_n)["rougeL"].fmeasure
        cf = chrf.sentence_score(pr_n, [gt_n]).score / 100.0

        all_c.append(c); all_w.append(w); all_r.append(r); all_cf.append(cf)
        fout.write(json.dumps({"i": int(i), "gt": gt, "pred": pred, "cer": c, "wer": w, "rougeL": r, "chrf": cf}, ensure_ascii=False) + "\n")

    fout.close()

    metrics = {
        "n": len(all_c),
        "CER_mean": float(np.mean(all_c)),
        "WER_mean": float(np.mean(all_w)),
        "ROUGE_L_f1_mean": float(np.mean(all_r)),
        "chrF_mean": float(np.mean(all_cf)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Saved:", preds_path)

if __name__ == "__main__":
    main()
