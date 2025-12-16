# prepare_dataset.py (FAST: store file paths, not PIL)
import argparse, json, os, random, time
from pathlib import Path
from datasets import Dataset, DatasetDict

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def extract_user_and_assistant(conversations):
    user_text = None
    assistant_text = None
    for msg in conversations:
        if msg.get("from") == "user" and user_text is None:
            user_text = msg.get("value", "")
        if msg.get("from") == "assistant" and assistant_text is None:
            assistant_text = msg.get("value", "")
    if user_text is None or assistant_text is None:
        raise ValueError("Missing user or assistant message in conversations.")
    return user_text, assistant_text

def strip_image_token(s: str) -> str:
    return s.replace("<image>", "").lstrip()

def to_file_uri(path: str) -> str:
    # absolute file:// URI (works well with Qwen2.5-VL) :contentReference[oaicite:1]{index=1}
    ap = os.path.abspath(path)
    return "file://" + ap

def main(jsonl_path, out_dir, seed=42, eval_ratio=0.1, data_root="data", log_every=50):
    t0 = time.time()
    random.seed(seed)

    rows = list(read_jsonl(jsonl_path))
    random.shuffle(rows)
    print(f"[INFO] Loaded {len(rows)} rows", flush=True)

    samples = []
    for idx, r in enumerate(rows):
        if idx % log_every == 0:
            print(f"[INFO] Building {idx+1}/{len(rows)}", flush=True)

        sample_id = r.get("id", f"idx_{idx}")
        img_rel = r["image"]
        user_value, gt = extract_user_and_assistant(r["conversations"])
        instruction = strip_image_token(user_value)

        # resolve file path
        img_path = img_rel
        if not os.path.exists(img_path):
            img_path = os.path.join(data_root, img_rel)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_rel} (resolved {img_path})")

        img_uri = to_file_uri(img_path)

        samples.append({
            "id": sample_id,
            "image_path": img_rel,
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img_uri},
                    {"type": "text", "text": instruction},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": gt},
                ]},
            ],
        })

    print("[INFO] Finished building samples", flush=True)

    n = len(samples)
    n_eval = max(1, int(n * eval_ratio))
    eval_samples = samples[:n_eval]
    train_samples = samples[n_eval:]

    ds = DatasetDict({
        "train": Dataset.from_list(train_samples),
        "eval": Dataset.from_list(eval_samples),
    })

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving dataset to: {out_dir}", flush=True)
    ds.save_to_disk(out_dir)
    print(f"[DONE] Saved. Total time: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_ratio", type=float, default=0.1)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()
    main(args.jsonl, args.out, args.seed, args.eval_ratio, args.data_root, args.log_every)
