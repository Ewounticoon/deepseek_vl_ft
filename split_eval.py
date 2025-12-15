import json, random, os

random.seed(0)

src = "data/dataset.jsonl"
train_out = "data/train.jsonl"
eval_out  = "data/eval.jsonl"

with open(src, "r", encoding="utf-8") as f:
    lines = [l for l in f.read().splitlines() if l.strip()]

random.shuffle(lines)

n_eval = min(50, max(20, int(0.02 * len(lines))))  # 2% ou min 20, max 50
eval_lines = lines[:n_eval]
train_lines = lines[n_eval:]

os.makedirs("data", exist_ok=True)
open(eval_out, "w", encoding="utf-8").write("\n".join(eval_lines) + "\n")
open(train_out, "w", encoding="utf-8").write("\n".join(train_lines) + "\n")

print("✅ train:", len(train_lines))
print("✅ eval :", len(eval_lines))
