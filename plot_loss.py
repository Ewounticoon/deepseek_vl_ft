# plot_loss.py
import argparse, json
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

hist = json.load(open(args.log, "r", encoding="utf-8"))
steps, losses = [], []
for row in hist:
    if "loss" in row and "step" in row:
        steps.append(row["step"]); losses.append(row["loss"])

plt.figure()
plt.plot(steps, losses)
plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training loss")
plt.savefig(args.out, dpi=160, bbox_inches="tight")
print("Saved:", args.out)
