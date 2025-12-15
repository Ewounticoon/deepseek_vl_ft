import csv, os
import matplotlib.pyplot as plt

csv_path = "outputs/TON_OUTPUT_DIR/loss.csv"   # change
out_png  = "outputs/TON_OUTPUT_DIR/loss.png"   # change

steps, losses = [], []
with open(csv_path, "r") as f:
    r = csv.DictReader(f)
    for row in r:
        steps.append(int(row["step"]))
        losses.append(float(row["loss"]))

plt.figure()
plt.plot(steps, losses)
plt.xlabel("step")
plt.ylabel("loss")
plt.title("Training loss")
plt.savefig(out_png, dpi=200)
print("✅ saved", out_png)
