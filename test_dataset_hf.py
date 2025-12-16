from datasets import load_from_disk

ds = load_from_disk("data/hf_dataset")
print(ds)
print(ds["train"][0].keys())
print(ds["train"][0]["messages"])

print(len(ds["train"]) + len(ds["eval"]))
ids = [x["id"] for x in ds["train"]] + [x["id"] for x in ds["eval"]]
print(len(ids), len(set(ids)))


import os

missing = []
for split in ["train", "eval"]:
    for s in ds[split]:
        path = s["messages"][0]["content"][0]["image"]
        # si chemin relatif
        if not path.startswith("file://"):
            path = os.path.join("data", path)
        else:
            path = path.replace("file://", "")
        if not os.path.exists(path):
            missing.append(path)

print("Missing images:", len(missing))
