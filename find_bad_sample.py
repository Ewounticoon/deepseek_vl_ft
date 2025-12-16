# find_bad_sample.py
import argparse, copy
from pathlib import Path
from tqdm import tqdm

import torch
from datasets import load_from_disk
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


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
                    continue
                new_content.append({"type": "text", "text": c["text"]})
            else:
                new_content.append(c)
        msg["content"] = new_content
    return m


def resolve_image_path(image_field: str, data_root: Path):
    # image_field can be:
    #  - "file:///abs/path.png"
    #  - "images/xxx.png"
    #  - "/abs/path.png"
    s = str(image_field)
    if s.startswith("file://"):
        s = s.replace("file://", "")
    p = Path(s)
    if not p.is_absolute():
        # most common: images/xxx.png stored relative to dataset root
        p = (data_root / p).resolve()
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="data/hf_dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--data_root", default="data", help="root folder that contains images/")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()

    ds = load_from_disk(args.dataset)[args.split]
    processor = AutoProcessor.from_pretrained(args.model)

    stats = []
    for i in tqdm(range(len(ds)), desc=f"Scanning {args.split}"):
        ex = ds[i]
        msgs = sanitize_messages_strict(ex["messages"])

        # Fix image path inside messages (file://C:\... etc)
        # We rewrite the image field so process_vision_info can load it correctly on the server
        for msg in msgs:
            for c in msg.get("content", []):
                if c.get("type") == "image" and isinstance(c.get("image"), str):
                    c["image"] = resolve_image_path(c["image"], data_root)

        # Build full supervised text (user+assistant), no generation prompt
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

        # Extract vision inputs (loads image)
        image_inputs, video_inputs = process_vision_info(msgs)

        # Tokenize with multimodal processor
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            truncation=False,  # IMPORTANT: we want the real length
            return_tensors="pt",
        )

        n_tok = int(inputs["input_ids"].shape[1])
        stats.append((n_tok, i, ex.get("id"), ex.get("image_path")))

    stats.sort(reverse=True, key=lambda x: x[0])

    print(f"\nTop {args.topk} longest samples by *model* tokens (text+image):")
    for n_tok, idx, _id, imgp in stats[:args.topk]:
        print(f"tokens={n_tok:6d} | idx={idx:4d} | id={_id} | image_path={imgp}")

    print("\nShortest example:", stats[-1])
    print("Longest example :", stats[0])


if __name__ == "__main__":
    main()
