import os
import json
from pprint import pprint
import yaml

from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images


# ----------------------------
# Helpers (copiés du train.py)
# ----------------------------
def resolve_image_path(image_root: str, image_field: str, strip_prefix: str | None):
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

def to_deepseek_conversation(sample, image_root, strip_prefix, system_prompt=None):
    user_text, target_md = user_assistant_from_conversations(sample["conversations"])
    img_path = resolve_image_path(image_root, sample["image"], strip_prefix)

    # 1) Si ton JSONL contient déjà <image>, on le remplace par <image_placeholder>
    if "<image>" in user_text:
        user_content = user_text.replace("<image>", "<image_placeholder>")
    else:
        user_content = "<image_placeholder>\n" + user_text

    # 2) Optionnel: injecter un system prompt
    if system_prompt:
        user_content = system_prompt.strip() + "\n\n" + user_content

    return [
        {"role": "User", "content": user_content, "images": [img_path]},
        {"role": "Assistant", "content": target_md},
    ]


# ----------------------------
# Main test
# ----------------------------
def main():
    # 1) Load config
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    jsonl_path = cfg["dataset"]["jsonl_path"]
    image_root = cfg["dataset"]["image_root"]
    strip_prefix = cfg["dataset"].get("strip_image_prefix", None)

    system_prompt = None
    if cfg["dataset"].get("add_system_prompt", False):
        system_prompt = cfg["dataset"].get("system_prompt", None)

    print("=== CONFIG ===")
    print("jsonl_path:", jsonl_path)
    print("image_root:", image_root)
    print("strip_prefix:", strip_prefix)
    print("system_prompt:", "YES" if system_prompt else "NO")
    print()

    # 2) Read first sample
    assert os.path.exists(jsonl_path), f"JSONL introuvable: {jsonl_path}"
    assert os.path.exists(image_root), f"Image root introuvable: {image_root}"

    with open(jsonl_path, "r", encoding="utf-8") as f:
        sample = json.loads(next(f))

    print("=== SAMPLE KEYS ===")
    print(sample.keys())
    print("image field:", sample.get("image"))
    print("conversations type:", type(sample.get("conversations")))
    print("conversations len:", len(sample.get("conversations", [])))
    print()

    # 3) Convert to deepseek conversation format
    conv = to_deepseek_conversation(sample, image_root, strip_prefix, system_prompt)
    print("=== CONVERTED CONVERSATION (preview) ===")
    pprint({
        "user_role": conv[0]["role"],
        "user_content_preview": conv[0]["content"][:200] + "...",
        "user_images": conv[0]["images"],
        "assistant_role": conv[1]["role"],
        "assistant_content_preview": conv[1]["content"][:200] + "...",
    })
    print()

    # 4) Check resolved image path exists
    img_path = conv[0]["images"][0]
    print("=== IMAGE PATH CHECK ===")
    print("resolved img_path:", img_path)
    print("exists:", os.path.exists(img_path))
    assert os.path.exists(img_path), f"Image introuvable: {img_path}"
    print()

    # 5) Load PIL images via DeepSeek util
    print("=== PIL LOAD CHECK ===")
    pil_images = load_pil_images(conv)
    print("nb PIL images:", len(pil_images))
    print("PIL[0] size:", pil_images[0].size)
    print()

    # 6) Processor test (no model load)
    print("=== PROCESSOR CHECK ===")
    processor = VLChatProcessor.from_pretrained(cfg["model"]["name"])
    out = processor(
        conversations=conv,
        images=pil_images,
        force_batchify=True,
    )
    print("processor output keys:", out.keys())
    print("input_ids shape:", tuple(out.input_ids.shape))
    print("attention_mask shape:", tuple(out.attention_mask.shape))
    print()

    # 7) Decode a tiny preview to be sure tokens make sense
    tok = processor.tokenizer
    decoded = tok.decode(out.input_ids[0].tolist()[:200], skip_special_tokens=False)
    print("=== TOKEN DECODE PREVIEW (first ~200 tokens) ===")
    print(decoded[:1200])
    print()

    print("✅ Dataset + images + processor OK. Ready to move to RunPod.")


if __name__ == "__main__":
    main()
