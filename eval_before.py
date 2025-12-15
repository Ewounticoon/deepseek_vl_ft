import json
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

MODEL_NAME = "deepseek-ai/deepseek-vl-7b-base"
JSONL_PATH = "data/dataset.jsonl"

print("Loading processor...")
processor = VLChatProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cuda",
).eval()

# Load ONE sample
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    sample = json.loads(next(f))

user_text = sample["conversations"][0]["value"]
target = sample["conversations"][1]["value"]

# Build DeepSeek-VL conversation
user_text = user_text.replace("<image>", "<image_placeholder>")

conv = [
    {
        "role": "User",
        "content": user_text,
        "images": [f"data/{sample['image']}"],
    }
]

images = load_pil_images(conv)

inputs = processor(
    conversations=conv,
    images=images,
    force_batchify=True
).to(model.device)

with torch.no_grad():
    inputs_embeds = model.prepare_inputs_embeds(**inputs)
    output = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs.attention_mask,
        max_new_tokens=512,
        do_sample=False,
    )

prediction = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

print("\n================ MODEL OUTPUT (BEFORE FT) ================\n")
print(prediction[:2000])

print("\n================ GROUND TRUTH ================\n")
print(target[:2000])
