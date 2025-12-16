python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# IMPORTANT Qwen2.5-VL: transformers from source
pip install -U git+https://github.com/huggingface/transformers accelerate
# sinon risque d'erreur KeyError: 'qwen2_5_vl' :contentReference[oaicite:3]{index=3}

playwright install chromium
