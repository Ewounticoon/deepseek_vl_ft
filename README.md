python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# IMPORTANT Qwen2.5-VL: transformers from source
pip install -U git+https://github.com/huggingface/transformers accelerate
# sinon risque d'erreur KeyError: 'qwen2_5_vl' :contentReference[oaicite:3]{index=3}

playwright install chromium


python prepare_dataset.py --jsonl data/dataset.jsonl --out data/hf_dataset --data_root data


python eval.py --model Qwen/Qwen2.5-VL-7B-Instruct --dataset data/hf_dataset --out outputs/baseline --n_samples 30 --max_new_tokens 4096


python train.py --config configs/qwen25vl_lora.yaml


python eval.py --model outputs/finetuned --dataset data/hf_dataset --out outputs/finetuned --n_samples 30 --max_new_tokens 4096


python plot_loss.py --log outputs/finetuned/trainer_log_history.json --out outputs/finetuned/loss_curve.png


python make_report.py --before outputs/baseline/predictions.jsonl --after outputs/finetuned/predictions.jsonl --out outputs/report
