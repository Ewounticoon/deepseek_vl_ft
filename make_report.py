# make_report.py
import argparse, json
from pathlib import Path
from playwright.sync_api import sync_playwright

def load_jsonl(p):
    return [json.loads(l) for l in open(p, "r", encoding="utf-8")]

def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

ap = argparse.ArgumentParser()
ap.add_argument("--before", required=True)
ap.add_argument("--after", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=8)
args = ap.parse_args()

before = load_jsonl(args.before)
after = load_jsonl(args.after)
after_by_i = {r["i"]: r for r in after}
common = [r for r in before if r["i"] in after_by_i][:args.n]

out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
report = out_dir / "report.html"

blocks = []
for r in common:
    ra = after_by_i[r["i"]]
    blocks.append(f"""
    <h3>Sample {r["i"]}</h3>
    <div class="grid">
      <div><h4>GT</h4><pre>{esc(r["gt"])}</pre></div>
      <div><h4>Before</h4><pre>{esc(r["pred"])}</pre></div>
      <div><h4>After</h4><pre>{esc(ra["pred"])}</pre></div>
    </div>
    <p><b>Before</b> CER={r["cer"]:.3f} WER={r["wer"]:.3f} ROUGE-L={r["rougeL"]:.3f} chrF={r["chrf"]:.3f}<br/>
       <b>After</b>  CER={ra["cer"]:.3f} WER={ra["wer"]:.3f} ROUGE-L={ra["rougeL"]:.3f} chrF={ra["chrf"]:.3f}</p>
    <hr/>
    """)

html = f"""
<html><head><meta charset="utf-8"/>
<style>
body{{font-family:Arial;margin:24px}}
pre{{background:#f6f6f6;padding:12px;white-space:pre-wrap}}
.grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}
</style></head><body>
<h1>Qwen2.5-VL — Before/After Fine-tune</h1>
{''.join(blocks)}
</body></html>
"""
report.write_text(html, encoding="utf-8")

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1600, "height": 900})
    page.goto(report.resolve().as_uri())
    page.wait_for_timeout(300)
    page.screenshot(path=str(out_dir / "report.png"), full_page=True)
    b.close()

print("Wrote:", report)
print("Screenshot:", out_dir / "report.png")
