# make_report.py
import argparse, json, re
from pathlib import Path
from difflib import SequenceMatcher
from playwright.sync_api import sync_playwright

def load_jsonl(p):
    return [json.loads(l) for l in open(p, "r", encoding="utf-8")]

def esc(s: str) -> str:
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

_word_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)
def tokenize_words(s: str):
    # split en mots + ponctuation (utile pour WER-style)
    return _word_re.findall(s or "")

def diff_ops(a_tokens, b_tokens):
    """
    Retourne les opcodes (equal/replace/delete/insert) de a->b
    """
    return SequenceMatcher(a=a_tokens, b=b_tokens).get_opcodes()

def render_diff_table(pred: str, gt: str, title: str):
    """
    Table HTML: ligne 'Pred' et ligne 'GT' avec surlignage des différences.
    - replace/delete côté Pred en rouge
    - replace/insert côté GT en vert
    """
    a = tokenize_words(pred)
    b = tokenize_words(gt)
    ops = diff_ops(a, b)

    pred_html = []
    gt_html = []

    # helpers
    def join_tokens(toks):
        # recolle avec espace "raisonnable"
        out = []
        for i, tok in enumerate(toks):
            if i > 0 and re.match(r"\w", tok) and re.match(r"\w", toks[i-1]):
                out.append(" ")
            out.append(tok)
        return "".join(out)

    for tag, i1, i2, j1, j2 in ops:
        a_seg = join_tokens(a[i1:i2])
        b_seg = join_tokens(b[j1:j2])

        if tag == "equal":
            pred_html.append(esc(a_seg))
            gt_html.append(esc(b_seg))
        elif tag == "replace":
            pred_html.append(f'<span class="bad">{esc(a_seg)}</span>')
            gt_html.append(f'<span class="good">{esc(b_seg)}</span>')
        elif tag == "delete":
            pred_html.append(f'<span class="bad">{esc(a_seg)}</span>')
            # rien côté GT
        elif tag == "insert":
            # rien côté Pred
            gt_html.append(f'<span class="good">{esc(b_seg)}</span>')

    table = f"""
    <div class="diffblock">
      <h4>{esc(title)}</h4>
      <table class="difftable">
        <tr><td class="label">Pred</td><td class="diffcell">{''.join(pred_html)}</td></tr>
        <tr><td class="label">GT</td><td class="diffcell">{''.join(gt_html)}</td></tr>
      </table>
    </div>
    """
    return table

def word_error_set(pred: str, gt: str):
    """
    Approx: ensemble des "unités" qui diffèrent (pour compter corrected/regressed).
    On encode les segments non-equal côté pred et côté gt avec des tags,
    juste pour compter de manière stable (pas parfait mais très parlant).
    """
    a = tokenize_words(pred)
    b = tokenize_words(gt)
    ops = diff_ops(a, b)

    errs = set()
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        a_seg = " ".join(a[i1:i2])
        b_seg = " ".join(b[j1:j2])
        errs.add((tag, a_seg, b_seg))
    return errs

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

    # sets d'erreurs (approx) pour dire "corrigé / régressé"
    eb = word_error_set(r["pred"], r["gt"])
    ea = word_error_set(ra["pred"], ra["gt"])
    corrected = len(eb - ea)
    regressed = len(ea - eb)

    # label simple
    status = "✅ Fully correct" if (ra["pred"] == r["gt"]) else (
        "📈 Improved" if ra["wer"] < r["wer"] else ("📉 Worse" if ra["wer"] > r["wer"] else "➖ Mixed")
    )

    blocks.append(f"""
    <h3>Sample {r["i"]} — {status}</h3>

    <div class="grid">
      <div><h4>GT</h4><pre>{esc(r["gt"])}</pre></div>
      <div><h4>Before</h4><pre>{esc(r["pred"])}</pre></div>
      <div><h4>After</h4><pre>{esc(ra["pred"])}</pre></div>
    </div>

    <p class="metrics">
      <b>Before</b> CER={r["cer"]:.3f} WER={r["wer"]:.3f} ROUGE-L={r["rougeL"]:.3f} chrF={r["chrf"]:.3f}<br/>
      <b>After</b>&nbsp;&nbsp; CER={ra["cer"]:.3f} WER={ra["wer"]:.3f} ROUGE-L={ra["rougeL"]:.3f} chrF={ra["chrf"]:.3f}
    </p>

    <p class="delta">
      <b>Pointed errors:</b> corrected={corrected} &nbsp; introduced={regressed} &nbsp;
      (net {corrected - regressed:+d})
    </p>

    {render_diff_table(r["pred"], r["gt"], "Diff: BEFORE ↔ GT")}
    {render_diff_table(ra["pred"], r["gt"], "Diff: AFTER ↔ GT")}

    <hr/>
    """)

html = f"""
<html><head><meta charset="utf-8"/>
<style>
body{{font-family:Arial;margin:24px}}
pre{{background:#f6f6f6;padding:12px;white-space:pre-wrap;border-radius:10px}}
.grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}
.metrics{{margin-top:10px}}
.delta{{margin-top:6px}}
.diffblock{{margin-top:10px}}
.difftable{{border-collapse:collapse;width:100%}}
.difftable .label{{width:70px;font-weight:bold;vertical-align:top;padding:8px 10px;color:#333}}
.difftable .diffcell{{padding:8px 10px;background:#fafafa;border-radius:10px}}
.bad{{background:#ffd6d6;padding:1px 2px;border-radius:4px}}
.good{{background:#d9ffe1;padding:1px 2px;border-radius:4px}}
</style></head><body>
<h1>Qwen2.5-VL — Before/After Fine-tune (with pointed errors)</h1>
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
