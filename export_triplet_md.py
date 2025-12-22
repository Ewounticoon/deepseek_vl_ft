#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON invalide à la ligne {lineno} dans {path}: {e}") from e


def load_by_i(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path.resolve()}")
    d = {}
    n_lines = 0
    for lineno, r in iter_jsonl(path):
        n_lines += 1
        if "i" not in r:
            raise RuntimeError(f"{path}: ligne {lineno}: champ 'i' manquant. Clés: {list(r.keys())}")
        d[r["i"]] = r
    return d, n_lines


def safe_name(i_value) -> str:
    s = str(i_value)
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s).strip("_")
    return s or "no_id"


def write_md(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "" if content is None else content
    if not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8")


def metrics_comment(r: dict) -> str:
    parts = []
    for k in ("cer", "wer", "rougeL", "chrf"):
        v = r.get(k, None)
        if isinstance(v, (int, float)):
            parts.append(f"{k}={v:.4f}")
    return ("<!-- " + " ".join(parts) + " -->\n\n") if parts else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", default=r"outputs/eval_after_ft/predictions.jsonl",
                    help="JSONL after fine-tune")
    ap.add_argument("--baseline", default=r"outputs/baseline/predictions.jsonl",
                    help="JSONL baseline")
    ap.add_argument("--out", default=r"outputs/md_exports",
                    help="Dossier de sortie")
    ap.add_argument("--n", type=int, default=None,
                    help="Limiter à N IDs")
    ap.add_argument("--mode", choices=["after", "intersection", "union"], default="after",
                    help="Quels IDs exporter: after (=IDs présents dans after), intersection (=communs), union (=tous)")
    ap.add_argument("--with-metrics-header", action="store_true",
                    help="Ajoute un header métriques")
    args = ap.parse_args()

    after_path = Path(args.after)
    baseline_path = Path(args.baseline)
    out_dir = Path(args.out)

    after_by_i, after_lines = load_by_i(after_path)
    base_by_i, base_lines = load_by_i(baseline_path)

    print(f"Loaded after:    {after_path}  (non-empty json lines: {after_lines}, unique i: {len(after_by_i)})")
    print(f"Loaded baseline: {baseline_path}  (non-empty json lines: {base_lines}, unique i: {len(base_by_i)})")

    if len(after_by_i) == 0:
        print("⚠️  Aucun sample trouvé dans le fichier after. Vérifie le chemin, ou si le fichier est vide.")
        return

    md_expected = out_dir / "md_expected"
    md_after    = out_dir / "md_after"
    md_base     = out_dir / "md_baseline"

    after_ids = set(after_by_i.keys())
    base_ids = set(base_by_i.keys())

    if args.mode == "after":
        ids = list(after_by_i.keys())  # garde l’ordre du fichier after
    elif args.mode == "intersection":
        ids = [i for i in after_by_i.keys() if i in base_ids]
    else:  # union
        # ordre: after d'abord, puis ceux qui manquent depuis baseline
        ids = list(after_by_i.keys()) + [i for i in base_by_i.keys() if i not in after_ids]

    if args.n is not None:
        ids = ids[:args.n]

    wrote_expected = wrote_after = wrote_baseline = 0
    missing_gt = 0
    missing_baseline = 0

    for i in ids:
        sid = safe_name(i)

        ra = after_by_i.get(i)
        rb = base_by_i.get(i)

        # Expected: prend gt depuis after sinon baseline
        gt = (ra.get("gt") if ra else None) or (rb.get("gt") if rb else None)
        if gt is None:
            missing_gt += 1
        else:
            write_md(md_expected / f"{sid}.md", gt)
            wrote_expected += 1

        # After
        if ra is not None:
            pred_after = ra.get("pred", "")
            if args.with_metrics_header:
                pred_after = f"<!-- sample={i} | AFTER -->\n\n" + metrics_comment(ra) + pred_after
            write_md(md_after / f"{sid}.md", pred_after)
            wrote_after += 1

        # Baseline
        if rb is None:
            missing_baseline += 1
        else:
            pred_base = rb.get("pred", "")
            if args.with_metrics_header:
                pred_base = f"<!-- sample={i} | BASELINE -->\n\n" + metrics_comment(rb) + pred_base
            write_md(md_base / f"{sid}.md", pred_base)
            wrote_baseline += 1

    print("\nOK")
    print(f"Exported IDs:     {len(ids)} (mode={args.mode})")
    print(f"Wrote expected:   {wrote_expected} (missing GT for {missing_gt})")
    print(f"Wrote after:      {wrote_after}")
    print(f"Wrote baseline:   {wrote_baseline} (missing baseline for {missing_baseline})")
    print("Output folders:")
    print(" -", md_expected)
    print(" -", md_after)
    print(" -", md_base)


if __name__ == "__main__":
    main()
