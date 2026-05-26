import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

sys.path.append(os.path.dirname(__file__))

from eval_llm_generation_quality import evaluate_rows

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "results"

BACKENDS = ("distilbert", "roberta")
LLM_ORDER = ("llama-70b-instruct", "qwen3-32b", "llama-8b-instruct", "qwen3-8b")
REQUIRED_COLUMNS = frozenset({"reference_response", "llm_parsed_response"})
OUTPUT_FIELDS = ("model", "bleu-1", "bleu-2", "rouge-l", "distinct-1", "distinct-2", "bert_score")
DEFAULT_OUTPUT_NAME = "generation_quality_summary_proposed.csv"


def discover_llm_dirs(root: Path) -> list:
    if not root.is_dir():
        return []
    return [
        p.name for p in sorted(root.iterdir())
        if p.is_dir() and not p.name.startswith(".") and any((p / b).is_dir() for b in BACKENDS)
    ]


def resolve_backend_csv(llm_dir: Path, backend: str):
    sub = llm_dir / backend
    if not sub.is_dir():
        return None
    csvs = sorted(sub.glob("*.csv"))
    return csvs[0] if csvs else None


def load_csv_rows(csv_path: Path, max_rows):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        fields = {str(h).strip() for h in reader.fieldnames if h}
        if not REQUIRED_COLUMNS.issubset(fields):
            return None
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(dict(row))
    return rows


def metric_to_row(model: str, gen: dict) -> dict:
    return {
        "model": model,
        "bleu-1": round(float(gen["bleu-1"]), 2),
        "bleu-2": round(float(gen["bleu-2"]), 2),
        "rouge-l": round(float(gen["rouge-l"]), 2),
        "distinct-1": round(float(gen["dist-1"]), 2),
        "distinct-2": round(float(gen["dist-2"]), 2),
        "bert_score": round(float(gen["bert_f1"]), 2),
    }


def write_summary_csv(out_path: Path, rows: list):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(OUTPUT_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in OUTPUT_FIELDS})


def run_one_llm(llm_dir: Path, max_rows, quiet: bool):
    out_rows, warnings = [], []
    for backend in BACKENDS:
        csv_path = resolve_backend_csv(llm_dir, backend)
        if csv_path is None:
            warnings.append(f"{llm_dir.name}/{backend}: CSV missing")
            continue
        rows = load_csv_rows(csv_path, max_rows)
        if rows is None or not rows:
            warnings.append(f"{csv_path}: unreadable or empty")
            continue
        if not quiet:
            print(f"    {backend}: {csv_path.name} ({len(rows)} rows)")
        gen = evaluate_rows(rows)
        if gen is None:
            warnings.append(f"{csv_path}: metric failure")
            continue
        out_rows.append(metric_to_row(backend, gen))
    return out_rows, warnings


def main():
    ap = argparse.ArgumentParser(description="Summarize generation quality for proposed backends.")
    ap.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    ap.add_argument("--llms", type=str, default="")
    ap.add_argument("--output-name", type=str, default=DEFAULT_OUTPUT_NAME)
    ap.add_argument("--master-csv", type=str, default="")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"[ERROR] root not found: {root}")
        sys.exit(1)

    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None
    if args.llms.strip():
        llm_names = [x.strip() for x in args.llms.split(",") if x.strip()]
    else:
        discovered = set(discover_llm_dirs(root))
        llm_names = [n for n in LLM_ORDER if n in discovered]
        llm_names += sorted(n for n in discovered if n not in LLM_ORDER)
    if not llm_names:
        print(f"[WARN] no LLM folders under: {root}")
        sys.exit(0)

    print(f"[INFO] backends={BACKENDS} root={root} llms={llm_names}")
    master_rows = []
    for name in llm_names:
        llm_dir = root / name
        if not llm_dir.is_dir():
            print(f"[WARN] missing dir: {llm_dir}")
            continue
        print(f"\n>>> LLM: {name}")
        rows, warns = run_one_llm(llm_dir, max_rows, args.quiet)
        for w in warns:
            print(f"    [warn] {w}")
        if not rows:
            continue
        out_csv = llm_dir / args.output_name
        write_summary_csv(out_csv, rows)
        print(f"    [OK] {out_csv}")
        master_rows.extend({"llm_alias": name, **r} for r in rows)

    if args.master_csv.strip() and master_rows:
        master_path = Path(args.master_csv).resolve()
        master_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ("llm_alias",) + OUTPUT_FIELDS
        with open(master_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(fields))
            writer.writeheader()
            writer.writerows(master_rows)
        print(f"\n[OK] master CSV: {master_path} ({len(master_rows)} rows)")


if __name__ == "__main__":
    main()
