"""Ablation study evaluation.

Reads the three flattened CSVs in this directory:
    llm_only.csv, strategy_probability.csv, stage_information.csv
    (columns: index, llm_response, reference_response)

Uses evaluate_rows() from ../eval_llm_generation_quality.py — the same
function used for the main support_generation evaluation — so metrics are
computed identically (BLEU-1/2, ROUGE-L, Distinct-1/2, BERTScore).

Writes:
    generation_quality_summary.csv

Usage (from repo root):
    python3 support_generation/ablation_study/eval_ablation.py
"""
import csv
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # support_generation/

from eval_llm_generation_quality import evaluate_rows  # noqa: E402

MODELS = ("llm_only", "strategy_probability", "stage_information")
OUTPUT_FIELDS = ("model", "bleu-1", "bleu-2", "rouge-l", "distinct-1", "distinct-2", "bert_score")
KEY_MAP = {
    "bleu-1": "bleu-1",
    "bleu-2": "bleu-2",
    "rouge-l": "rouge-l",
    "distinct-1": "dist-1",
    "distinct-2": "dist-2",
    "bert_score": "bert_f1",
}


def load_rows(csv_path: Path) -> list:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            # evaluate_rows expects `llm_parsed_response`
            row["llm_parsed_response"] = row.get("llm_response", "")
            rows.append(row)
    return rows


def main():
    summary_rows = []
    for model in MODELS:
        print(f"[eval] {model} ...", flush=True)
        rows = load_rows(HERE / f"{model}.csv")
        result = evaluate_rows(rows)
        row = {"model": model, **{c: round(float(result[KEY_MAP[c]]), 2) for c in OUTPUT_FIELDS if c != "model"}}
        summary_rows.append(row)
        print(f"       bleu-1={row['bleu-1']}  rouge-l={row['rouge-l']}  bert_score={row['bert_score']}")

    out_path = HERE / "generation_quality_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(OUTPUT_FIELDS))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[saved] {out_path}")
    print("\n" + ",".join(OUTPUT_FIELDS))
    for r in summary_rows:
        print(",".join(str(r[c]) for c in OUTPUT_FIELDS))


if __name__ == "__main__":
    main()
