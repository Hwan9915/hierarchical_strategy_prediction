"""
Compute generation & classification metrics for baseline models.
Outputs: summary.csv
Uses the same Metric class as eval_llm_generation_quality.py for consistency.
"""
import json, csv, sys
from pathlib import Path

import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

DST = Path(__file__).parent
sys.path.insert(0, str(DST.parent / "support_generation"))
from eval_llm_generation_quality import Metric  # reuse exact same metric class


def classification_f1(trues: list, preds: list) -> tuple:
    le = LabelEncoder()
    le.fit(trues + preds)
    t = le.transform(trues)
    p = le.transform(preds)
    macro = f1_score(t, p, average="macro", zero_division=0)
    weighted = f1_score(t, p, average="weighted", zero_division=0)
    return macro, weighted


def load_misc_kemi_emstremo(path: Path):
    with open(path) as f:
        data = json.load(f)
    refs, hyps, trues, preds = [], [], [], []
    for r in data:
        refs.append(str(r.get("reference_response", "") or ""))
        hyps.append(str(r.get("generated_response", "") or ""))
        trues.append(str(r.get("true_strategy", "") or ""))
        preds.append(str(r.get("predicted_strategy", "") or ""))
    return refs, hyps, trues, preds


def load_emodynamix(path: Path):
    with open(path) as f:
        data = json.load(f)
    trues, preds = [], []
    for r in data:
        trues.append(str(r.get("reference", "") or ""))
        preds.append(str(r.get("prediction", "") or ""))
    return trues, preds


def compute_gen(refs: list, hyps: list) -> tuple:
    m = Metric(hyps=hyps, refs=refs)
    result = m.close()
    return (
        result["bleu-1"],
        result["bleu-2"],
        result["rouge-l"],
        result["dist-1"],
        result["dist-2"],
        result["bert_f1"],
    )


rows = []
MODELS = ["MISC", "KEMI", "Emstremo"]

for name in MODELS:
    print(f"[{name}] computing...")
    refs, hyps, trues, preds = load_misc_kemi_emstremo(DST / f"detailed_{name}.json")
    b1, b2, rl, d1, d2, bs = compute_gen(refs, hyps)
    macro, weighted = classification_f1(trues, preds)
    rows.append({
        "model": name,
        "B-1": f"{b1:.2f}", "B-2": f"{b2:.2f}", "R-L": f"{rl:.2f}",
        "D-1": f"{d1:.2f}", "D-2": f"{d2:.2f}", "BERTScore": f"{bs:.2f}",
        "macro-F1": f"{macro:.4f}", "weighted-F1": f"{weighted:.4f}",
    })
    print(f"  B1={b1:.2f} B2={b2:.2f} RL={rl:.2f} D1={d1:.2f} D2={d2:.2f} BS={bs:.2f} mF1={macro:.4f} wF1={weighted:.4f}")

print("[EmoDynamiX] computing F1 only (no generation output)...")
trues, preds = load_emodynamix(DST / "detailed_EmoDynamiX.json")
macro, weighted = classification_f1(trues, preds)
rows.append({
    "model": "EmoDynamiX",
    "B-1": "-", "B-2": "-", "R-L": "-",
    "D-1": "-", "D-2": "-", "BERTScore": "-",
    "macro-F1": f"{macro:.4f}", "weighted-F1": f"{weighted:.4f}",
})
print(f"  mF1={macro:.4f} wF1={weighted:.4f}")

# Save CSV
out_csv = DST / "summary.csv"
fields = ["model", "B-1", "B-2", "R-L", "D-1", "D-2", "BERTScore", "macro-F1", "weighted-F1"]
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"\nSaved → {out_csv}")
print()
print(f"{'model':<12} {'B-1':>6} {'B-2':>6} {'R-L':>6} {'D-1':>6} {'D-2':>6} {'BERTScore':>10} {'macro-F1':>10} {'weighted-F1':>12}")
print("-" * 88)
for r in rows:
    print(f"{r['model']:<12} {r['B-1']:>6} {r['B-2']:>6} {r['R-L']:>6} {r['D-1']:>6} {r['D-2']:>6} {r['BERTScore']:>10} {r['macro-F1']:>10} {r['weighted-F1']:>12}")
