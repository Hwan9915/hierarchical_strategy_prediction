"""Strategy cascade inference: comforting_predictor uses only the current-turn utterance,
other modules use the full cumulative context. Entry point for scripts/infer.sh."""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm

import strategy_prediction.module as pipeline_module
import strategy_prediction.dataloader_prediction as dataloader
from strategy_prediction.config import MODULE_DICT, STAGE_KEYWORDS, STRATEGY_KEYWORDS
from strategy_prediction.model_settings import (
    DEFAULT_MODEL_DIR,
    DEFAULT_OUT_BASE,
    get_ckpt_path,
    get_output_file,
    resolve_encoder,
)
from strategy_prediction.utils import setup_log_file


STRATEGY_PROB_COLS = ["prob_qu", "prob_rp", "prob_rf", "prob_sd", "prob_ar", "prob_ps", "prob_in"]
STAGE_CLASS_NAMES = ["Exploration", "Comforting", "Action", "Others"]
STAGE_PROB_COLS = ["prob_ex", "prob_co", "prob_ac", "prob_ot"]


def _write_csv(out_dir: str, encoder_name: str, filename: str, rows: list) -> str:
    out_path = get_output_file(out_dir, encoder_name, filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    return str(out_path)


def save_strategy_predictions_csv(out_dir: str, encoder_name: str, rows: list) -> str:
    return _write_csv(out_dir, encoder_name, "strategy_prediction_predictions.csv", rows)


def save_stage_predictions_csv(out_dir: str, encoder_name: str, rows: list) -> str:
    return _write_csv(out_dir, encoder_name, "stage_identification_predictions.csv", rows)


def save_stage_prf_csv(out_dir: str, turn: int, encoder_name: str, y_true: list, y_pred: list) -> str:
    label_to_idx = {name: i for i, name in enumerate(STAGE_CLASS_NAMES)}
    y_true_idx = [label_to_idx[t] for t in y_true]
    y_pred_idx = [label_to_idx[p] for p in y_pred]
    labels = list(range(len(STAGE_CLASS_NAMES)))
    prec, rec, f1_cls, _ = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, labels=labels, zero_division=0
    )

    rows = []
    for i, cls in enumerate(STAGE_CLASS_NAMES):
        rows.append({
            "turn": turn,
            "encoder": encoder_name,
            "class": cls,
            "precision": round(prec[i], 4),
            "recall": round(rec[i], 4),
            "f1": round(f1_cls[i], 4),
        })
    return _write_csv(out_dir, encoder_name, "stage_identification_prf.csv", rows)


def save_strategy_metrics_csv(
    out_dir: str,
    turn: int,
    encoder_name: str,
    macro_f1: float,
    weighted_f1: float,
    accuracy: float,
    total_params: float,
) -> str:
    rows = [{
        "turn": turn,
        "encoder": encoder_name,
        "m_f1": round(macro_f1, 4),
        "w_f1": round(weighted_f1, 4),
        "acc": round(accuracy, 4),
        "total_params_M": round(total_params, 2),
    }]
    return _write_csv(out_dir, encoder_name, "strategy_prediction_metrics.csv", rows)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_batch_size", type=int, default=1)
    p.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    p.add_argument("--encoder_key", type=str, default="distilbert")
    p.add_argument("--encoder_name", type=str, default="")
    p.add_argument("--out_dir", type=str, default="",
                   help="Flat output directory. Files are named {encoder_short}_{basename}.")
    p.add_argument("--save_stage_predictions", action="store_true", default=False,
                   help="Also dump per-row stage_identification_predictions.csv (off by default).")
    p.add_argument("--log_file", type=str, default="")
    return p


def main() -> None:
    args = get_parser().parse_args()
    setup_log_file(args.log_file)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    pl.seed_everything(args.seed, workers=True)
    _, resolved_encoder = resolve_encoder(args.encoder_key, args.encoder_name)
    args.encoder_name = resolved_encoder

    if not args.out_dir:
        args.out_dir = DEFAULT_OUT_BASE
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_labels, _, label_mapping_dict, sampling = MODULE_DICT["test"]
    _, _, test_loader = dataloader.create_dataloader(
        sampling=sampling,
        num_labels=num_labels,
        label_mapping_dict=label_mapping_dict,
        train_batch_size=32,
        valid_batch_size=16,
        test_batch_size=args.test_batch_size,
        turns=5,
        is_train=0,
        is_llm=1,
    )

    all_modules = [
        "exploration_support_identifier",
        "comforting_action_identifier",
        "exploration_predictor",
        "comforting_predictor",
        "action_predictor",
    ]

    paths = [get_ckpt_path(args.model_dir, m, args.encoder_name) for m in all_modules]
    models = [
        pipeline_module.TransformerClassifier.load_from_checkpoint(
            checkpoint_path=str(p),
            encoder_name=args.encoder_name,
        ).to(device)
        for p in paths
    ]
    for m in models:
        m.eval()

    model_params = [sum(p.numel() for p in m.parameters()) / 1e6 for m in models]
    total_params_all = sum(model_params)

    y_true, y_pred = [], []
    active_params = []
    stage_y_true, stage_y_pred = [], []
    stage_active_params = []
    prediction_rows, stage_prediction_rows = [], []
    stage_to_idx = {name: i for i, name in enumerate(STAGE_CLASS_NAMES)}

    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Strategy inference"):
            context_t1, context_t2 = batch["context_t1"], batch["context_t2"]
            label_context_list = batch.get("label_context", None)
            full_ctx = f"{context_t1[0]} {context_t2[0]}".strip()
            current_turn_ctx = context_t2[0].strip()
            cur_label_context = ""
            if label_context_list is not None and len(label_context_list) > 0:
                cur_label_context = label_context_list[0]

            probs = {c: -1.0 for c in STRATEGY_PROB_COLS}
            stage_probs = {c: -1.0 for c in STAGE_PROB_COLS}

            current_active = model_params[0]
            emb0 = models[0]([full_ctx])
            logits0 = models[0].classifier(emb0[0])
            probs0 = torch.softmax(logits0, dim=-1).cpu().numpy().flatten()
            ex_prob = round(float(probs0[0]), 4)
            su_prob = round(float(probs0[1]), 4)
            pred0 = int(probs0.argmax())

            if pred0 == 0:
                stage_probs["prob_ex"] = ex_prob
                emb2 = models[2]([full_ctx])
                logits2 = models[2].classifier(emb2[0])
                probs2 = torch.softmax(logits2, dim=-1).cpu().numpy().flatten()
                pred_strategy = int(probs2.argmax())
                probs["prob_qu"] = round(float(probs2[0]), 4)
                probs["prob_rp"] = round(float(probs2[1]), 4)
                current_active += model_params[2]
            else:
                emb1 = models[1]([full_ctx])
                logits1 = models[1].classifier(emb1[0])
                probs1 = torch.softmax(logits1, dim=-1).cpu().numpy().flatten()
                pred1 = int(probs1.argmax())
                co_prob = round(float(probs1[0]), 4)
                ac_prob = round(float(probs1[1]), 4)
                ot_prob = round(float(probs1[2]), 4)
                stage_probs["prob_co"] = co_prob
                stage_probs["prob_ac"] = ac_prob
                stage_probs["prob_ot"] = ot_prob
                current_active += model_params[1]

                if pred1 == 0:
                    emb3 = models[3]([current_turn_ctx])
                    logits3 = models[3].classifier(emb3[0])
                    probs3 = torch.softmax(logits3, dim=-1).cpu().numpy().flatten()
                    pred_strategy = int(probs3.argmax()) + 2
                    probs["prob_rf"] = round(float(probs3[0]), 4)
                    probs["prob_sd"] = round(float(probs3[1]), 4)
                    probs["prob_ar"] = round(float(probs3[2]), 4)
                    current_active += model_params[3]
                elif pred1 == 1:
                    emb4 = models[4]([full_ctx])
                    logits4 = models[4].classifier(emb4[0])
                    probs4 = torch.softmax(logits4, dim=-1).cpu().numpy().flatten()
                    pred_strategy = int(probs4.argmax()) + 5
                    probs["prob_ps"] = round(float(probs4[0]), 4)
                    probs["prob_in"] = round(float(probs4[1]), 4)
                    current_active += model_params[4]
                else:
                    pred_strategy = 7

            raw_label = batch["label"][0]
            true_idx = int(raw_label.item() if hasattr(raw_label, "item") else raw_label)
            y_true.append(STRATEGY_KEYWORDS[true_idx])
            y_pred.append(STRATEGY_KEYWORDS[pred_strategy])
            active_params.append(current_active)

            stage_active = model_params[0] + (model_params[1] if pred0 != 0 else 0.0)
            stage_active_params.append(stage_active)
            stage_true_str = STAGE_KEYWORDS[true_idx]
            stage_pred_str = STAGE_KEYWORDS[pred_strategy]
            stage_y_true.append(stage_true_str)
            stage_y_pred.append(stage_pred_str)

            stage_true_idx = stage_to_idx[stage_true_str]
            stage_pred_idx = stage_to_idx[stage_pred_str]
            strategy_true_str = STRATEGY_KEYWORDS[true_idx]
            strategy_pred_str = STRATEGY_KEYWORDS[pred_strategy]

            row = {
                "context": full_ctx,
                "label_context": cur_label_context,
                "stage_pred_id": stage_pred_idx,
                "stage_true_id": stage_true_idx,
                "stage_pred": stage_pred_str,
                "stage_true": stage_true_str,
                "strategy_pred_id": pred_strategy,
                "strategy_true_id": true_idx,
                "strategy_pred": strategy_pred_str,
                "strategy_true": strategy_true_str,
                "prob_ex": ex_prob,
                "prob_su": su_prob,
                "prob_co": stage_probs["prob_co"],
                "prob_ac": stage_probs["prob_ac"],
                "prob_ot": stage_probs["prob_ot"],
            }
            for c in STRATEGY_PROB_COLS:
                row[c] = probs[c]
            prediction_rows.append(row)

            stage_row = {
                "context": full_ctx,
                "y_pred": stage_pred_idx,
                "y_true": stage_true_idx,
            }
            for c in STAGE_PROB_COLS:
                stage_row[c] = stage_probs[c]
            stage_prediction_rows.append(stage_row)

    elapsed_time = time.time() - start_time
    total_samples = len(y_true)
    latency_ms = (elapsed_time / total_samples) * 1000 if total_samples > 0 else 0.0

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    active_params_mean = (sum(active_params) / len(active_params)) if active_params else 0.0
    sparsity_pct = (1 - (active_params_mean / total_params_all)) * 100 if total_params_all > 0 else 0.0

    stage_total_params = model_params[0] + model_params[1]
    stage_acc = accuracy_score(stage_y_true, stage_y_pred)
    stage_macro_f1 = f1_score(stage_y_true, stage_y_pred, average="macro", zero_division=0)
    stage_weighted_f1 = f1_score(stage_y_true, stage_y_pred, average="weighted", zero_division=0)
    stage_active_mean = (sum(stage_active_params) / len(stage_active_params)) if stage_active_params else 0.0

    pred_csv = save_strategy_predictions_csv(args.out_dir, args.encoder_name, prediction_rows)
    print(f"[Strategy][CSV] predictions saved: {pred_csv}")
    if args.save_stage_predictions:
        stage_pred_csv = save_stage_predictions_csv(args.out_dir, args.encoder_name, stage_prediction_rows)
        print(f"[Stage][CSV] predictions saved: {stage_pred_csv}")
    else:
        print("[Stage][CSV] predictions skipped (use --save_stage_predictions to enable)")
    stage_prf_csv = save_stage_prf_csv(args.out_dir, 5, args.encoder_name, stage_y_true, stage_y_pred)
    print(f"[Stage][CSV] prf saved: {stage_prf_csv}")
    metrics_csv = save_strategy_metrics_csv(
        args.out_dir, 5, args.encoder_name,
        macro_f1, weighted_f1, accuracy, total_params_all,
    )
    print(f"[Strategy][CSV] metrics saved: {metrics_csv}")

    stage_metrics_text = (
        f"\n===== Stage Classification Metrics =====\n"
        f"Accuracy     : {stage_acc:.4f}\n"
        f"Macro F1     : {stage_macro_f1:.4f}\n"
        f"Weighted F1  : {stage_weighted_f1:.4f}\n"
        f"===== Parameter & Efficiency Analysis =====\n"
        f"Total Params : {stage_total_params:.2f} M\n"
        f"Active Params: {stage_active_mean:.2f} M (Average per sample)\n"
        f"=============================================\n"
    )
    strategy_metrics_text = (
        f"\n===== Strategy Classification Metrics =====\n"
        f"Macro F1     : {macro_f1:.4f}\n"
        f"Weighted F1  : {weighted_f1:.4f}\n"
        f"===== Parameter & Efficiency Analysis =====\n"
        f"Total Params : {total_params_all:.2f} M\n"
        f"Active Params: {active_params_mean:.2f} M (Average per sample)\n"
        f"Sparsity     : {sparsity_pct:.1f}% parameters ignored per inference\n"
        f"Latency      : {latency_ms:.2f} ms / sample (Total time: {elapsed_time:.2f}s for {total_samples} samples)\n"
        f"=============================================\n"
    )
    print(stage_metrics_text, end="")
    print(strategy_metrics_text, end="")

    log_path = get_output_file(args.out_dir, args.encoder_name, "throughput.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(stage_metrics_text)
        f.write(strategy_metrics_text)
    print(f"[Strategy][LOG] throughput metrics saved: {log_path}")


if __name__ == "__main__":
    main()
