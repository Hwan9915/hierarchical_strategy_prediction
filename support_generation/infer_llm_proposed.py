import argparse
import os
import sys
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(__file__))

from common import (
    build_esconv_lookup,
    build_record,
    build_stage_aware_prompt,
    infer_stage_for_prompt,
    load_csv_inference_rows,
    normalize_dialogue_to_turns,
    pick_dialogue_context_from_esconv,
    resolve_llm_ids,
    run_inference_parallel,
    save_records,
    strategy_probs_module_from_csv_row,
    to_llm_alias,
    try_apply_chat_template,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "strategy_prediction" / "output"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results"

CLASSIFIERS = ["roberta", "distilbert"]
CSV_NAMES = {
    "roberta": "roberta-base_strategy_predictions.csv",
    "distilbert": "distilbert_strategy_predictions.csv",
}


def parse_args():
    ap = argparse.ArgumentParser(description="Support generation inference (proposed: roberta/distilbert).")
    ap.add_argument("--llm_id", type=str, default="")
    ap.add_argument("--all_llm", action="store_true")
    ap.add_argument("--classifier", type=str, choices=CLASSIFIERS, default="roberta")
    ap.add_argument("--all_classifier", action="store_true")
    ap.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    ap.add_argument("--api_key", type=str, default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--max_workers", type=int, default=25)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--use_esconv_input_dialog", type=int, default=1)
    return ap.parse_args()


def resolve_classifiers(args):
    return CLASSIFIERS[:] if args.all_classifier else [args.classifier]


def build_prompts(data_slice, tokenizer, esconv_lookup):
    prompts, meta_list = [], []
    for item in tqdm(data_slice, desc="Building prompts"):
        prediction = str(item.get("strategy_pred", "")).strip()
        prediction_id = item.get("strategy_pred_id", "")
        reference = str(item.get("strategy_true", "")).strip()
        reference_id = item.get("strategy_true_id", "")
        reference_response = str(item.get("label_context", "")).strip()

        situation = ""
        if esconv_lookup is not None:
            try:
                context_source, dialogue_context, situation = pick_dialogue_context_from_esconv(
                    {"reference_strategy": reference, "reference_response": reference_response},
                    esconv_lookup,
                )
            except ValueError:
                dialogue_context = str(item.get("context", "")).strip()
                context_source = "context"
                situation = ""
        else:
            dialogue_context = str(item.get("context", "")).strip()
            context_source = "context"

        dialogue_context = normalize_dialogue_to_turns(dialogue_context)
        probs = strategy_probs_module_from_csv_row(item)
        stage = infer_stage_for_prompt(item, prediction, prediction_id)
        prompt_text = build_stage_aware_prompt(dialogue_context, probs, stage, situation)

        chat_prompt = try_apply_chat_template(
            tokenizer, [{"role": "system", "content": prompt_text}]
        )
        prompts.append(chat_prompt)
        meta_list.append(
            {
                "dialogue_context": dialogue_context,
                "context_source": context_source,
                "situation": situation,
                "reference_response": reference_response,
                "prediction": prediction,
                "prediction_id": prediction_id,
                "reference": reference,
                "reference_id": reference_id,
            }
        )
    return prompts, meta_list


def run_combo(args, llm_id: str, classifier: str, esconv_lookup):
    data_dir = Path(args.data_dir).resolve()
    output_root = Path(args.output_root).resolve()
    csv_path = data_dir / CSV_NAMES[classifier]
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    data = load_csv_inference_rows(str(csv_path))
    print(f"[INFO] classifier={classifier} CSV: {csv_path} (rows={len(data)})")

    start = max(0, args.start)
    end = len(data) if args.end == -1 else min(args.end, len(data))
    data_slice = data[start:end]
    if not data_slice:
        raise ValueError("Empty slice — check --start/--end.")

    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    prompts, meta_list = build_prompts(data_slice, tokenizer, esconv_lookup)
    raw_outputs = run_inference_parallel(
        prompts, llm_id, client, args.temperature, args.max_tokens, args.max_workers
    )

    llm_alias = to_llm_alias(llm_id)
    out_dir = output_root / llm_alias / classifier
    records = [
        build_record(llm_alias, llm_id, classifier, meta_list[i], prompts[i], raw_outputs[i])
        for i in range(len(prompts))
    ]
    save_records(out_dir, llm_id, classifier, records)


def main():
    args = parse_args()
    llm_ids = resolve_llm_ids(args)
    classifiers = resolve_classifiers(args)

    print(f"[INFO] llm targets: {llm_ids}")
    print(f"[INFO] classifier targets: {classifiers}")
    print(f"[INFO] output_root={args.output_root}")

    esconv_lookup = None
    if args.use_esconv_input_dialog == 1:
        print("[INFO] Building ESConv lookup …")
        esconv_lookup = build_esconv_lookup()

    failures = []
    for llm_id in llm_ids:
        for classifier in classifiers:
            print(f"\n===== run: llm={llm_id} / classifier={classifier} =====")
            try:
                run_combo(args, llm_id, classifier, esconv_lookup)
            except Exception as e:
                print(f"[ERROR] llm={llm_id} classifier={classifier}: {e}")
                failures.append((llm_id, classifier, str(e)))

    if failures:
        print("\n[SUMMARY] failed combinations:")
        for llm_id, classifier, msg in failures:
            print(f"  - llm={llm_id}, classifier={classifier}, error={msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
