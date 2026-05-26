import ast
import csv
import json
import math
import re
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from ftfy import fix_text
from openai import OpenAI
from tqdm import tqdm

from prompt import (
    HIE_DIFINITON_STAGE_STRATEGY_PROMPT,
    PROMPT_HIE_PROB_HISTORY,
    PROMPT_OTHERS_HISTORY,
    PROMPT_PROB_DICT,
)

LLM_ALIAS_TO_ID = {
    "llama-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-32b": "Qwen/Qwen3-32B",
}
ALL_LLM_ALIASES = list(LLM_ALIAS_TO_ID.keys())

DEFAULT_PROB_STR = "-1.000"
CSV_STAGE_ORDER = ["Exploration", "Comforting", "Action", "Others"]
CSV_STAGE_PROB_KEYS = {
    "Exploration": ["prob_ex"],
    "Comforting": ["prob_su", "prob_co"],
    "Action": ["prob_ac"],
    "Others": ["prob_ot"],
}
STRATEGY_ID_TO_STAGE = {
    0: "Exploration",
    1: "Exploration",
    2: "Comforting",
    3: "Comforting",
    4: "Comforting",
    5: "Action",
    6: "Action",
    7: "Others",
}
STRATEGY_NAME_TO_STAGE = {
    "question": "Exploration",
    "restatement or paraphrasing": "Exploration",
    "reflection of feelings": "Comforting",
    "self-disclosure": "Comforting",
    "affirmation and reassurance": "Comforting",
    "providing suggestions": "Action",
    "information": "Action",
    "others": "Others",
}

TAG_SPEAKER = re.compile(r"\[(supporter|seeker)\]", re.IGNORECASE)
LINE_SPEAKER = re.compile(r"^(Supporter|Seeker)\s*:\s*", re.IGNORECASE | re.MULTILINE)

OUTPUT_FIELDNAMES = [
    "llm_alias",
    "llm_id",
    "classifier",
    "prediction",
    "prediction_id",
    "reference",
    "reference_id",
    "reference_response",
    "llm_raw_output",
    "llm_predict_strategy",
    "llm_parsed_response",
    "input_dialog",
    "context_source",
    "situation",
    "prompt",
]


def normalize_text(text) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().split())


def format_probability(value) -> str:
    try:
        if value is None or value == "":
            return DEFAULT_PROB_STR
        prob = float(value)
    except (TypeError, ValueError):
        return DEFAULT_PROB_STR
    if math.isnan(prob) or math.isinf(prob):
        return DEFAULT_PROB_STR
    return f"{prob:.3f}"


def parse_int(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def sanitize_name(name: str) -> str:
    return str(name).strip().replace("/", "_")


def to_llm_alias(llm_id: str) -> str:
    for alias, real_id in LLM_ALIAS_TO_ID.items():
        if llm_id == alias or llm_id == real_id:
            return alias
    return sanitize_name(llm_id)


def format_situation_for_prompt(situation) -> str:
    s = (situation or "").strip()
    return s if s else "(No situation text provided.)"


def strategy_probs_module_from_csv_row(row: dict) -> dict:
    return {
        "module_qu": format_probability(row.get("prob_qu", -1.0)),
        "module_rp": format_probability(row.get("prob_rp", -1.0)),
        "module_rf": format_probability(row.get("prob_rf", -1.0)),
        "module_sd": format_probability(row.get("prob_sd", -1.0)),
        "module_ar": format_probability(row.get("prob_ar", -1.0)),
        "module_ps": format_probability(row.get("prob_ps", -1.0)),
        "module_in": format_probability(row.get("prob_in", -1.0)),
        "module_ot": format_probability(row.get("prob_ot", -1.0)),
    }


def build_stage_aware_prompt(dialogue_context: str, probs: dict, stage: str, situation: str) -> str:
    sit = format_situation_for_prompt(situation)
    if stage == "Others":
        return PROMPT_OTHERS_HISTORY.format(dialogue_context=dialogue_context, situation=sit)
    return PROMPT_HIE_PROB_HISTORY.format(
        module_qu=probs.get("module_qu", DEFAULT_PROB_STR),
        module_rp=probs.get("module_rp", DEFAULT_PROB_STR),
        module_rf=probs.get("module_rf", DEFAULT_PROB_STR),
        module_sd=probs.get("module_sd", DEFAULT_PROB_STR),
        module_ar=probs.get("module_ar", DEFAULT_PROB_STR),
        module_ps=probs.get("module_ps", DEFAULT_PROB_STR),
        module_in=probs.get("module_in", DEFAULT_PROB_STR),
        module_ot=probs.get("module_ot", DEFAULT_PROB_STR),
        dialogue_context=dialogue_context,
        situation=sit,
        stage_strategy_definition=HIE_DIFINITON_STAGE_STRATEGY_PROMPT[stage],
        probability_information=PROMPT_PROB_DICT[stage].format(**probs),
    )


def normalize_stage_name(raw_stage: str):
    stage_aliases = {
        "exploration": "Exploration",
        "comforting": "Comforting",
        "action": "Action",
        "others": "Others",
        "supporting": "Comforting",
        "support": "Comforting",
        "su": "Comforting",
    }
    return stage_aliases.get(normalize_text(raw_stage))


def infer_stage_for_prompt(item: dict, prediction: str, prediction_id) -> str:
    for key in ("stage_pred", "stage_true"):
        stage = normalize_stage_name(str(item.get(key, "")).strip())
        if stage:
            return stage

    pred_stage = STRATEGY_NAME_TO_STAGE.get(normalize_text(prediction))
    if pred_stage:
        return pred_stage

    pred_idx = parse_int(prediction_id)
    if pred_idx in STRATEGY_ID_TO_STAGE:
        return STRATEGY_ID_TO_STAGE[pred_idx]

    scored = []
    for stage in CSV_STAGE_ORDER:
        vals = []
        for col in CSV_STAGE_PROB_KEYS[stage]:
            try:
                vals.append(float(item.get(col, -1.0)))
            except (TypeError, ValueError):
                vals.append(-1.0)
        if vals:
            scored.append((stage, max(vals)))
    if scored:
        return max(scored, key=lambda x: x[1])[0]
    return "Others"


def normalize_dialogue_to_turns(text) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""

    matches = list(TAG_SPEAKER.finditer(s))
    if matches:
        lines = []
        for j, m in enumerate(matches):
            sp = m.group(1).lower()
            label = "Supporter" if sp == "supporter" else "Seeker"
            i = m.end()
            while i < len(s) and s[i].isspace():
                i += 1
            while i < len(s) and s[i] == "[":
                end = s.find("]", i)
                if end == -1:
                    break
                inner = s[i + 1:end].strip().lower()
                if inner in ("supporter", "seeker"):
                    break
                i = end + 1
                while i < len(s) and s[i].isspace():
                    i += 1
            end_pos = matches[j + 1].start() if j + 1 < len(matches) else len(s)
            content = s[i:end_pos].strip()
            if content:
                lines.append(f"{label} : {content}")
        return "\n".join(lines).strip()

    if LINE_SPEAKER.search(s):
        return "\n".join(line.strip() for line in s.splitlines() if line.strip()).strip()

    return s


def build_accumulated_input_dialog(dialog, target_idx: int) -> str:
    lines = []
    for utt in dialog[:target_idx]:
        speaker = utt.get("speaker", "")
        txt = str(utt.get("text", "")).strip()
        if not txt:
            continue
        if speaker == "sys":
            lines.append(f"Supporter : {txt}")
        elif speaker == "usr":
            lines.append(f"Seeker : {txt}")
        else:
            lines.append(f"{speaker} : {txt}")
    return "\n".join(lines).strip()


def build_esconv_lookup() -> dict:
    ds = load_dataset("thu-coai/esconv")
    by_pair = defaultdict(deque)
    by_response = defaultdict(deque)
    for split in ("train", "validation", "test"):
        for raw in ds[split]["text"]:
            sample = ast.literal_eval(fix_text(raw))
            situation = str(sample.get("situation", "")).strip()
            dialog = sample.get("dialog", [])
            for i, utt in enumerate(dialog):
                if utt.get("speaker") != "sys":
                    continue
                strategy = normalize_text(str(utt.get("strategy", "")).strip())
                response = normalize_text(str(utt.get("text", "")).strip())
                input_dialog = build_accumulated_input_dialog(dialog, i)
                pair = (input_dialog, situation)
                by_pair[(strategy, response)].append(pair)
                by_response[response].append(pair)
    return {"by_pair": by_pair, "by_response": by_response}


def pick_dialogue_context_from_esconv(item: dict, esconv_lookup: dict):
    ref_strategy = normalize_text(str(item.get("reference_strategy", "")).strip())
    ref_response = normalize_text(str(item.get("reference_response", "")).strip())
    if not ref_response:
        raise ValueError("Empty reference_response for ESConv lookup")
    by_pair = esconv_lookup["by_pair"]
    by_response = esconv_lookup["by_response"]
    cand = by_pair.get((ref_strategy, ref_response))
    if cand and len(cand) > 0:
        input_dialog, situation = cand.popleft()
        return "ESConv", input_dialog, situation
    cand = by_response.get(ref_response)
    if cand and len(cand) > 0:
        input_dialog, situation = cand.popleft()
        return "ESConv", input_dialog, situation
    raise ValueError(
        f"No ESConv match for strategy/response: {ref_strategy!r} / {ref_response[:80]!r}"
    )


def load_csv_inference_rows(path: str) -> list:
    out = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("context", "")).strip() == "<START>":
                continue
            out.append(row)
    return out


def try_apply_chat_template(tokenizer, messages: list) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def call_single_prompt(client: OpenAI, model_name: str, prompt: str, temperature: float, max_tokens: int) -> str:
    try:
        response = client.completions.create(
            model=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].text or "").strip()
    except Exception as e:
        return f"Error: {e}"


def run_inference_parallel(
    prompts,
    llm_id: str,
    client: OpenAI,
    temperature: float,
    max_tokens: int,
    max_workers: int,
):
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                call_single_prompt, client, llm_id, prompts[i], temperature, max_tokens
            ): i
            for i in range(len(prompts))
        }
        for fut in tqdm(as_completed(future_to_idx), total=len(prompts), desc="Calling vLLM API"):
            results[future_to_idx[fut]] = fut.result()
    return results


def format_strategy_llm_output(text: str):
    pred = (text or "").strip()
    strategy, utterance = "N/A", pred
    m_full = re.search(
        r"Strategy\s*:\s*(.*?)\s*Response\s*:\s*(.*)",
        pred,
        re.DOTALL | re.IGNORECASE,
    )
    if m_full:
        return m_full.group(1).strip(), m_full.group(2).strip()
    m_str = re.search(r"Strategy\s*:\s*(.*?)(?:Response|$)", pred, re.IGNORECASE | re.DOTALL)
    m_rsp = re.search(r"Response\s*:\s*(.*)", pred, re.IGNORECASE | re.DOTALL)
    if m_str:
        strategy = m_str.group(1).strip()
    if m_rsp:
        utterance = m_rsp.group(1).strip()
    return strategy, utterance


def save_records(out_dir: Path, llm_id: str, classifier: str, records: list):
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_llm_id = sanitize_name(llm_id)
    csv_path = out_dir / f"{safe_llm_id}_{classifier}.csv"
    json_path = out_dir / f"{safe_llm_id}_{classifier}.json"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved: {csv_path} (rows={len(records)})")
    print(f"[OK] Saved: {json_path} (rows={len(records)})")
    return csv_path, json_path


def build_record(
    llm_alias: str,
    llm_id: str,
    classifier: str,
    meta: dict,
    prompt_text: str,
    raw_output: str,
) -> dict:
    predicted_strategy, parsed_response = format_strategy_llm_output(raw_output)
    return {
        "llm_alias": llm_alias,
        "llm_id": llm_id,
        "classifier": classifier,
        "prediction": meta["prediction"],
        "prediction_id": meta["prediction_id"],
        "reference": meta["reference"],
        "reference_id": meta["reference_id"],
        "reference_response": meta["reference_response"],
        "llm_raw_output": raw_output,
        "llm_predict_strategy": predicted_strategy,
        "llm_parsed_response": parsed_response,
        "input_dialog": meta["dialogue_context"],
        "context_source": meta["context_source"],
        "situation": meta["situation"],
        "prompt": prompt_text,
    }


def resolve_llm_ids(args) -> list:
    if args.all_llm:
        return [LLM_ALIAS_TO_ID[x] for x in ALL_LLM_ALIASES]
    if not args.llm_id:
        raise ValueError("`--llm_id` or `--all_llm` is required.")
    return [LLM_ALIAS_TO_ID.get(args.llm_id, args.llm_id)]
