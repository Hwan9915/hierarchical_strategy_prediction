# ESConv context is accumulated from the beginning of each dialog (the `turns`
# argument is kept for API / checkpoint-path compatibility).
import random

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader

from strategy_prediction.config import STRATEGY_DICT


def remove_surrogate_chars(text) -> str:
    if not isinstance(text, str) or text is None:
        return ""
    return text.encode("utf-8", "ignore").decode("utf-8")


def build_prompt(i, dialog, turns, is_train):
    """
    Build the context from an ESConv dialog up to (but not including) the
    current supporter utterance at index `i`.

    Uses a sliding window of the previous `turns` utterances:
    range(max(0, i - turns), i). The per-module window size is selected by
    `turn_for_module` (comforting_predictor=1; all other modules=5).
    """

    def _tag_for(idx: int) -> str:
        speaker = dialog[idx]["speaker"]
        base_tag = "[supporter]" if speaker == "sys" else "[seeker]"
        return f"{base_tag} [{dialog[idx]['strategy']}]" if speaker == "sys" else base_tag

    def _txt(idx: int) -> str:
        return remove_surrogate_chars(dialog[idx]["text"])

    def _utt(idx: int) -> str:
        return f"{_tag_for(idx)} {_txt(idx)}"

    if i == 0:
        return ("", "<START>")

    start = max(0, i - turns)
    utterances = [_utt(j) for j in range(start, i)]

    if is_train:
        return ("", " ".join(utterances))

    if len(utterances) == 1:
        c1, c2 = "", utterances[0]
    else:
        c1, c2 = " ".join(utterances[:-1]), utterances[-1]
    return (c1, c2)


def collate_fn(batch):
    batch_out = {}
    for key in batch[0]:
        vals = []
        for d in batch:
            if d[key] is None:
                print(f"[ERROR] None detected in key={key}, sample={d}")
                vals.append("")
            else:
                vals.append(d[key])
        batch_out[key] = vals
    return batch_out


def preprocess_dataset(dataset, num_labels: int, label_mapping_dict, turns, is_train, is_llm):
    example_idx = 0

    dataset_dict = {
        "idx": [],
        "label": [],
    }
    if is_llm:
        dataset_dict["dialogue"] = []
        dataset_dict["label_context"] = []

    if is_train:
        dataset_dict["context"] = []
    else:
        dataset_dict["context_t1"], dataset_dict["context_t2"] = [], []

    for raw in dataset["text"]:
        data = eval(raw)
        for i, now_utt in enumerate(data.get("dialog", [])):
            if now_utt.get("speaker") != "sys":
                continue

            mapped_label = label_mapping_dict[STRATEGY_DICT[now_utt["strategy"]]]
            if mapped_label == -1:
                continue
            c1, c2 = build_prompt(i, data["dialog"], turns, is_train)

            prev_dialogue = []
            for utt in data["dialog"][:i]:
                clean_utt = utt.copy()
                clean_utt["text"] = remove_surrogate_chars(utt.get("text"))
                prev_dialogue.append(clean_utt)

            if is_train:
                dataset_dict["context"].append(c2)
            else:
                dataset_dict["context_t1"].append(c1)
                dataset_dict["context_t2"].append(c2)

            dataset_dict["idx"].append(example_idx)
            dataset_dict["label"].append(mapped_label)
            if is_llm:
                dataset_dict["dialogue"].append(prev_dialogue)
                dataset_dict["label_context"].append(remove_surrogate_chars(now_utt.get("text")))

            example_idx += 1

    return Dataset.from_dict(dataset_dict)


def _get_label_indices(dataset):
    label_to_idx = {}
    for i, lab in enumerate(dataset["label"]):
        label_to_idx.setdefault(int(lab), []).append(i)
    return label_to_idx


def oversample(dataset, second: bool = False):
    label_to_idx = _get_label_indices(dataset)
    class_sizes = sorted(len(v) for v in label_to_idx.values())

    target_cnt = class_sizes[-2] if second else class_sizes[-1]
    parts = []
    for _, idx_list in label_to_idx.items():
        cur_cnt = len(idx_list)

        if cur_cnt > target_cnt:
            sampled = random.sample(idx_list, target_cnt)
        elif cur_cnt < target_cnt:
            reps, rem = divmod(target_cnt, cur_cnt)
            sampled = idx_list * reps + random.sample(idx_list, rem)
        else:
            sampled = idx_list

        parts.append(dataset.select(sampled))

    balanced = concatenate_datasets(parts).shuffle(seed=42)
    return balanced.with_format("torch")


def create_dataloader(
    sampling: int,
    num_labels: int,
    label_mapping_dict: dict,
    train_batch_size: int,
    valid_batch_size: int,
    test_batch_size: int,
    turns: int = 2,
    is_train: int = 1,
    is_llm: int = 0,
):
    if turns not in (1, 2, 3, 4, 5):
        raise ValueError(f"turns must be 1-5, got {turns}")
    esconv = load_dataset("thu-coai/esconv")

    train_tok = preprocess_dataset(esconv["train"], num_labels, label_mapping_dict, turns, is_train, is_llm)
    if sampling == 1:
        train_tok = oversample(train_tok)
    else:
        train_tok = train_tok.with_format("torch")

    valid_tok = preprocess_dataset(esconv["validation"], num_labels, label_mapping_dict, turns, is_train, is_llm)
    test_tok = preprocess_dataset(esconv["test"], num_labels, label_mapping_dict, turns, is_train, is_llm)

    def _worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_tok,
        shuffle=True,
        batch_size=train_batch_size,
        num_workers=4,
        worker_init_fn=_worker_init_fn,
        generator=torch.Generator().manual_seed(42),
    )
    valid_loader = DataLoader(
        valid_tok,
        batch_size=valid_batch_size,
        num_workers=4,
        worker_init_fn=_worker_init_fn,
    )
    test_loader = DataLoader(
        test_tok,
        batch_size=test_batch_size,
        num_workers=4,
        worker_init_fn=_worker_init_fn,
        collate_fn=None if not is_llm else collate_fn,
    )

    return train_loader, valid_loader, test_loader
