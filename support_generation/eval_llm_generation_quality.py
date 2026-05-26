import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import re
from collections import Counter

import numpy as np
import transformers

transformers.logging.set_verbosity_error()

from bert_score import score as bert_score
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

split_punct = lambda x: " ".join(y for y in re.findall(r"[\w']+|[.,!?;]", str(x)))


def my_lcs(string, sub):
    if len(string) < len(sub):
        sub, string = string, sub
    lengths = [[0 for _ in range(len(sub) + 1)] for _ in range(len(string) + 1)]
    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
    return lengths[len(string)][len(sub)]


class Metric(object):
    def __init__(self, hyps, refs, use_nltk=True):
        self.refs = []
        self.hyps = []
        self.refs_raw = refs
        self.hyps_raw = hyps
        self.use_nltk = use_nltk
        for r, h in zip(refs, hyps):
            self.forward([r], h)

    def forward(self, refs, hyp):
        if self.use_nltk:
            self.refs.append([word_tokenize(split_punct(e).lower()) for e in refs])
            self.hyps.append(word_tokenize(split_punct(hyp.lower())))
        else:
            self.refs.append([str(e).split() for e in refs])
            self.hyps.append(str(hyp).split())

    def calc_bleu_k(self, k):
        weights = [1.0 / k] * k + (4 - k) * [0.0]
        smoothie = SmoothingFunction().method4
        try:
            return corpus_bleu(self.refs, self.hyps, weights=weights, smoothing_function=smoothie)
        except Exception:
            return 0.0

    def calc_distinct_k(self, k):
        all_ngrams = []
        for sen in self.hyps:
            if len(sen) >= k:
                ngrams = [tuple(sen[i:i + k]) for i in range(len(sen) - k + 1)]
                all_ngrams.extend(ngrams)
        return len(set(all_ngrams)) / max(1, len(all_ngrams))

    def calc_unigram_f1(self):
        f1_scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            scores = []
            for ref in refs:
                cross = Counter(hyp) & Counter(ref)
                cross = sum(cross.values())
                p = cross / max(len(hyp), 1e-10)
                r = cross / max(len(ref), 1e-10)
                f1 = 2 * p * r / max(p + r, 1e-10)
                scores.append(f1)
            f1_scores.append(max(scores))
        return np.mean(f1_scores)

    def calc_rouge_l(self, beta=1.2):
        scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            prec, rec = [], []
            for ref in refs:
                lcs = my_lcs(ref, hyp)
                prec.append(lcs / max(len(hyp), 1e-10))
                rec.append(lcs / max(len(ref), 1e-10))
            prec_max, rec_max = max(prec), max(rec)
            score = (
                ((1 + beta ** 2) * prec_max * rec_max)
                / float(rec_max + beta ** 2 * prec_max)
                if prec_max + rec_max > 0
                else 0.0
            )
            scores.append(score)
        return np.mean(scores)

    def calc_bert_score(self, lang="en"):
        _, _, F1 = bert_score(self.hyps_raw, self.refs_raw, lang=lang, verbose=False)
        return float(F1.mean())

    def close(self):
        return {
            "length": float(np.mean([len(h) for h in self.hyps])),
            "dist-1": 100 * self.calc_distinct_k(1),
            "dist-2": 100 * self.calc_distinct_k(2),
            "bleu-1": 100 * self.calc_bleu_k(1),
            "bleu-2": 100 * self.calc_bleu_k(2),
            "f1": 100 * self.calc_unigram_f1(),
            "rouge-l": 100 * self.calc_rouge_l(),
            "bert_f1": 100 * self.calc_bert_score(),
        }


def load_ref_hyp_pairs(rows):
    refs, hyps = [], []
    for row in rows:
        ref = row.get("reference_response", "")
        ref = "" if ref is None else str(ref)
        hyp = row.get("llm_parsed_response", "")
        hyp = "" if hyp is None else str(hyp)
        refs.append(ref)
        hyps.append(hyp)
    return refs, hyps


def evaluate_rows(rows):
    if not rows:
        return None
    refs, hyps = load_ref_hyp_pairs(rows)
    if not refs:
        return None
    return Metric(hyps=hyps, refs=refs).close()
