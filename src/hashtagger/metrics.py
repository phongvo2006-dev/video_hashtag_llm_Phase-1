"""Ranking + set metrics for multi-label hashtag prediction."""
from __future__ import annotations

import numpy as np


def prf_at_k(pred_ranked: list[list[str]], gold: list[list[str]], k: int) -> dict:
    """Precision / recall / F1 of the top-k predicted hashtags, averaged over examples."""
    P, R, F = [], [], []
    for p, g in zip(pred_ranked, gold):
        p, g = p[:k], set(g)
        hit = len(set(p) & g)
        prec = hit / k                     # standard P@k: emitting fewer than k tags is not rewarded
        rec = hit / max(len(g), 1)
        P.append(prec)
        R.append(rec)
        F.append(0.0 if hit == 0 else 2 * prec * rec / (prec + rec))
    return {f"P@{k}": float(np.mean(P)), f"R@{k}": float(np.mean(R)), f"F1@{k}": float(np.mean(F))}


def set_f1(pred_sets: list[list[str]], gold: list[list[str]]) -> dict:
    """Example-based F1 when the model chooses how many tags to emit."""
    F, n = [], []
    for p, g in zip(pred_sets, gold):
        p, g = set(p), set(g)
        hit = len(p & g)
        F.append(0.0 if hit == 0 else 2 * hit / (len(p) + len(g)))
        n.append(len(p))
    return {"SetF1": float(np.mean(F)), "avg_tags": float(np.mean(n))}


def category_accuracy(scores: np.ndarray, cat_idx: list[int], true_cat: list[int]) -> float:
    """Top-1 accuracy among the 20 category hashtags (scores: [N, |vocab|])."""
    sub = scores[:, cat_idx]
    return float(np.mean(sub.argmax(1) == np.asarray(true_cat)))


def f1_reward(pred: set, gold: set) -> float:
    hit = len(pred & gold)
    return 0.0 if hit == 0 else 2 * hit / (len(pred) + len(gold))


# ---------------------------------------------------------------------------------------------
# Graded-relevance metrics. MSR-VTT has 20 captions per clip, so a tag's relevance can be graded
# by annotator agreement: gain(tag) = share of the OTHER 19 captions that mention it (the input
# caption is excluded, so the metric never rewards copying the input). Category tag gain = 1.
# ---------------------------------------------------------------------------------------------
def _dcg(gains: list[float]) -> float:
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(gains)))


def ndcg_at_k(pred: list[str], gain: dict[str, float], k: int = 5, binary: bool = False) -> float:
    """nDCG@k; normalised by the best achievable ranking for THIS clip, so the ceiling is 1.0."""
    g = (lambda t: 1.0 if t in gain else 0.0) if binary else (lambda t: gain.get(t, 0.0))
    ideal = sorted([1.0] * len(gain) if binary else gain.values(), reverse=True)[:k]
    return _dcg([g(t) for t in pred[:k]]) / max(_dcg(ideal), 1e-12)


def weighted_recall_at_k(pred: list[str], gain: dict[str, float], k: int = 5) -> float:
    """Share of annotator 'votes' covered by the top-k tags."""
    return sum(gain.get(t, 0.0) for t in pred[:k]) / max(sum(gain.values()), 1e-12)


def ranking_suite(preds: list[list[str]], gold: list[list[str]], gains: list[dict], k: int = 5) -> dict:
    return {
        f"wnDCG@{k}": float(np.mean([ndcg_at_k(p, g, k) for p, g in zip(preds, gains)])),
        f"nDCG@{k}": float(np.mean([ndcg_at_k(p, g, k, binary=True) for p, g in zip(preds, gains)])),
        f"wRecall@{k}": float(np.mean([weighted_recall_at_k(p, g, k) for p, g in zip(preds, gains)])),
        f"Hit@{k}": float(np.mean([len(set(p[:k]) & set(g)) > 0 for p, g in zip(preds, gold)])),
        **prf_at_k(preds, gold, k),
    }


def category_suite(scores: np.ndarray, cat_idx: list[int], true_cat: list[int]) -> dict:
    """Top-1 / top-2 accuracy and macro-F1 over the 20 category tags."""
    order = np.argsort(-scores[:, cat_idx], axis=1)
    t = np.asarray(true_cat)
    p = order[:, 0]
    f1 = [2 * ((p == c) & (t == c)).sum() / max((p == c).sum() + (t == c).sum(), 1) for c in range(len(cat_idx))]
    return {"CatTop1": float(np.mean(p == t)), "CatTop2": float(np.mean((order[:, :2] == t[:, None]).any(1))),
            "CatMacroF1": float(np.mean(f1))}
