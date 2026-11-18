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
