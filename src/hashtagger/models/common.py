"""Shared helpers: label encoding, batching, decoding scores into hashtags."""
from __future__ import annotations

import numpy as np


class LabelSpace:
    def __init__(self, vocab: list[str], category_tags: list[str]):
        self.vocab = list(vocab)
        self.index = {t: i for i, t in enumerate(self.vocab)}
        self.category_tags = list(category_tags)
        self.cat_idx = [self.index[t] for t in self.category_tags]

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, tag_lists: list[list[str]]) -> np.ndarray:
        Y = np.zeros((len(tag_lists), len(self.vocab)), dtype=np.float32)
        for i, tags in enumerate(tag_lists):
            for t in tags:
                j = self.index.get(t)
                if j is not None:
                    Y[i, j] = 1.0
        return Y

    def topk(self, scores: np.ndarray, k: int) -> list[list[str]]:
        idx = np.argsort(-scores, axis=1)[:, :k]
        return [[self.vocab[j] for j in row] for row in idx]

    def threshold(self, probs: np.ndarray, thr: float, min_tags: int = 1, max_tags: int = 10) -> list[list[str]]:
        """Adaptive-size tag sets: every tag with p >= thr (at least ``min_tags``)."""
        out = []
        order = np.argsort(-probs, axis=1)
        for p, o in zip(probs, order):
            chosen = [j for j in o[:max_tags] if p[j] >= thr]
            if len(chosen) < min_tags:
                chosen = list(o[:min_tags])
            out.append([self.vocab[j] for j in chosen])
        return out


def tune_threshold(probs: np.ndarray, gold: list[list[str]], labels: LabelSpace) -> float:
    """Grid-search one global threshold on validation data to maximise set-F1."""
    from ..metrics import set_f1
    best, best_t = -1.0, 0.5
    for t in np.arange(0.05, 0.71, 0.025):
        f = set_f1(labels.threshold(probs, float(t)), gold)["SetF1"]
        if f > best:
            best, best_t = f, float(t)
    return round(best_t, 3)
