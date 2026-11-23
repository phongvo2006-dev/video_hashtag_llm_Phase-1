"""Baseline 0 — rule-based keyword hashtags (what a naive tool does).

Takes the terms that literally appear in the caption (TensorFlow tokenisation + stop-list), keeps those in the
hashtag vocabulary and ranks them by training frequency. No learning, so it
can never produce tags the caption does not spell out (e.g. category tags).
"""
from __future__ import annotations

import numpy as np

from .. import text
from .common import LabelSpace


class KeywordTagger:
    def __init__(self, labels: LabelSpace, term_map: dict[str, str], prior: np.ndarray):
        self.labels = labels
        self.extractor = text.TermExtractor(term_map)
        self.prior = prior / prior.max()            # training tag frequency, in [0, 1]

    def scores(self, captions: list[str]) -> np.ndarray:
        S = np.tile(self.prior * 1e-3, (len(captions), 1))   # tiny prior only breaks ties
        for i, terms in enumerate(self.extractor.terms(captions)):
            for t in terms:
                j = self.labels.index.get(t)
                if j is not None:
                    S[i, j] = 1.0 + self.prior[j]
        return S

    def predict(self, captions: list[str]) -> list[list[str]]:
        S = self.scores(captions)
        return [[self.labels.vocab[j] for j in np.argsort(-row) if row[j] >= 1.0] for row in S]
