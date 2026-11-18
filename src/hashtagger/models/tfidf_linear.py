"""Baseline 1 — TF-IDF (word 1-2 grams) + multi-label logistic regression.

One linear layer with sigmoid outputs trained with BCE = one-vs-rest logistic
regression over all hashtags at once. Strong, cheap, CPU-friendly baseline.
"""
from __future__ import annotations

import pickle

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn


class TfidfLinear:
    def __init__(self, n_labels: int, max_features: int = 30000, ngram_range=(1, 2)):
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=tuple(ngram_range),
                                   sublinear_tf=True, min_df=2)
        self.n_labels = n_labels
        self.linear: nn.Linear | None = None

    @staticmethod
    def _to_torch(X) -> torch.Tensor:
        X = X.tocoo()
        idx = torch.tensor(np.vstack([X.row, X.col]), dtype=torch.long)
        return torch.sparse_coo_tensor(idx, torch.tensor(X.data, dtype=torch.float32), X.shape,
                                       check_invariants=False)

    def fit(self, captions: list[str], Y: np.ndarray, epochs=5, lr=0.5, batch_size=512,
            weight_decay=0.0, log=print) -> list[float]:
        X = self.vec.fit_transform(captions).tocsr()
        self.linear = nn.Linear(X.shape[1], self.n_labels)
        with torch.no_grad():                       # start from the label prior, not p=0.5
            nn.init.zeros_(self.linear.weight)
            prior = Y.mean(0).clip(1e-4, 1 - 1e-4)
            self.linear.bias.copy_(torch.from_numpy(np.log(prior / (1 - prior))))
        # Adagrad (as in classic sparse linear models): rare n-gram features get large steps.
        # NOTE: keep weight_decay=0. With adaptive optimisers an L2 term is *normalised* per
        # coordinate, so for a feature absent from the batch even 1e-6 decay becomes a
        # full-size step towards zero; the model never fit (see README "lessons learned").
        opt = torch.optim.Adagrad(self.linear.parameters(), lr=lr, weight_decay=weight_decay)
        lossf = nn.BCEWithLogitsLoss()
        Yt = torch.from_numpy(Y)
        hist = []
        for ep in range(epochs):
            perm = np.random.permutation(X.shape[0])
            tot = 0.0
            for b in range(0, len(perm), batch_size):
                ix = perm[b:b + batch_size]
                logits = torch.sparse.mm(self._to_torch(X[ix]), self.linear.weight.T) + self.linear.bias
                loss = lossf(logits, Yt[ix])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(ix)
            hist.append(tot / len(perm))
            log(f"  [tfidf] epoch {ep + 1}/{epochs} bce={hist[-1]:.4f}")
        return hist

    @torch.no_grad()
    def predict_proba(self, captions: list[str]) -> np.ndarray:
        X = self._to_torch(self.vec.transform(captions))
        return torch.sigmoid(torch.sparse.mm(X, self.linear.weight.T) + self.linear.bias).numpy()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.linear.parameters())

    def save(self, path: str) -> None:
        pickle.dump({"vec": self.vec, "state": self.linear.state_dict(), "n_labels": self.n_labels},
                    open(path, "wb"))

    @classmethod
    def load(cls, path: str) -> "TfidfLinear":
        d = pickle.load(open(path, "rb"))
        m = cls(d["n_labels"])
        m.vec = d["vec"]
        w = d["state"]["weight"]
        m.linear = nn.Linear(w.shape[1], w.shape[0])
        m.linear.load_state_dict(d["state"])
        return m
