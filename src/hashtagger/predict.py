"""Inference API used by the demo app and CLI.

    from hashtagger.predict import Hashtagger
    h = Hashtagger.load()                       # RoBERTa+RL if trained locally, else the 14 MB transformer
    h.predict("a chef slices onions in a kitchen")
    -> [("cooking", 0.93), ("food", 0.88), ("kitchen", 0.71), ...]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import transformer as T
from .utils import ROOT


class Hashtagger:
    def __init__(self, model, tok, vocab, category_tags, threshold=0.5, max_len=32):
        self.model, self.tok, self.vocab = model, tok, vocab
        self.category_tags, self.threshold, self.max_len = category_tags, threshold, max_len

    @classmethod
    def load(cls, kind: str = "auto") -> "Hashtagger":
        """kind: 'roberta' (fine-tuned RoBERTa-base + RL, best; ~500 MB, run `make roberta` first),
        'transformer' (small model shipped in the repo, 14 MB) or 'auto' (roberta if available)."""
        ck = ROOT / "checkpoints"
        if kind == "roberta" or (kind == "auto" and (ck / "roberta_rl.pt").exists()):
            import torch
            from .models.pretrained import RobertaTagger
            from .utils import load_config
            extra = T.load(str(ck / "transformer_rl.pt"))[2]           # label space
            rc = load_config()["roberta"]
            model = RobertaTagger(len(extra["vocab"]), rc["source"])
            model.load_state_dict(torch.load(ck / "roberta_rl.pt"))
            model.eval()
            h = cls(model, None, extra["vocab"], extra["category_tags"], 0.5, rc["max_len"])
            h.kind = "roberta"
            return h
        model, tok, extra = T.load(str(ck / "transformer_rl.pt"))
        h = cls(model, tok, extra["vocab"], extra["category_tags"], extra.get("threshold", 0.5),
                model.cfg["max_len"])
        h.kind = "transformer"
        return h

    def proba(self, captions: list[str]) -> np.ndarray:
        if getattr(self, "kind", "") == "keras":                       # Keras model takes raw strings
            import tensorflow as tf
            return np.asarray(self.model(tf.constant(captions)))
        if self.tok is None:                                            # RoBERTa
            from .models.pretrained import predict_proba
            return predict_proba(self.model, self.model.backbone.encode_text(captions, self.max_len))
        return T.predict_proba(self.model, self.tok.encode(captions, self.max_len))

    def predict(self, caption: str, max_tags: int = 8, min_tags: int = 3) -> list[tuple[str, float]]:
        p = self.proba([caption])[0]
        order = np.argsort(-p)
        chosen = [j for j in order[:max_tags] if p[j] >= self.threshold]
        if len(chosen) < min_tags:
            chosen = list(order[:min_tags])
        return [(self.vocab[j], float(p[j])) for j in chosen]

    def candidates(self, caption: str, n: int = 12) -> list[tuple[str, float]]:
        p = self.proba([caption])[0]
        return [(self.vocab[j], float(p[j])) for j in np.argsort(-p)[:n]]

    def category(self, caption: str) -> tuple[str, float]:
        p = self.proba([caption])[0]
        idx = [self.vocab.index(c) for c in self.category_tags]
        j = idx[int(np.argmax(p[idx]))]
        return self.vocab[j], float(p[j])
