"""Inference API used by the demo app and CLI.

    from hashtagger.predict import Hashtagger
    h = Hashtagger.load()            # RoBERTa + RL if trained locally, else the 14 MB Keras Transformer
    h.predict("a chef slices onions in a kitchen")
    -> [("cooking", 0.93), ("food", 0.88), ("kitchen", 0.71), ...]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .utils import ROOT


class Hashtagger:
    def __init__(self, model, kind, vocab, category_tags, threshold=0.5, max_len=40):
        self.model, self.kind, self.vocab = model, kind, vocab
        self.category_tags, self.threshold, self.max_len = category_tags, threshold, max_len

    @classmethod
    def load(cls, kind: str = "auto", ckpt_dir: str | Path | None = None) -> "Hashtagger":
        """kind: 'roberta' — fine-tuned RoBERTa-base + RL, the best model (~500 MB, run `make roberta`);
        'transformer' — the small Keras model shipped in the repo (14 MB, raw strings in);
        'auto' — roberta if its checkpoint exists, else transformer."""
        ck = Path(ckpt_dir) if ckpt_dir else ROOT / "checkpoints"
        lab = json.loads((ck / "labels.json").read_text())
        if kind == "roberta" or (kind == "auto" and (ck / "roberta_rl.pt").exists()):
            import torch
            from .models.pretrained import RobertaTagger
            from .utils import load_config
            rc = load_config()["roberta"]
            model = RobertaTagger(len(lab["vocab"]), rc["source"])
            model.load_state_dict(torch.load(ck / "roberta_rl.pt"))
            model.eval()
            return cls(model, "roberta", lab["vocab"], lab["category_tags"], 0.5, rc["max_len"])
        import keras
        from .models import keras_transformer  # noqa: F401  (registers the custom layers)
        model = keras.models.load_model(ck / "keras_bce.keras")
        return cls(model, "transformer", lab["vocab"], lab["category_tags"], lab.get("keras_threshold", 0.5))

    def proba(self, captions: list[str]) -> np.ndarray:
        if self.kind == "transformer":                                  # Keras model takes raw strings
            import tensorflow as tf
            return np.asarray(self.model(tf.constant(captions)))
        from .models.pretrained import predict_proba
        return predict_proba(self.model, self.model.backbone.encode_text(captions, self.max_len))

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
