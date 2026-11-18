"""Model 2 — small Transformer encoder trained from scratch for multi-label hashtags.

caption -> word tokens -> [CLS] + embeddings + positions -> 2-layer encoder
-> masked mean-pool -> MLP -> |vocab| logits (sigmoid = P(tag | caption)).
~3M parameters; trains on CPU in minutes.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import torch
from torch import nn

_TOK = re.compile(r"[a-z]+|\d+")
PAD, UNK, CLS = 0, 1, 2


class WordTokenizer:
    def __init__(self, itos: list[str] | None = None):
        self.itos = itos or ["[PAD]", "[UNK]", "[CLS]"]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 2) -> "WordTokenizer":
        cnt = Counter(w for t in texts for w in _TOK.findall(t.lower()))
        return cls(["[PAD]", "[UNK]", "[CLS]"] + sorted(w for w, c in cnt.items() if c >= min_freq))

    def encode(self, texts: list[str], max_len: int) -> torch.Tensor:
        out = torch.zeros(len(texts), max_len, dtype=torch.long)
        for i, t in enumerate(texts):
            ids = [CLS] + [self.stoi.get(w, UNK) for w in _TOK.findall(t.lower())][: max_len - 1]
            out[i, : len(ids)] = torch.tensor(ids)
        return out


class HashtagTransformer(nn.Module):
    def __init__(self, vocab_size: int, n_labels: int, max_len=32, d_model=192, n_heads=4,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, n_labels=n_labels, max_len=max_len, d_model=d_model,
                        n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, n_labels))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = ids == PAD
        h = self.emb(ids) + self.pos[:, : ids.shape[1]]
        h = self.norm(self.encoder(h, src_key_padding_mask=mask))
        keep = (~mask).unsqueeze(-1).float()
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
        return self.head(pooled)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_bce(model: HashtagTransformer, ids: torch.Tensor, Y: np.ndarray, epochs: int, lr: float,
              batch_size: int, log=print) -> list[float]:
    with torch.no_grad():                           # output bias = label prior log-odds
        prior = np.clip(Y.mean(0), 1e-4, 1 - 1e-4)
        model.head[-1].bias.copy_(torch.from_numpy(np.log(prior / (1 - prior))).float())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(ids) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    lossf = nn.BCEWithLogitsLoss()
    Yt = torch.from_numpy(Y)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ids))
        tot = 0.0
        for b in range(0, len(perm), batch_size):
            ix = perm[b:b + batch_size]
            loss = lossf(model(ids[ix]), Yt[ix])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(ix)
        hist.append(tot / len(perm))
        log(f"  [transformer] epoch {ep + 1}/{epochs} bce={hist[-1]:.4f}")
    return hist


@torch.no_grad()
def predict_proba(model: HashtagTransformer, ids: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(ids[b:b + batch_size])) for b in range(0, len(ids), batch_size)]
    return torch.cat(out).numpy()


def save(model: HashtagTransformer, tok: WordTokenizer, path: str, extra: dict | None = None) -> None:
    torch.save({"cfg": model.cfg, "state": model.state_dict(), "itos": tok.itos, **(extra or {})}, path)


def load(path: str) -> tuple[HashtagTransformer, WordTokenizer, dict]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    m = HashtagTransformer(**d["cfg"])
    m.load_state_dict(d["state"])
    m.eval()
    return m, WordTokenizer(d["itos"]), d
