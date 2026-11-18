"""Model 5 — fine-tuned pretrained RoBERTa-base (the industry-standard recipe for text tagging).

caption -> byte-BPE -> RoBERTa-base (12 layers, 768-d, 125M params, pretrained on
160 GB of English text) -> masked mean-pool -> dropout -> linear -> |vocab| logits.
All weights are fine-tuned (encoder lr 3e-5, head lr 1e-3).

Two ways to get the pretrained weights:
  source="hf"    : Hugging Face ``roberta-base`` via ``transformers`` (needs huggingface.co)
  source="spacy" : the identical RoBERTa-base weights shipped inside spaCy's
                   ``en_core_web_trf`` package (hosted on GitHub releases). Used here
                   because the build machine could reach GitHub but not Hugging Face.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

PAD_ID, BOS_ID, EOS_ID = 1, 0, 2          # RoBERTa special ids (same in both sources)


class RobertaBackbone(nn.Module):
    def __init__(self, source: str = "spacy"):
        super().__init__()
        self.source = source
        if source == "hf":
            from transformers import AutoModel, AutoTokenizer
            self.encoder = AutoModel.from_pretrained("roberta-base")
            self._tok = AutoTokenizer.from_pretrained("roberta-base")
        elif source == "spacy":
            import spacy
            nlp = spacy.load("en_core_web_trf", exclude=["tagger", "parser", "attribute_ruler",
                                                          "lemmatizer", "ner"])
            walk = list(nlp.get_pipe("transformer").model.walk())
            self.encoder = next(n for n in walk if n.name == "pytorch").shims[0]._model
            self._bpe = next(n for n in walk if n.name == "byte_bpe_encoder").attrs["byte_bpe_processor"]
        else:
            raise ValueError(source)
        self.hidden = 768

    def encode_text(self, texts: list[str], max_len: int = 40) -> list[list[int]]:
        if self.source == "hf":
            return [self._tok(t, truncation=True, max_length=max_len)["input_ids"] for t in texts]
        return [[BOS_ID] + self._bpe.encode_as_ids(" " + t.strip())[: max_len - 2] + [EOS_ID] for t in texts]

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.source == "hf":
            return self.encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        from curated_transformers.models.attention import AttentionMask
        return self.encoder(ids, AttentionMask(mask)).last_hidden_layer_states


class RobertaTagger(nn.Module):
    def __init__(self, n_labels: int, source: str = "spacy", dropout: float = 0.1):
        super().__init__()
        self.backbone = RobertaBackbone(source)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.hidden, n_labels)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.backbone(ids, mask)
        m = mask.unsqueeze(-1).float()
        return self.head(self.drop((h * m).sum(1) / m.sum(1).clamp(min=1)))


def collate(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic padding: pad only to the longest caption in the batch."""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    return ids, ids != PAD_ID


@torch.no_grad()
def predict_proba(model: RobertaTagger, seqs: list[list[int]], batch_size: int = 128) -> np.ndarray:
    model.eval()
    order = np.argsort([len(s) for s in seqs])                # length-bucketed for speed
    out = np.zeros((len(seqs), model.head.out_features), dtype=np.float32)
    for b in range(0, len(seqs), batch_size):
        ix = order[b:b + batch_size]
        ids, mask = collate([seqs[i] for i in ix])
        out[ix] = torch.sigmoid(model(ids, mask)).numpy()
    return out
