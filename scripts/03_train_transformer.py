"""Step 3 — supervised (BCE) training of the Transformer hashtag model."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hashtagger import data  # noqa: E402
from hashtagger.models import transformer as T  # noqa: E402
from hashtagger.models.common import LabelSpace, tune_threshold  # noqa: E402
from hashtagger.utils import load_config, rpath, save_json, seed_everything  # noqa: E402

cfg = load_config()
seed_everything(cfg["seed"])
torch.set_num_threads(max(1, torch.get_num_threads()))
videos, meta = data.load_processed(str(rpath(cfg["data"]["processed_dir"])))
labels = LabelSpace(meta["vocab"], meta["category_tags"])
Xtr, Ytr_tags = data.train_examples(videos)
Ytr = labels.encode(Ytr_tags)
tc = cfg["transformer"]

tok = T.WordTokenizer.build(Xtr, tc["min_token_freq"])
ids = tok.encode(Xtr, tc["max_len"])
model = T.HashtagTransformer(len(tok.itos), len(labels), tc["max_len"], tc["d_model"], tc["n_heads"],
                             tc["n_layers"], tc["dropout"])
print(f"tokenizer vocab={len(tok.itos)}  params={model.n_params():,}  examples={len(ids):,}")

t0 = time.perf_counter()
hist = T.train_bce(model, ids, Ytr, tc["epochs"], tc["lr"], tc["batch_size"])
train_sec = time.perf_counter() - t0

pred_dir = rpath("results/preds"); pred_dir.mkdir(parents=True, exist_ok=True)
probs = {}
for split in ["val", "test"]:
    X, _, _ = data.eval_examples(videos, split)
    probs[split] = T.predict_proba(model, tok.encode(X, tc["max_len"]))
    np.save(pred_dir / f"transformer_bce_{split}.npy", probs[split].astype(np.float32))

_, Yval, _ = data.eval_examples(videos, "val")
thr = tune_threshold(probs["val"], Yval, labels)
T.save(model, tok, str(rpath("checkpoints/transformer_bce.pt")), {"vocab": labels.vocab,
       "category_tags": labels.category_tags, "threshold": thr})
save_json({"train_sec": train_sec, "params": model.n_params(), "loss": hist, "val_threshold": thr},
          rpath("results/train_log_transformer.json"))
print(f"done in {train_sec:.0f}s, tuned val threshold={thr}")
