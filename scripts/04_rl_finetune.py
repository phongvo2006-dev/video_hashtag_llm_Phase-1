"""Step 4 — reinforcement fine-tuning (REINFORCE + RLOO) of the BCE Transformer on set-F1 reward."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hashtagger import data  # noqa: E402
from hashtagger.metrics import set_f1  # noqa: E402
from hashtagger.models import transformer as T  # noqa: E402
from hashtagger.models.common import LabelSpace, tune_threshold  # noqa: E402
from hashtagger.models.rl import rl_finetune  # noqa: E402
from hashtagger.utils import load_config, rpath, save_json, seed_everything  # noqa: E402

cfg = load_config()
seed_everything(cfg["seed"])
videos, meta = data.load_processed(str(rpath(cfg["data"]["processed_dir"])))
labels = LabelSpace(meta["vocab"], meta["category_tags"])
Xtr, Ytr_tags = data.train_examples(videos)
Ytr = labels.encode(Ytr_tags)
rc, tc = cfg["rl"], cfg["transformer"]

model, tok, _ = T.load(str(rpath("checkpoints/transformer_bce.pt")))
ids = tok.encode(Xtr, tc["max_len"])
Xval, Yval, _ = data.eval_examples(videos, "val")
val_ids = tok.encode(Xval, tc["max_len"])


def eval_fn() -> float:
    p = T.predict_proba(model, val_ids)
    return set_f1(labels.threshold(p, 0.5), Yval)["SetF1"]


t0 = time.perf_counter()
hist = rl_finetune(model, ids, Ytr, rc["epochs"], rc["lr"], rc["batch_size"], rc["n_samples"],
                   rc["bce_anchor"], rc["entropy_coef"], eval_fn=eval_fn, eval_every=50)
train_sec = time.perf_counter() - t0

pred_dir = rpath("results/preds")
probs = {}
for split in ["val", "test"]:
    X, _, _ = data.eval_examples(videos, split)
    probs[split] = T.predict_proba(model, tok.encode(X, tc["max_len"]))
    np.save(pred_dir / f"transformer_rl_{split}.npy", probs[split].astype(np.float32))
thr = tune_threshold(probs["val"], Yval, labels)
T.save(model, tok, str(rpath("checkpoints/transformer_rl.pt")), {"vocab": labels.vocab,
       "category_tags": labels.category_tags, "threshold": 0.5, "tuned_threshold": thr})
save_json({"train_sec": train_sec, "params": model.n_params(), "history": hist, "val_threshold": thr},
          rpath("results/train_log_rl.json"))
print(f"RL done in {train_sec:.0f}s; val SetF1 {hist['val_setf1'][0]:.4f} -> {hist['val_setf1'][-1]:.4f}")
