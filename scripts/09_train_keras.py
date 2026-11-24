"""Step 9 — the hashtag Transformer re-implemented in TensorFlow/Keras: BCE training + RL fine-tuning.

Uses exactly the same labels, data split, architecture and hyper-parameters as the PyTorch model
(configs/default.yaml: `transformer` and `rl` sections), so the two frameworks compare like-for-like.
Leakage guards: vocabulary from TRAIN captions only; thresholds and RL monitoring on VALIDATION only;
the test split is predicted once, after training.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import keras  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hashtagger import data  # noqa: E402
from hashtagger.metrics import prf_at_k, set_f1  # noqa: E402
from hashtagger.models import keras_transformer as KT  # noqa: E402
from hashtagger.models.common import LabelSpace, tune_threshold  # noqa: E402
from hashtagger.utils import load_config, rpath, save_json  # noqa: E402

cfg = load_config()
tc, rc = cfg["transformer"], cfg["rl"]
keras.utils.set_random_seed(cfg["seed"])
videos, meta = data.load_processed(str(rpath(cfg["data"]["processed_dir"])))
labels = LabelSpace(meta["vocab"], meta["category_tags"])
Xtr, Ytr_tags = data.train_examples(videos)
Ytr = labels.encode(Ytr_tags)
Xval, Yval, _ = data.eval_examples(videos, "val")
Xte, _, _ = data.eval_examples(videos, "test")

vocab = KT.build_vocabulary(Xtr, tc["min_token_freq"])             # TRAIN captions only
vec = KT.make_vectorizer(vocab, tc["max_len"])
enc_ids = lambda X: vec(tf.constant(X)).numpy().astype("int32")     # noqa: E731
ids, val_ids, te_ids = enc_ids(Xtr), enc_ids(Xval), enc_ids(Xte)
model = KT.HashtagEncoder(len(vec.get_vocabulary()), len(labels), tc["max_len"], tc["d_model"], tc["n_heads"],
                          tc["n_layers"], tc["dropout"], prior=Ytr.mean(0))
model(ids[:2])
n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
print(f"keras vocab={len(vec.get_vocabulary())} params={n_params:,} examples={len(ids):,}", flush=True)

pred_dir, ck = rpath("results/preds"), rpath("checkpoints")
log = {"params": n_params, "framework": f"TensorFlow {tf.__version__} / Keras {keras.__version__}"}

# ---------------------------------------------------------------- 1. supervised (BCE)
t0 = time.perf_counter()
log["bce_loss"] = KT.train_bce(model, ids, Ytr, tc["epochs"], tc["lr"], tc["batch_size"], cfg["seed"])
log["bce_train_sec"] = time.perf_counter() - t0
pv = KT.predict_proba(model, val_ids)
np.save(pred_dir / "keras_bce_val.npy", pv)
np.save(pred_dir / "keras_bce_test.npy", KT.predict_proba(model, te_ids))
log["bce_val"] = {**prf_at_k(labels.topk(pv, 5), Yval, 5), **set_f1(labels.threshold(pv, 0.5), Yval)}
log["bce_val_threshold"] = tune_threshold(pv, Yval, labels)
KT.string_model(vec, model).save(ck / "keras_bce.keras")
print("BCE val", log["bce_val"], flush=True)


# ---------------------------------------------------------------- 2. RL fine-tuning
def eval_fn():
    return set_f1(labels.threshold(KT.predict_proba(model, val_ids), 0.5), Yval)["SetF1"]


t0 = time.perf_counter()
log["rl_history"] = KT.rl_finetune(model, ids, Ytr, rc["epochs"], rc["lr"], rc["batch_size"], rc["n_samples"],
                                   rc["bce_anchor"], eval_fn=eval_fn, eval_every=50, seed=cfg["seed"])
log["rl_train_sec"] = time.perf_counter() - t0
pv = KT.predict_proba(model, val_ids)
np.save(pred_dir / "keras_rl_val.npy", pv)
np.save(pred_dir / "keras_rl_test.npy", KT.predict_proba(model, te_ids))
log["rl_val_threshold"] = tune_threshold(pv, Yval, labels)
KT.string_model(vec, model).save(ck / "keras_rl.keras")
save_json(log, rpath("results/train_log_keras.json"))
h = log["rl_history"]["val_setf1"]
print(f"done: BCE {log['bce_train_sec'] / 60:.1f} min, RL {log['rl_train_sec'] / 60:.1f} min; "
      f"val SetF1 {h[0]:.4f} -> {h[-1]:.4f}", flush=True)
