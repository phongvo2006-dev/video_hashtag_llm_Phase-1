"""Step 2 — baselines: rule-based keyword tagger and TF-IDF + logistic regression."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hashtagger import data  # noqa: E402
from hashtagger.models.common import LabelSpace  # noqa: E402
from hashtagger.models.keyword import KeywordTagger  # noqa: E402
from hashtagger.models.tfidf_linear import TfidfLinear  # noqa: E402
from hashtagger.utils import load_config, rpath, save_json, seed_everything  # noqa: E402

cfg = load_config()
seed_everything(cfg["seed"])
videos, meta = data.load_processed(str(rpath(cfg["data"]["processed_dir"])))
labels = LabelSpace(meta["vocab"], meta["category_tags"])
Xtr, Ytr_tags = data.train_examples(videos)
Ytr = labels.encode(Ytr_tags)
pred_dir, ckpt = rpath("results/preds"), rpath("checkpoints")
pred_dir.mkdir(parents=True, exist_ok=True); ckpt.mkdir(exist_ok=True)
log = {}

# ---- keyword baseline (no training) -------------------------------------------
prior = Ytr.sum(0)
kw = KeywordTagger(labels, meta["gerunds"], prior)
np.save(ckpt / "keyword_prior.npy", prior)
for split in ["val", "test"]:
    X, _, _ = data.eval_examples(videos, split)
    np.save(pred_dir / f"keyword_{split}.npy", kw.scores(X).astype(np.float32))
log["keyword"] = {"train_sec": 0.0, "params": 0}

# ---- TF-IDF + logistic regression ------------------------------------------------
tc = cfg["tfidf"]
m = TfidfLinear(len(labels), tc["max_features"], tc["ngram_range"])
t0 = time.perf_counter()
hist = m.fit(Xtr, Ytr, epochs=tc["epochs"], lr=tc["lr"], batch_size=tc["batch_size"])
log["tfidf_lr"] = {"train_sec": time.perf_counter() - t0, "params": m.n_params(), "loss": hist}
m.save(str(ckpt / "tfidf_lr.pkl"))
for split in ["val", "test"]:
    X, _, _ = data.eval_examples(videos, split)
    np.save(pred_dir / f"tfidf_lr_{split}.npy", m.predict_proba(X).astype(np.float32))

save_json(log, rpath("results/train_log_baselines.json"))
print({k: round(v["train_sec"], 1) for k, v in log.items()})
