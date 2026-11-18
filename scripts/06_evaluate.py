"""Step 6 — quality + efficiency evaluation of every model; writes results/metrics.json & results.md."""
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hashtagger import data  # noqa: E402
from hashtagger.metrics import category_accuracy, prf_at_k, set_f1  # noqa: E402
from hashtagger.models import transformer as T  # noqa: E402
from hashtagger.models.common import LabelSpace, tune_threshold  # noqa: E402
from hashtagger.models.keyword import KeywordTagger  # noqa: E402
from hashtagger.models.tfidf_linear import TfidfLinear  # noqa: E402
from hashtagger.utils import load_config, load_json, rpath, save_json  # noqa: E402

cfg = load_config()
K = cfg["eval"]["k"]
videos, meta = data.load_processed(str(rpath(cfg["data"]["processed_dir"])))
labels = LabelSpace(meta["vocab"], meta["category_tags"])
Xte, Yte, vte = data.eval_examples(videos, "test")
_, Yval, _ = data.eval_examples(videos, "val")
true_cat = [v.category for v in vte]
P = lambda name, split: np.load(rpath(f"results/preds/{name}_{split}.npy"))  # noqa: E731

NAMES = {"keyword": "Keyword rules (no learning)", "tfidf_lr": "TF-IDF + LogReg",
         "transformer_bce": "Transformer (BCE)", "transformer_rl": "Transformer + RL (ours)",
         "llm_zero_shot": "LLM zero-shot", "llm_hybrid": "Transformer+RL → LLM rerank (ours)"}


def sets_for(name, scores, val_scores):
    if name == "keyword":
        return ([[labels.vocab[j] for j in np.argsort(-r) if r[j] >= 1.0] for r in scores], None)
    thr = tune_threshold(val_scores, Yval, labels)
    return labels.threshold(scores, 0.5), (labels.threshold(scores, thr), thr)


# ============================ 1. full test set (2,990 videos) =========================
rows, full = [], {}
for name in ["keyword", "tfidf_lr", "transformer_bce", "transformer_rl"]:
    S, Sv = P(name, "test"), P(name, "val")
    m = prf_at_k(labels.topk(S, K), Yte, K)
    sets05, tuned = sets_for(name, S, Sv)
    m.update({f"{k}@0.5": v for k, v in set_f1(sets05, Yte).items()})
    if tuned:
        m.update({f"{k}@tuned": v for k, v in set_f1(tuned[0], Yte).items()}, tuned_thr=tuned[1])
    m["CatAcc"] = category_accuracy(S, labels.cat_idx, true_cat)
    full[name] = m

# ============================ 2. LLM subset (200 videos) ==============================
sub_ids = load_json(rpath("results/llm_subset_ids.json"))
pos = {v.video_id: i for i, v in enumerate(vte)}
sub = [pos[v] for v in sub_ids]
Ysub = [Yte[i] for i in sub]
subset = {}
for name in ["keyword", "tfidf_lr", "transformer_bce", "transformer_rl"]:
    S = P(name, "test")[sub]
    m = prf_at_k(labels.topk(S, K), Ysub, K)
    sets05, _ = sets_for(name, S, P(name, "val"))
    m.update(set_f1(sets05, Ysub))
    m["CatAcc"] = category_accuracy(S, labels.cat_idx, [true_cat[i] for i in sub])
    subset[name] = m
for stage in ["zero_shot", "hybrid"]:
    f = rpath(f"results/preds/llm_{stage}_test_subset.json")
    if not f.exists():
        continue
    pr = load_json(f)
    preds = [pr[v]["hashtags"] for v in sub_ids]
    m = prf_at_k(preds, Ysub, K)
    m.update(set_f1(preds, Ysub))
    cats = set(labels.category_tags)
    m["CatAcc"] = float(np.mean([next((t for t in p if t in cats), None) == labels.category_tags[true_cat[i]]
                                 for p, i in zip(preds, sub)]))
    subset[f"llm_{stage}"] = m

# ============================ 2b. paired bootstrap CIs ===============================
def per_example_f1(pred_sets, gold):
    return np.array([0.0 if not (set(p) & set(g)) else 2 * len(set(p) & set(g)) / (len(set(p)) + len(set(g)))
                     for p, g in zip(pred_sets, gold)])


def boot(a, b, n=5000, seed=0):
    d = a - b
    rs = np.random.RandomState(seed)
    m = d[rs.randint(0, len(d), (n, len(d)))].mean(1)
    return {"delta": float(d.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}


bce, rl = P("transformer_bce", "test"), P("transformer_rl", "test")
thr_bce = full["transformer_bce"]["tuned_thr"]
significance = {
    "RL vs BCE (SetF1, thr 0.5, full test)": boot(per_example_f1(labels.threshold(rl, 0.5), Yte),
                                                  per_example_f1(labels.threshold(bce, 0.5), Yte)),
    "RL vs BCE+tuned threshold (SetF1, full test)": boot(per_example_f1(labels.threshold(rl, 0.5), Yte),
                                                         per_example_f1(labels.threshold(bce, thr_bce), Yte)),
}
f = rpath("results/preds/llm_hybrid_test_subset.json")
if f.exists():
    hy = load_json(f)
    significance["Hybrid vs RL (F1@5, LLM subset)"] = boot(
        per_example_f1([hy[v]["hashtags"] for v in sub_ids], Ysub),
        per_example_f1(labels.topk(rl[sub], K), Ysub))
    zs = load_json(rpath("results/preds/llm_zero_shot_test_subset.json"))
    significance["Hybrid vs LLM zero-shot (F1@5, LLM subset)"] = boot(
        per_example_f1([hy[v]["hashtags"] for v in sub_ids], Ysub),
        per_example_f1([zs[v]["hashtags"] for v in sub_ids], Ysub))

# ============================ 3. per-category F1@K (test) =============================
per_cat = {}
for name in ["keyword", "tfidf_lr", "transformer_rl"]:
    top = labels.topk(P(name, "test"), K)
    per_cat[name] = {}
    for c, tag in enumerate(labels.category_tags):
        ix = [i for i, t in enumerate(true_cat) if t == c]
        per_cat[name][tag] = prf_at_k([top[i] for i in ix], [Yte[i] for i in ix], K)[f"F1@{K}"]

# ============================ 4. efficiency ===========================================
torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "1")))
bench_caps = Xte[:300]


def bench(fn, batch_caps):
    fn(bench_caps[:5])                                        # warm-up
    lat = []
    for c in bench_caps[:200]:
        t = time.perf_counter(); fn([c]); lat.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter(); fn(batch_caps); thr = len(batch_caps) / (time.perf_counter() - t)
    return statistics.median(lat), float(np.percentile(lat, 95)), thr


ckpt = rpath("checkpoints")
mb = lambda p: os.path.getsize(p) / 2**20  # noqa: E731
eff = {}
kw = KeywordTagger(labels, meta["gerunds"], np.load(ckpt / "keyword_prior.npy"))
eff["keyword"] = (*bench(kw.scores, Xte), 0.0, 0, 0.0)
tf = TfidfLinear.load(str(ckpt / "tfidf_lr.pkl"))
logs = {"tfidf_lr": load_json(rpath("results/train_log_baselines.json"))["tfidf_lr"],
        "transformer_bce": load_json(rpath("results/train_log_transformer.json")),
        "transformer_rl": load_json(rpath("results/train_log_rl.json"))}
eff["tfidf_lr"] = (*bench(tf.predict_proba, Xte), mb(ckpt / "tfidf_lr.pkl"), logs["tfidf_lr"]["params"],
                   logs["tfidf_lr"]["train_sec"])
for name in ["transformer_bce", "transformer_rl"]:
    mdl, tok, _ = T.load(str(ckpt / f"{name}.pt"))
    fn = lambda caps, mdl=mdl, tok=tok: T.predict_proba(mdl, tok.encode(caps, mdl.cfg["max_len"]))  # noqa: E731
    tr = logs[name]["train_sec"] + (logs["transformer_bce"]["train_sec"] if name == "transformer_rl" else 0)
    eff[name] = (*bench(fn, Xte), mb(ckpt / f"{name}.pt"), mdl.n_params(), tr)
efficiency = {n: dict(zip(["latency_ms_p50", "latency_ms_p95", "throughput_per_sec", "size_mb", "params",
                           "train_sec"], v)) for n, v in eff.items()}
for stage in ["zero_shot", "hybrid"]:
    f = rpath(f"results/llm_{stage}_usage.json")
    if f.exists():
        efficiency[f"llm_{stage}"] = load_json(f)

save_json({"k": K, "test_full": full, "test_llm_subset": subset, "per_category_f1": per_cat,
           "efficiency": efficiency, "significance": significance, "n_test": len(Xte), "n_subset": len(sub)},
          rpath("results/metrics.json"))

# ============================ 5. markdown tables =====================================
f3 = lambda x: f"{x:.3f}"  # noqa: E731
t1 = [[NAMES[n], f3(m[f"P@{K}"]), f3(m[f"R@{K}"]), f3(m[f"F1@{K}"]), f3(m["SetF1@0.5"]),
       f3(m.get("SetF1@tuned", m["SetF1@0.5"])), f"{m['avg_tags@0.5']:.1f}", f3(m["CatAcc"])]
      for n, m in full.items()]
t2 = [[NAMES[n], f3(m[f"P@{K}"]), f3(m[f"R@{K}"]), f3(m[f"F1@{K}"]), f3(m["SetF1"]), f"{m['avg_tags']:.1f}",
       f3(m["CatAcc"])] for n, m in subset.items()]
t3 = [[NAMES[n], f"{e['latency_ms_p50']:.2f}", f"{e['latency_ms_p95']:.2f}", f"{e['throughput_per_sec']:,.0f}",
       f"{e['size_mb']:.1f}", f"{e['params']/1e6:.2f}M" if e["params"] else "–", f"{e['train_sec']/60:.1f}"]
      for n, e in efficiency.items() if not n.startswith("llm")]
md = ["### Test set — 2,990 videos, one caption each\n",
      tabulate(t1, ["Model", f"P@{K}", f"R@{K}", f"F1@{K}", "SetF1 (thr 0.5)", "SetF1 (val-tuned thr)",
                    "#tags", "Category acc"], tablefmt="github"),
      f"\n\n### LLM comparison — fixed random subset of {len(sub)} test videos\n",
      tabulate(t2, ["Model", f"P@{K}", f"R@{K}", f"F1@{K}", "SetF1", "#tags", "Category acc"], tablefmt="github"),
      "\n\n### Efficiency — CPU, 1 thread\n",
      tabulate(t3, ["Model", "Latency p50 (ms)", "p95 (ms)", "Throughput (captions/s, batch)", "Size (MB)",
                    "Params", "Train (min, 2 vCPU)"], tablefmt="github")]
t4 = [[k, f"{v['delta']:+.3f}", f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]",
       "yes" if v["ci95"][0] > 0 or v["ci95"][1] < 0 else "no"] for k, v in significance.items()]
t5 = [[NAMES[n], f"{e['mean_in_tokens']:.0f}", f"{e['mean_out_tokens']:.0f}", e["calls"]]
      for n, e in efficiency.items() if n.startswith("llm")]
md += ["\n\n### Paired bootstrap (5,000 resamples)\n",
       tabulate(t4, ["Comparison", "Δ", "95% CI", "Significant"], tablefmt="github"),
       "\n\n### LLM cost per video\n",
       tabulate(t5, ["Stage", "Input tokens", "Output tokens", "Calls"], tablefmt="github")]
(rpath("results/results.md")).write_text("\n".join(md) + "\n")
print("\n".join(md))
