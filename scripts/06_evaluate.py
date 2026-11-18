"""Step 6 — quality + efficiency evaluation of every model; writes results/metrics.json & results.md.

Primary metric: wnDCG@5 (annotator-agreement-weighted nDCG, ceiling 1.0, input caption excluded
from the weights). Also: nDCG@5, weighted recall, Hit@5, P/R/F1@5, Set-F1, category top-1/top-2/
macro-F1, paired bootstrap CIs, a 'clean' test split without verbatim train duplicates, and
CPU latency / throughput / size.
"""
import json
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
from hashtagger.metrics import category_suite, ndcg_at_k, ranking_suite, set_f1  # noqa: E402
from hashtagger.models.common import LabelSpace, tune_threshold  # noqa: E402
from hashtagger.utils import load_config, load_json, rpath, save_json  # noqa: E402

cfg = load_config()
K = cfg["eval"]["k"]
videos, meta = data.load_processed(str(rpath(cfg["data"]["processed_dir"])))
labels = LabelSpace(meta["vocab"], meta["category_tags"])
Xte, Yte, vte = data.eval_examples(videos, "test")
_, Yval, _ = data.eval_examples(videos, "val")
true_cat = [v.category for v in vte]
CATS = set(labels.category_tags)

gpath = rpath(cfg["data"]["processed_dir"]) / "test_gains.json"
if not gpath.exists():
    json.dump(data.other_caption_gains(vte, meta["gerunds"], labels.category_tags), open(gpath, "w"))
gains = json.load(open(gpath))

# 'clean' test: drop clips whose input caption appears verbatim among training captions
train_caps = {c.lower().strip() for v in videos if v.split == "train" for c in v.captions}
clean = [i for i, v in enumerate(vte) if v.eval_caption.lower().strip() not in train_caps]

NAMES = {"keyword": "Keyword rules (no learning)", "tfidf_lr": "TF-IDF + LogReg",
         "transformer_bce": "Transformer from scratch (BCE)", "transformer_rl": "Transformer from scratch + RL",
         "roberta_bce": "RoBERTa-base fine-tuned (BCE)", "roberta_rl": "RoBERTa-base fine-tuned + RL",
         "llm_zero_shot": "LLM zero-shot", "llm_hybrid": "Transformer+RL → LLM rerank"}
MODELS = [n for n in ["keyword", "tfidf_lr", "transformer_bce", "transformer_rl", "roberta_bce", "roberta_rl"]
          if rpath(f"results/preds/{n}_test.npy").exists()]
P = {n: (np.load(rpath(f"results/preds/{n}_test.npy")), np.load(rpath(f"results/preds/{n}_val.npy")))
     for n in MODELS}


def sets(name, S, thr=0.5):
    if name == "keyword":
        return [[labels.vocab[j] for j in np.argsort(-r) if r[j] >= 1.0] for r in S]
    return labels.threshold(S, thr)


def per_ex(metric, preds, idx):
    if metric == "wnDCG":
        return np.array([ndcg_at_k(p, gains[i], K) for p, i in zip(preds, idx)])
    return np.array([0.0 if not set(p) & set(Yte[i]) else
                     2 * len(set(p) & set(Yte[i])) / (len(set(p)) + len(set(Yte[i]))) for p, i in zip(preds, idx)])


def boot(a, b, n=5000):
    d = a - b
    m = d[np.random.RandomState(0).randint(0, len(d), (n, len(d)))].mean(1)
    return {"delta": float(d.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}


ALL = list(range(len(vte)))
# ============================ 1. full test set ========================================
full, top_cache, set_cache = {}, {}, {}
for n in MODELS:
    S, Sv = P[n]
    top_cache[n] = labels.topk(S, K)
    m = ranking_suite(top_cache[n], Yte, gains, K)
    set_cache[n] = sets(n, S)
    m.update({f"{k}@0.5": v for k, v in set_f1(set_cache[n], Yte).items()})
    if n != "keyword":
        thr = tune_threshold(Sv, Yval, labels)                          # tuned on VALIDATION
        m.update({f"{k}@tuned": v for k, v in set_f1(sets(n, S, thr), Yte).items()}, tuned_thr=thr)
    m.update(category_suite(S, labels.cat_idx, true_cat))
    full[n] = m

# ============================ 2. clean test (no verbatim train duplicates) =============
clean_res = {}
for n in MODELS:
    S = P[n][0][clean]
    m = ranking_suite(labels.topk(S, K), [Yte[i] for i in clean], [gains[i] for i in clean], K)
    m.update(category_suite(S, labels.cat_idx, [true_cat[i] for i in clean]))
    clean_res[n] = m

# ============================ 3. LLM subset ===========================================
sub_ids = load_json(rpath("results/llm_subset_ids.json"))
pos = {v.video_id: i for i, v in enumerate(vte)}
sub = [pos[v] for v in sub_ids]
subset, llm_preds = {}, {}
for n in MODELS:
    preds = [top_cache[n][i] for i in sub]
    m = ranking_suite(preds, [Yte[i] for i in sub], [gains[i] for i in sub], K)
    m.update(set_f1([set_cache[n][i] for i in sub], [Yte[i] for i in sub]))
    m["CatTop1"] = category_suite(P[n][0][sub], labels.cat_idx, [true_cat[i] for i in sub])["CatTop1"]
    subset[n] = m
for stage in ["zero_shot", "hybrid"]:
    f = rpath(f"results/preds/llm_{stage}_test_subset.json")
    if f.exists():
        pr = load_json(f)
        preds = llm_preds[f"llm_{stage}"] = [pr[v]["hashtags"] for v in sub_ids]
        m = ranking_suite(preds, [Yte[i] for i in sub], [gains[i] for i in sub], K)
        m.update(set_f1(preds, [Yte[i] for i in sub]))
        m["CatTop1"] = float(np.mean([next((t for t in p if t in CATS), None) == labels.category_tags[true_cat[i]]
                                      for p, i in zip(preds, sub)]))
        subset[f"llm_{stage}"] = m

# ============================ 4. paired bootstrap =====================================
sig = {}


def add(name, a, b):
    sig[name] = boot(a, b)


if "roberta_bce" in MODELS:
    add("RoBERTa-BCE vs Transformer-BCE (wnDCG@5)", per_ex("wnDCG", top_cache["roberta_bce"], ALL),
        per_ex("wnDCG", top_cache["transformer_bce"], ALL))
if "roberta_rl" in MODELS:
    add("RoBERTa+RL vs RoBERTa-BCE (Set-F1)", per_ex("f1", set_cache["roberta_rl"], ALL),
        per_ex("f1", set_cache["roberta_bce"], ALL))
    add("RoBERTa+RL vs RoBERTa-BCE (wnDCG@5)", per_ex("wnDCG", top_cache["roberta_rl"], ALL),
        per_ex("wnDCG", top_cache["roberta_bce"], ALL))
add("Transformer+RL vs Transformer-BCE (Set-F1)", per_ex("f1", set_cache["transformer_rl"], ALL),
    per_ex("f1", set_cache["transformer_bce"], ALL))
if "llm_hybrid" in llm_preds:
    rl_sub = [top_cache["transformer_rl"][i] for i in sub]
    for metric in ["wnDCG", "f1"]:
        tag = "wnDCG@5" if metric == "wnDCG" else "F1@5"
        add(f"LLM rerank vs Transformer+RL ({tag}, subset)", per_ex(metric, llm_preds["llm_hybrid"], sub),
            per_ex(metric, [p[:K] for p in rl_sub], sub))
    if "roberta_rl" in MODELS:
        add("RoBERTa+RL vs LLM rerank (wnDCG@5, subset)",
            per_ex("wnDCG", [top_cache["roberta_rl"][i] for i in sub], sub),
            per_ex("wnDCG", llm_preds["llm_hybrid"], sub))

# ============================ 5. per-category =========================================
per_cat = {}
for n in [m for m in ["keyword", "tfidf_lr", "transformer_rl", "roberta_rl"] if m in MODELS]:
    per_cat[n] = {}
    for c, tag in enumerate(labels.category_tags):
        ix = [i for i, t in enumerate(true_cat) if t == c]
        per_cat[n][tag] = float(per_ex("wnDCG", [top_cache[n][i] for i in ix], ix).mean())

# ============================ 6. efficiency ===========================================
torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "1")))
caps = Xte[:300]


def bench(fn, n_single=100, batch=256):
    fn(caps[:4])
    lat = []
    for c in caps[:n_single]:
        t = time.perf_counter(); fn([c]); lat.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter(); fn(caps[:batch]); thr = batch / (time.perf_counter() - t)
    return {"latency_ms_p50": statistics.median(lat), "latency_ms_p95": float(np.percentile(lat, 95)),
            "throughput_per_sec": thr}


ck = rpath("checkpoints")
mb = lambda p: os.path.getsize(p) / 2**20 if os.path.exists(p) else float("nan")  # noqa: E731
eff = {}
from hashtagger.models.keyword import KeywordTagger  # noqa: E402
from hashtagger.models.tfidf_linear import TfidfLinear  # noqa: E402
from hashtagger.models import transformer as T  # noqa: E402

kw = KeywordTagger(labels, meta["gerunds"], np.load(ck / "keyword_prior.npy"))
eff["keyword"] = {**bench(kw.scores), "size_mb": 0.0, "params": 0, "train_sec": 0.0}
tf = TfidfLinear.load(str(ck / "tfidf_lr.pkl"))
lb = load_json(rpath("results/train_log_baselines.json"))["tfidf_lr"]
eff["tfidf_lr"] = {**bench(tf.predict_proba), "size_mb": mb(ck / "tfidf_lr.pkl"), "params": lb["params"],
                   "train_sec": lb["train_sec"]}
lt, lr_ = load_json(rpath("results/train_log_transformer.json")), load_json(rpath("results/train_log_rl.json"))
for n, tr in [("transformer_bce", lt["train_sec"]), ("transformer_rl", lt["train_sec"] + lr_["train_sec"])]:
    mdl, tok, _ = T.load(str(ck / f"{n}.pt"))
    eff[n] = {**bench(lambda c, mdl=mdl, tok=tok: T.predict_proba(mdl, tok.encode(c, mdl.cfg["max_len"]))),
              "size_mb": mb(ck / f"{n}.pt"), "params": mdl.n_params(), "train_sec": tr}
if "roberta_rl" in MODELS:
    from hashtagger.models.pretrained import RobertaTagger, predict_proba as rp  # noqa: E402
    lg = load_json(rpath("results/train_log_roberta.json"))
    rob = RobertaTagger(len(labels), cfg["roberta"]["source"])
    rob.load_state_dict(torch.load(ck / "roberta_rl.pt"))
    fn = lambda c: rp(rob, rob.backbone.encode_text(c, cfg["roberta"]["max_len"]), 64)  # noqa: E731
    b = bench(fn, n_single=50, batch=128)
    eff["roberta_bce"] = {**b, "size_mb": mb(ck / "roberta_bce.pt"), "params": lg["params"],
                          "train_sec": lg["bce_train_sec"]}
    eff["roberta_rl"] = {**b, "size_mb": mb(ck / "roberta_rl.pt"), "params": lg["params"],
                         "train_sec": lg["bce_train_sec"] + lg["rl_train_sec"]}
for stage in ["zero_shot", "hybrid"]:
    f = rpath(f"results/llm_{stage}_usage.json")
    if f.exists():
        eff[f"llm_{stage}"] = load_json(f)

save_json({"k": K, "primary_metric": f"wnDCG@{K}", "test_full": full, "test_clean": clean_res,
           "test_llm_subset": subset, "significance": sig, "per_category_wndcg": per_cat, "efficiency": eff,
           "n_test": len(vte), "n_clean": len(clean), "n_subset": len(sub)}, rpath("results/metrics.json"))

# ============================ 7. markdown =============================================
f3 = lambda x: f"{x:.3f}"  # noqa: E731
md = [f"### Test set — {len(vte):,} clips, one caption each (primary metric: wnDCG@{K})\n",
      tabulate([[NAMES[n], f3(m[f"wnDCG@{K}"]), f3(m[f"nDCG@{K}"]), f3(m[f"wRecall@{K}"]), f"{m[f'Hit@{K}']:.1%}",
                 f3(m[f"F1@{K}"]), f3(m["SetF1@0.5"]), f3(m.get("SetF1@tuned", m["SetF1@0.5"])),
                 f"{m['CatTop1']:.1%}", f"{m['CatTop2']:.1%}", f3(m["CatMacroF1"])] for n, m in full.items()],
               ["Model", f"wnDCG@{K}", f"nDCG@{K}", f"wRecall@{K}", f"Hit@{K}", f"F1@{K}", "SetF1 (0.5)",
                "SetF1 (val thr)", "Cat top-1", "Cat top-2", "Cat macro-F1"], tablefmt="github"),
      f"\n\n### Clean test — {len(clean):,} clips whose input caption never appears verbatim in training\n",
      tabulate([[NAMES[n], f3(m[f"wnDCG@{K}"]), f3(m[f"F1@{K}"]), f"{m[f'Hit@{K}']:.1%}", f"{m['CatTop1']:.1%}"]
                for n, m in clean_res.items()], ["Model", f"wnDCG@{K}", f"F1@{K}", f"Hit@{K}", "Cat top-1"],
               tablefmt="github"),
      f"\n\n### LLM comparison — fixed random subset of {len(sub)} test clips\n",
      tabulate([[NAMES[n], f3(m[f"wnDCG@{K}"]), f3(m[f"nDCG@{K}"]), f"{m[f'Hit@{K}']:.1%}", f3(m[f"F1@{K}"]),
                 f3(m["SetF1"]), f"{m['CatTop1']:.1%}"] for n, m in subset.items()],
               ["Model", f"wnDCG@{K}", f"nDCG@{K}", f"Hit@{K}", f"F1@{K}", "SetF1", "Cat top-1"], tablefmt="github"),
      "\n\n### Paired bootstrap (5,000 resamples)\n",
      tabulate([[k, f"{v['delta']:+.3f}", f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]",
                 "yes" if v["ci95"][0] > 0 or v["ci95"][1] < 0 else "no"] for k, v in sig.items()],
               ["Comparison", "Δ", "95% CI", "Significant"], tablefmt="github"),
      "\n\n### Efficiency — CPU, 1 thread\n",
      tabulate([[NAMES[n], f"{e['latency_ms_p50']:.2f}", f"{e['latency_ms_p95']:.2f}",
                 f"{e['throughput_per_sec']:,.0f}", f"{e['size_mb']:.0f}",
                 f"{e['params'] / 1e6:.1f}M" if e["params"] else "–", f"{e['train_sec'] / 60:.1f}"]
                for n, e in eff.items() if not n.startswith("llm")],
               ["Model", "Latency p50 (ms)", "p95 (ms)", "Throughput (captions/s)", "Size (MB)", "Params",
                "Train (min, 2 vCPU)"], tablefmt="github"),
      "\n\n### LLM cost per clip (tokens estimated at ~4 chars/token)\n",
      tabulate([[NAMES[n], f"{e['mean_in_tokens']:.0f}", f"{e['mean_out_tokens']:.0f}", e["calls"]]
                for n, e in eff.items() if n.startswith("llm")], ["Stage", "Input tokens", "Output tokens", "Calls"],
               tablefmt="github")]
rpath("results/results.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
