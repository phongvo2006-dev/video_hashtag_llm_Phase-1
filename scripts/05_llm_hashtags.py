"""Step 5 — LLM stages on a fixed test subset.

  zero_shot : LLM alone, caption -> 5 hashtags from the allowed vocabulary
  hybrid    : RoBERTa + RL (the best fine-tuned tagger) proposes top-12 candidates + confidences,
              the LLM re-ranks / filters / adds one missed tag (+ creative tags)

Usage
  python scripts/05_llm_hashtags.py --stage zero_shot              # uses configs/default.yaml llm.provider
  python scripts/05_llm_hashtags.py --stage hybrid --provider anthropic
  # offline / batch workflow (e.g. Anthropic Batch API or any chat UI):
  python scripts/05_llm_hashtags.py --stage hybrid --export-pending pending.jsonl
  python scripts/05_llm_hashtags.py --stage hybrid --import-answers answers.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hashtagger import data  # noqa: E402
from hashtagger.llm import prompts  # noqa: E402
from hashtagger.llm.client import (LLMClient, approx_tokens, enforce_constraints, normalise,
                                   parse_json, prompt_key)  # noqa: E402
from hashtagger.models.common import LabelSpace  # noqa: E402
from hashtagger.utils import load_config, rpath, save_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["zero_shot", "hybrid"], required=True)
ap.add_argument("--provider")
ap.add_argument("--model")
ap.add_argument("--export-pending")
ap.add_argument("--import-answers")
args = ap.parse_args()

cfg = load_config()
lc, k = cfg["llm"], cfg["llm"]["final_tags"]
provider, model = args.provider or lc["provider"], args.model or lc["model"]
videos, meta = data.load_processed(str(rpath(cfg["data"]["processed_dir"])))
labels = LabelSpace(meta["vocab"], meta["category_tags"])
Xte, _, vte = data.eval_examples(videos, "test")

rng = np.random.RandomState(cfg["seed"])
subset = sorted(rng.choice(len(vte), cfg["eval"]["llm_subset_size"], replace=False).tolist())
save_json([vte[i].video_id for i in subset], rpath("results/llm_subset_ids.json"))

if args.stage == "hybrid":
    P = np.load(rpath("results/preds/roberta_rl_test.npy"))

jobs, cands = [], {}
for i in subset:
    v = vte[i]
    if args.stage == "zero_shot":
        p = prompts.zero_shot(v.eval_caption, labels.vocab, labels.category_tags, k)
    else:
        top = np.argsort(-P[i])[: lc["candidates"]]
        cands[v.video_id] = [labels.vocab[j] for j in top]
        p = prompts.hybrid(v.eval_caption, [(labels.vocab[j], float(P[i, j])) for j in top],
                           labels.category_tags, k)
    jobs.append((v.video_id, p))

cache_file = rpath(f"data/llm_cache/{args.stage}.jsonl")
client = LLMClient(provider, model, cache_file)

if args.import_answers:          # {"video_id": ..., "response": "<raw LLM text>"} per line
    ans = {json.loads(l)["video_id"]: json.loads(l)["response"] for l in open(args.import_answers)}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "a") as f:
        for vid, p in jobs:
            key = prompt_key(model, prompts.SYSTEM, p)
            if vid in ans and key not in client.cache:
                f.write(json.dumps({"key": key, "model": model, "response": ans[vid], "video_id": vid,
                                    "in_tokens": approx_tokens(prompts.SYSTEM + p),
                                    "out_tokens": approx_tokens(ans[vid]), "latency_sec": None}) + "\n")
    client = LLMClient(provider, model, cache_file)

if args.export_pending:
    with open(args.export_pending, "w") as f:
        for vid, p in jobs:
            if prompt_key(model, prompts.SYSTEM, p) not in client.cache:
                f.write(json.dumps({"video_id": vid, "system": prompts.SYSTEM, "prompt": p}) + "\n")
    print(f"wrote pending prompts -> {args.export_pending}")
    sys.exit(0)

out = {}
allowed = set(labels.vocab)
for vid, p in jobs:
    raw = client.complete(prompts.SYSTEM, p, {"video_id": vid})
    js = parse_json(raw)
    tags = normalise(js.get("hashtags", []), allowed)
    out[vid] = {"raw": js.get("hashtags", []),
                "hashtags": enforce_constraints(tags, set(labels.category_tags), k, cands.get(vid)),
                "creative": js.get("creative", [])}
save_json(out, rpath(f"results/preds/llm_{args.stage}_test_subset.json"))
s = client.stats
save_json({"provider": provider, "model": model, "calls": s["calls"], "cache_hits": s["cache_hits"],
           "mean_in_tokens": float(np.mean(s["in_tokens"])) if s["in_tokens"] else None,
           "mean_out_tokens": float(np.mean(s["out_tokens"])) if s["out_tokens"] else None,
           "mean_latency_sec": float(np.mean(s["latency_sec"])) if s["latency_sec"] else None},
          rpath(f"results/llm_{args.stage}_usage.json"))
print(f"{args.stage}: {len(out)} videos, cache hits {s['cache_hits']}/{s['calls']}")
