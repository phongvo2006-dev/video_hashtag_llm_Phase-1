"""Step 8 — fine-tune pretrained RoBERTa-base (BCE), then RL (REINFORCE + RLOO, set-F1 reward).

Resumable: a full training state (weights + optimizer + scheduler + position) is saved every
SAVE_EVERY steps and at each epoch end; `python scripts/04_finetune_roberta.py --resume` continues.

Leakage guards
  * label vocabulary, label prior and all training captions come from the TRAIN split only
  * the tokenizer is the frozen pretrained byte-BPE (nothing is fit on val/test)
  * checkpoint selection and thresholds use the VALIDATION split only
  * the test split is predicted exactly once, after training is finished
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hashtagger import data  # noqa: E402
from hashtagger.metrics import prf_at_k, set_f1  # noqa: E402
from hashtagger.models.common import LabelSpace, tune_threshold  # noqa: E402
from hashtagger.models.pretrained import RobertaTagger, collate, predict_proba  # noqa: E402
from hashtagger.models.rl import f1_reward  # noqa: E402
from hashtagger.utils import load_config, load_json, rpath, save_json, seed_everything  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--resume", action="store_true")
args = ap.parse_args()
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "200"))

cfg = load_config()
rc = cfg["roberta"]
seed_everything(cfg["seed"])
torch.set_num_threads(max(1, torch.get_num_threads()))
videos, meta = data.load_processed(str(rpath(cfg["data"]["processed_dir"])))
labels = LabelSpace(meta["vocab"], meta["category_tags"])
train = [v for v in videos if v.split == "train"]
Xval, Yval, _ = data.eval_examples(videos, "val")
Xte, _, _ = data.eval_examples(videos, "test")
SMOKE = os.environ.get("SMOKE") == "1"                 # tiny end-to-end run to test the code path
if SMOKE:
    train, Xval, Yval, Xte = train[:48], Xval[:40], Yval[:40], Xte[:40]

model = RobertaTagger(len(labels), rc["source"], rc["dropout"])
enc = model.backbone.encode_text
train_ids = [enc(v.captions, rc["max_len"]) for v in train]        # [video][caption] -> ids
train_Y = labels.encode([v.tags for v in train])
val_ids, test_ids = enc(Xval, rc["max_len"]), enc(Xte, rc["max_len"])
n_params = sum(p.numel() for p in model.parameters())
print(f"RoBERTa tagger: {n_params / 1e6:.1f}M params, {len(train)} train clips")

with torch.no_grad():                                                # head bias = label prior
    prior = np.clip(train_Y.mean(0), 1e-4, 1 - 1e-4)
    model.head.bias.copy_(torch.from_numpy(np.log(prior / (1 - prior))).float())


def epoch_examples(k: int, epoch: int):
    rng = np.random.RandomState(cfg["seed"] * 100 + epoch)
    ex = [(caps[j], i) for i, caps in enumerate(train_ids)
          for j in rng.choice(len(caps), k, replace=False)]
    rng.shuffle(ex)
    return ex


def val_scores():
    p = predict_proba(model, val_ids)
    return p, {**prf_at_k(labels.topk(p, 5), Yval, 5), **set_f1(labels.threshold(p, 0.5), Yval)}


ckpt = rpath("checkpoints/smoke" if SMOKE else "checkpoints"); ckpt.mkdir(parents=True, exist_ok=True)
log = {"params": n_params, "source": rc["source"], "bce": [], "rl": []}

# ------------------------------------------------------------------ 1. supervised (BCE)
bs, E = rc["batch_size"], rc["epochs"]
steps = E * ((len(train) * rc["captions_per_video"] + bs - 1) // bs)
opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": rc["lr_encoder"]},
                         {"params": model.head.parameters(), "lr": rc["lr_head"]}], weight_decay=0.01)
warm = int(rc["warmup"] * steps)
sched = torch.optim.lr_scheduler.LambdaLR(            # linear warm-up, then linear decay
    opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / (steps - warm)))
Yt = torch.from_numpy(train_Y)
state_path = ckpt / "roberta_state.pt"
best, step, start_ep, start_b, elapsed, bce_done = -1.0, 0, 0, 0, 0.0, False


def save_state(ep, next_b, tot):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "epoch": ep, "next_b": next_b, "step": step, "best": best, "tot": tot, "log": log,
                "elapsed": elapsed + time.perf_counter() - t0, "bce_done": bce_done}, str(state_path) + ".tmp")
    os.replace(str(state_path) + ".tmp", state_path)       # atomic: a kill mid-write never corrupts it


tot_resume = 0.0
if args.resume and state_path.exists():                   # exact resume
    st = torch.load(state_path, weights_only=False)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    start_ep, start_b, step, best = st["epoch"], st["next_b"], st["step"], st["best"]
    log, elapsed, bce_done, tot_resume = st["log"], st["elapsed"], st["bce_done"], st["tot"]
    print(f"resumed exactly: epoch {start_ep + 1}, step {step}", flush=True)
elif args.resume and (rpath("results/roberta_resume.json")).exists():   # weights-only resume
    info = load_json(rpath("results/roberta_resume.json"))
    model.load_state_dict(torch.load(ckpt / "roberta_bce.pt"))
    start_ep, best, elapsed, log["bce"] = info["epochs_done"], info["best_val_f1"], info["elapsed_sec"], info["bce"]
    step = start_ep * ((len(train) * rc["captions_per_video"] + bs - 1) // bs)
    for _ in range(step):                                 # fast-forward the LR schedule
        sched.step()
    log["resume_note"] = info["note"]
    print(f"resumed from epoch-{start_ep} weights (fresh AdamW moments), step {step}", flush=True)

t0 = time.perf_counter()
for ep in range(start_ep, E):
    if bce_done:
        break
    model.train()
    ex = epoch_examples(rc["captions_per_video"], ep)
    tot = tot_resume if ep == start_ep else 0.0
    for b in range(start_b if ep == start_ep else 0, len(ex), bs):
        batch = ex[b:b + bs]
        ids, mask = collate([s for s, _ in batch])
        y = Yt[[i for _, i in batch]]
        loss = F.binary_cross_entropy_with_logits(model(ids, mask), y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        tot += loss.item() * len(batch)
        if step % 100 == 0:
            print(f"  [roberta] ep {ep + 1} step {step}/{steps} bce={tot / (b + len(batch)):.4f} "
                  f"{(elapsed + time.perf_counter() - t0) / 60:.1f} min", flush=True)
        if step % SAVE_EVERY == 0:
            save_state(ep, b + bs, tot)
    _, m = val_scores()
    log["bce"].append({"epoch": ep + 1, "train_bce": tot / len(ex), **m})
    print(f"  [roberta] epoch {ep + 1} val {m}", flush=True)
    if m["F1@5"] > best:                                  # model selection on VALIDATION only
        best = m["F1@5"]
        torch.save(model.state_dict(), ckpt / "roberta_bce.pt")
    save_state(ep + 1, 0, 0.0)
bce_done = True
log["bce_train_sec"] = elapsed + time.perf_counter() - t0
save_state(E, 0, 0.0)
model.load_state_dict(torch.load(ckpt / "roberta_bce.pt"))

pred_dir = rpath("results/preds/smoke" if SMOKE else "results/preds"); pred_dir.mkdir(parents=True, exist_ok=True)
pv, _ = val_scores()
np.save(pred_dir / "roberta_bce_val.npy", pv)
np.save(pred_dir / "roberta_bce_test.npy", predict_proba(model, test_ids))
log["bce_val_threshold"] = tune_threshold(pv, Yval, labels)

# ------------------------------------------------------------------ 2. RL fine-tuning
K = rc["rl_n_samples"]
opt = torch.optim.AdamW(model.parameters(), lr=rc["rl_lr"], weight_decay=0.0)
ex = epoch_examples(rc["rl_captions_per_video"], 99)
t0 = time.perf_counter()
_, m0 = val_scores()
log["rl"].append({"step": 0, **m0})
for n, b in enumerate(range(0, len(ex), bs), 1):
    model.train()
    batch = ex[b:b + bs]
    ids, mask = collate([s for s, _ in batch])
    y = Yt[[i for _, i in batch]]
    logits = model(ids, mask)
    probs = torch.sigmoid(logits)
    with torch.no_grad():
        S = torch.bernoulli(probs.unsqueeze(1).expand(-1, K, -1))
        r = f1_reward(S, y.unsqueeze(1))
        adv = r - (r.sum(1, keepdim=True) - r) / (K - 1)                 # RLOO baseline
    logp = -F.binary_cross_entropy_with_logits(logits.unsqueeze(1).expand_as(S), S,
                                               reduction="none").sum(-1)
    loss = -(adv * logp).mean() + rc["rl_bce_anchor"] * F.binary_cross_entropy_with_logits(logits, y)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if n % 100 == 0 or b + bs >= len(ex):
        _, m = val_scores()
        log["rl"].append({"step": n, "reward": r.mean().item(), **m})
        print(f"  [roberta-rl] step {n} reward={r.mean().item():.3f} val SetF1={m['SetF1']:.4f} "
              f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
log["rl_train_sec"] = time.perf_counter() - t0
torch.save(model.state_dict(), ckpt / "roberta_rl.pt")
pv, _ = val_scores()
np.save(pred_dir / "roberta_rl_val.npy", pv)
np.save(pred_dir / "roberta_rl_test.npy", predict_proba(model, test_ids))
log["rl_val_threshold"] = tune_threshold(pv, Yval, labels)
save_json(log, rpath("results/smoke_log.json" if SMOKE else "results/train_log_roberta.json"))
print("done", {k: v for k, v in log.items() if k.endswith("sec")})
