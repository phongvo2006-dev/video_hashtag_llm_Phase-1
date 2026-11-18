"""Stage 3 — reinforcement fine-tuning of the hashtag policy (REINFORCE + RLOO baseline).

Why RL here? BCE training optimises each hashtag independently, but users see
(and we evaluate) a *set* of hashtags: what matters is set-level F1 — the right
number of tags, the right ones together. F1 is not differentiable, so we treat
the model as a stochastic policy and optimise it directly:

  policy      pi(S | caption) = prod_j Bernoulli(s_j ; sigmoid(z_j))
  action      sample K tag-sets S_1..S_K per caption
  reward      r(S) = F1(S, gold)                         (any black-box reward works,
                                                          e.g. an LLM-judge score)
  baseline    b_i = mean_{k != i} r(S_k)                 (RLOO, Ahmadian et al. 2024)
  loss        -(r_i - b_i) * log pi(S_i)  +  beta * BCE  (supervised anchor, like the
                                                          KL/SFT anchor in RLHF)

At inference the policy is decoded greedily (s_j = 1 iff p_j >= 0.5).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def f1_reward(S: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Vectorised example-F1 between binary sets. S: [..., L], Y broadcastable."""
    hit = (S * Y).sum(-1)
    denom = S.sum(-1) + Y.sum(-1)
    return torch.where(hit > 0, 2 * hit / denom.clamp(min=1), torch.zeros_like(hit))


def rl_finetune(model: nn.Module, ids: torch.Tensor, Y: np.ndarray, epochs: int, lr: float,
                batch_size: int, n_samples: int = 4, bce_anchor: float = 0.5,
                entropy_coef: float = 0.0, eval_fn=None, eval_every: int = 100,
                log=print) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    Yt = torch.from_numpy(Y)
    hist = {"step": [], "reward_sample": [], "reward_greedy": [], "val_setf1": []}
    step = 0
    if eval_fn is not None:
        hist["step"].append(0); hist["val_setf1"].append(eval_fn())
        hist["reward_sample"].append(float("nan")); hist["reward_greedy"].append(float("nan"))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ids))
        rs, rg = [], []
        for b in range(0, len(perm), batch_size):
            ix = perm[b:b + batch_size]
            y = Yt[ix]                                           # [B, L]
            logits = model(ids[ix])                              # [B, L]
            probs = torch.sigmoid(logits)
            with torch.no_grad():
                S = torch.bernoulli(probs.unsqueeze(1).expand(-1, n_samples, -1))   # [B, K, L]
                r = f1_reward(S, y.unsqueeze(1))                                    # [B, K]
                base = (r.sum(1, keepdim=True) - r) / (n_samples - 1)               # RLOO
                adv = r - base
                greedy = f1_reward((probs >= 0.5).float(), y)
            # log pi(S) = -BCE(logits, S) summed over labels
            logp = -F.binary_cross_entropy_with_logits(
                logits.unsqueeze(1).expand_as(S), S, reduction="none").sum(-1)     # [B, K]
            pg = -(adv * logp).mean()
            loss = pg + bce_anchor * F.binary_cross_entropy_with_logits(logits, y)
            if entropy_coef:
                ent = -(probs * probs.clamp_min(1e-8).log()
                        + (1 - probs) * (1 - probs).clamp_min(1e-8).log()).sum(-1).mean()
                loss = loss - entropy_coef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            rs.append(r.mean().item()); rg.append(greedy.mean().item())
            if eval_fn is not None and step % eval_every == 0:
                hist["step"].append(step); hist["val_setf1"].append(eval_fn())
                hist["reward_sample"].append(float(np.mean(rs[-eval_every:])))
                hist["reward_greedy"].append(float(np.mean(rg[-eval_every:])))
                model.train()
        log(f"  [rl] epoch {ep + 1}/{epochs} reward(sample)={np.mean(rs):.4f} "
            f"reward(greedy)={np.mean(rg):.4f}")
    return hist
