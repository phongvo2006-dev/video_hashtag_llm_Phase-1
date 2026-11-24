"""Reinforcement fine-tuning of the RoBERTa hashtag policy (REINFORCE + RLOO baseline).

The training loop lives in scripts/04_finetune_roberta.py; this module holds the reward.

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

import torch


def f1_reward(S: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Vectorised example-F1 between binary sets. S: [..., L], Y broadcastable."""
    hit = (S * Y).sum(-1)
    denom = S.sum(-1) + Y.sum(-1)
    return torch.where(hit > 0, 2 * hit / denom.clamp(min=1), torch.zeros_like(hit))
