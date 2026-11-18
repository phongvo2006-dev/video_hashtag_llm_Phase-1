import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hashtagger.llm.client import normalise, parse_json  # noqa: E402
from hashtagger.metrics import prf_at_k, set_f1  # noqa: E402
from hashtagger.models.common import LabelSpace  # noqa: E402
from hashtagger.models.rl import f1_reward  # noqa: E402


def test_prf_at_k():
    m = prf_at_k([["a", "b", "c"]], [["a", "c", "d", "e"]], k=3)
    assert abs(m["P@3"] - 2 / 3) < 1e-9 and abs(m["R@3"] - 0.5) < 1e-9


def test_set_f1_and_reward_agree():
    pred, gold = ["a", "b"], ["b", "c", "d"]
    ls = LabelSpace(["a", "b", "c", "d"], ["a"])
    S, Y = torch.tensor(ls.encode([pred])), torch.tensor(ls.encode([gold]))
    assert abs(f1_reward(S, Y).item() - set_f1([pred], [gold])["SetF1"]) < 1e-6


def test_threshold_min_tags():
    ls = LabelSpace(["a", "b", "c"], ["a"])
    assert ls.threshold(np.array([[0.1, 0.2, 0.05]]), 0.5, min_tags=1) == [["b"]]


def test_llm_parsing():
    js = parse_json('Sure!\n{"hashtags": ["#Cooking", "food", "unknowntag", "food"]}')
    assert normalise(js["hashtags"], {"cooking", "food"}) == ["cooking", "food"]
