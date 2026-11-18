from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | None = None) -> dict:
    path = path or os.environ.get("HASHTAG_CONFIG", str(ROOT / "configs" / "default.yaml"))
    return yaml.safe_load(open(path))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def rpath(p: str) -> Path:
    """Resolve a repo-relative path."""
    return ROOT / p


@contextmanager
def timer(store: dict, key: str):
    t = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2)


def load_json(path):
    return json.load(open(path))
