"""Provider-agnostic LLM client with an on-disk response cache.

Providers
---------
cache      : read responses from ``data/llm_cache/<stage>.jsonl`` only (no network, no key).
anthropic  : Claude via the official SDK  (env ANTHROPIC_API_KEY)
openai     : any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama ``/v1``)
             env OPENAI_API_KEY, optional OPENAI_BASE_URL
ollama     : shortcut for openai provider with base_url=http://localhost:11434/v1

Every live response is appended to the cache, so experiments re-run for free and
results in ``results/`` are reproducible without an API key.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path


def prompt_key(model: str, system: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\n{system}\n{prompt}".encode()).hexdigest()[:16]


class LLMClient:
    def __init__(self, provider: str, model: str, cache_file: str | Path):
        self.provider, self.model = provider, model
        self.cache_file = Path(cache_file)
        self.cache: dict[str, dict] = {}
        if self.cache_file.exists():
            for line in open(self.cache_file):
                r = json.loads(line)
                self.cache[r["key"]] = r
        self._client = None
        self.stats = {"calls": 0, "cache_hits": 0, "latency_sec": [], "in_tokens": [], "out_tokens": []}

    # ------------------------------------------------------------------ providers
    def _live(self, system: str, prompt: str) -> tuple[str, int, int]:
        if self.provider == "anthropic":
            import anthropic
            self._client = self._client or anthropic.Anthropic()
            r = self._client.messages.create(model=self.model, max_tokens=200, temperature=0,
                                             system=system,
                                             messages=[{"role": "user", "content": prompt}])
            return r.content[0].text, r.usage.input_tokens, r.usage.output_tokens
        if self.provider in ("openai", "ollama"):
            from openai import OpenAI
            base = os.environ.get("OPENAI_BASE_URL") or (
                "http://localhost:11434/v1" if self.provider == "ollama" else None)
            self._client = self._client or OpenAI(base_url=base,
                                                  api_key=os.environ.get("OPENAI_API_KEY", "ollama"))
            r = self._client.chat.completions.create(
                model=self.model, temperature=0, max_tokens=200,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
            u = r.usage
            return r.choices[0].message.content, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0)
        raise RuntimeError(f"provider={self.provider!r} has no cached answer for this prompt; "
                           "set llm.provider to anthropic / openai / ollama to call a live model")

    # ------------------------------------------------------------------ public
    def complete(self, system: str, prompt: str, meta: dict | None = None) -> str:
        key = prompt_key(self.model, system, prompt)
        if key in self.cache:
            self.stats["cache_hits"] += 1
            r = self.cache[key]
        else:
            t0 = time.perf_counter()
            text, tin, tout = self._live(system, prompt)
            r = {"key": key, "model": self.model, "response": text, "in_tokens": tin,
                 "out_tokens": tout, "latency_sec": time.perf_counter() - t0, **(meta or {})}
            self.cache[key] = r
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "a") as f:
                f.write(json.dumps(r) + "\n")
        self.stats["calls"] += 1
        for k in ("in_tokens", "out_tokens", "latency_sec"):
            if r.get(k) is not None:
                self.stats[k].append(r[k])
        return r["response"]


def parse_json(text: str) -> dict:
    """Robustly pull the first JSON object out of an LLM reply."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def normalise(tags: list[str], allowed: set[str]) -> list[str]:
    """'#Cooking ' -> 'cooking'; drop tags outside the scored vocabulary; dedupe."""
    out = []
    for t in tags:
        t = re.sub(r"[^a-z0-9]", "", str(t).lower())
        if t in allowed and t not in out:
            out.append(t)
    return out


def enforce_constraints(tags: list[str], category_tags: set[str], k: int,
                        fill_from: list[str] | None = None) -> list[str]:
    """Post-validate an LLM answer: keep only the first category tag (prompts ask for exactly
    one; LLMs often add a second, e.g. #food + #cooking), cap at k, and optionally refill
    dropped slots from the fine-tuned model's ranked candidates (hybrid stage)."""
    out, seen_cat = [], False
    for t in tags:
        if t in category_tags:
            if seen_cat:
                continue
            seen_cat = True
        out.append(t)
    for t in fill_from or []:
        if len(out) >= k:
            break
        if t not in out and t not in category_tags:
            out.append(t)
    return out[:k]


def approx_tokens(text: str) -> int:
    """~4 chars/token heuristic, used only when a provider gives no usage numbers."""
    return max(1, round(len(text) / 4))
