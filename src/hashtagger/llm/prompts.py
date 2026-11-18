"""Prompt templates for the LLM stages."""
from __future__ import annotations

SYSTEM = (
    "You are a social-media assistant that writes hashtags for short videos. "
    "You only see a one-sentence caption describing the video. Answer with JSON only."
)

ZERO_SHOT = """Caption: "{caption}"

Pick the {k} hashtags that best describe this video, most relevant first.
Use ONLY tags from this allowed list (no '#', lowercase):
{allowed}

Include exactly one video-category tag from: {categories}.
Return JSON: {{"hashtags": ["tag1", "tag2", ...]}}"""

HYBRID = """Caption: "{caption}"

A fine-tuned hashtag model proposed these candidates (with its confidence):
{candidates}

Task: choose the {k} best hashtags for this video, most relevant first.
- Prefer candidates; drop ones that contradict the caption even if confident.
- You may add at most ONE tag from the allowed list that the model missed.
- Include exactly one video-category tag from: {categories}.
Also suggest up to 2 creative, open-vocabulary hashtags for reach (not scored).
Return JSON: {{"hashtags": [...], "creative": [...]}}"""


def zero_shot(caption: str, allowed: list[str], categories: list[str], k: int = 5) -> str:
    return ZERO_SHOT.format(caption=caption, k=k, allowed=", ".join(allowed),
                            categories=", ".join(categories))


def hybrid(caption: str, candidates: list[tuple[str, float]], categories: list[str], k: int = 5) -> str:
    cand = "\n".join(f"- {t} ({p:.2f})" for t, p in candidates)
    return HYBRID.format(caption=caption, candidates=cand, k=k, categories=", ".join(categories))
