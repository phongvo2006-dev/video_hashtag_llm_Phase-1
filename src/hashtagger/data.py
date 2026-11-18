"""Load MSR-VTT, build hashtag labels, and serve train / eval examples."""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import text


@dataclass
class Video:
    video_id: str
    split: str
    category: int
    captions: list[str]
    eval_caption: str = ""            # the single caption used as model input at eval time
    tags: list[str] = field(default_factory=list)  # gold hashtags (without '#')


def load_msrvtt(raw_json: str, train_end: int, val_end: int) -> list[Video]:
    d = json.load(open(raw_json))
    caps: dict[str, list[str]] = defaultdict(list)
    for s in sorted(d["sentences"], key=lambda s: (s["video_id"], s["sen_id"])):
        caps[s["video_id"]].append(s["caption"].strip())
    videos = []
    for v in d["videos"]:
        idx = int(v["video_id"].replace("video", ""))
        split = "train" if idx < train_end else ("val" if idx < val_end else "test")
        videos.append(Video(v["video_id"], split, int(v["category"]), caps[v["video_id"]]))
    videos.sort(key=lambda v: int(v.video_id[5:]))
    return videos


def build_labels(videos: list[Video], category_tags: dict[int, str], min_votes: int,
                 min_video_freq: int, max_vocab: int, seed: int = 42) -> dict:
    """Weak-supervision labelling from annotator consensus.

    Each MSR-VTT clip has 20 captions written by different people. A term becomes
    a gold hashtag for a clip when >= ``min_votes`` annotators independently
    mention it; the clip's category adds one category hashtag. A model only sees
    ONE caption at test time, so it has to infer tags the other 19 people saw.
    """
    text.ensure_nltk()
    all_caps = [c for v in videos for c in v.captions]
    tagged = text.tag_sents([text.tokenize(c) for c in all_caps])
    gerunds = text.learn_gerunds(tagged)

    # votes per video
    votes: dict[str, Counter] = {}
    i = 0
    for v in videos:
        cnt: Counter = Counter()
        for _ in v.captions:
            cnt.update(text.candidate_terms(tagged[i], gerunds))
            i += 1
        votes[v.video_id] = cnt

    # vocabulary from training videos only (no test leakage)
    vfreq: Counter = Counter()
    for v in videos:
        if v.split == "train":
            vfreq.update(t for t, n in votes[v.video_id].items() if n >= min_votes)
    cat_vocab = [category_tags[c] for c in sorted(category_tags)]
    # category names are reserved for the category hashtag, so "#cooking" always means
    # "this is a cooking video" rather than "someone mentioned cooking"
    topic_vocab = [t for t, n in vfreq.most_common()
                   if n >= min_video_freq and t not in set(cat_vocab)][:max_vocab]
    vocab = sorted(set(topic_vocab) | set(cat_vocab))
    topic_set = set(topic_vocab)

    rng = random.Random(seed)
    for v in videos:
        topics = [t for t, n in votes[v.video_id].most_common() if n >= min_votes and t in topic_set]
        v.tags = sorted(set(topics) | {category_tags[v.category]})
        v.eval_caption = rng.choice(v.captions)
    return {"vocab": vocab, "category_tags": cat_vocab, "gerunds": gerunds}


def save_processed(videos: list[Video], meta: dict, out_dir: str) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "videos.jsonl", "w") as f:
        for v in videos:
            f.write(json.dumps(v.__dict__) + "\n")
    json.dump(meta, open(p / "meta.json", "w"))


def load_processed(out_dir: str) -> tuple[list[Video], dict]:
    p = Path(out_dir)
    videos = [Video(**json.loads(l)) for l in open(p / "videos.jsonl")]
    meta = json.load(open(p / "meta.json"))
    return videos, meta


def train_examples(videos: list[Video]) -> tuple[list[str], list[list[str]]]:
    """Every caption of a training clip is one example (20x augmentation)."""
    X, Y = [], []
    for v in videos:
        if v.split == "train":
            for c in v.captions:
                X.append(c)
                Y.append(v.tags)
    return X, Y


def eval_examples(videos: list[Video], split: str) -> tuple[list[str], list[list[str]], list[Video]]:
    vs = [v for v in videos if v.split == split]
    return [v.eval_caption for v in vs], [v.tags for v in vs], vs


def other_caption_gains(videos: list[Video], gerunds: dict[str, str], category_tags: list[str]) -> list[dict]:
    """Graded relevance per gold tag = share of the clip's OTHER captions (input caption excluded)
    that mention it. The category tag gets gain 1.0."""
    text.ensure_nltk()
    out = []
    for v in videos:
        others = [c for c in v.captions if c != v.eval_caption] or v.captions
        cnt: Counter = Counter()
        for s in text.tag_sents([text.tokenize(c) for c in others]):
            cnt.update(text.candidate_terms(s, gerunds))
        g = {t: cnt.get(t, 0) / len(others) for t in v.tags if t not in category_tags}
        g[category_tags[v.category]] = 1.0
        out.append(g)
    return out
