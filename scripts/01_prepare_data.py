"""Step 1 — load MSR-VTT, build consensus hashtag labels, write data/processed/."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hashtagger import data  # noqa: E402
from hashtagger.utils import load_config, rpath, save_json  # noqa: E402

cfg = load_config()
dc, lc = cfg["data"], cfg["labels"]
videos = data.load_msrvtt(str(rpath(dc["raw_json"])), **dc["split_boundaries"])
meta = data.build_labels(videos, {int(k): v for k, v in lc["category_tags"].items()},
                         lc["min_caption_votes"], lc["min_video_freq"], lc["max_vocab"], cfg["seed"])
data.save_processed(videos, meta, str(rpath(dc["processed_dir"])))

# ---- dataset card stats -------------------------------------------------------
splits = Counter(v.split for v in videos)
ntags = [len(v.tags) for v in videos]
tag_freq = Counter(t for v in videos if v.split == "train" for t in v.tags)
stats = {
    "videos": dict(splits),
    "captions_total": sum(len(v.captions) for v in videos),
    "train_examples": sum(len(v.captions) for v in videos if v.split == "train"),
    "vocab_size": len(meta["vocab"]),
    "category_tags": len(meta["category_tags"]),
    "tags_per_video_mean": float(np.mean(ntags)),
    "tags_per_video_median": float(np.median(ntags)),
    "top_tags": tag_freq.most_common(30),
    "examples": [{"video_id": v.video_id, "category": meta["category_tags"][v.category],
                  "eval_caption": v.eval_caption, "gold": v.tags}
                 for v in videos if v.split == "test"][:8],
}
save_json(stats, rpath("results/dataset_stats.json"))
print({k: stats[k] for k in ["videos", "captions_total", "train_examples", "vocab_size",
                             "tags_per_video_mean"]})
for e in stats["examples"]:
    print(e)
