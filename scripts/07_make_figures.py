"""Step 7 — figures for the README (results/figures/*.png)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hashtagger.utils import load_json, rpath  # noqa: E402

M = load_json(rpath("results/metrics.json"))
FIG = rpath("results/figures"); FIG.mkdir(parents=True, exist_ok=True)
K = M["k"]
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
INK, INK2 = "#0b0b0b", "#52514e"
NAMES = {"keyword": "Keyword rules", "tfidf_lr": "TF-IDF + LogReg", "transformer_bce": "Transformer (BCE)",
         "transformer_rl": "Transformer + RL", "llm_zero_shot": "LLM zero-shot",
         "llm_hybrid": "Transformer+RL → LLM"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def hbar(ax, labels, values, colors, fmt="{:.3f}"):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.6)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.01, yi, fmt.format(v), va="center", color=INK, fontsize=9)
    ax.set_yticks(y, labels)
    ax.grid(axis="x", color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(0, max(values) * 1.15)


# 1. quality on full test set ---------------------------------------------------------
full = M["test_full"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
names = list(full)
hbar(axes[0], [NAMES[n] for n in names], [full[n][f"F1@{K}"] for n in names],
     [BLUE if n == "transformer_rl" else GRAY for n in names])
axes[0].set_title(f"F1@{K} (ranked top-{K})", color=INK, loc="left")
hbar(axes[1], [NAMES[n] for n in names], [full[n]["SetF1@0.5"] for n in names],
     [BLUE if n == "transformer_rl" else GRAY for n in names])
axes[1].set_title("Set-F1 (model picks how many tags)", color=INK, loc="left")
fig.suptitle(f"Test set: {M['n_test']:,} MSR-VTT videos, one caption each", color=INK, x=0.01, ha="left", y=1.04)
fig.savefig(FIG / "quality_full_test.png"); plt.close(fig)

# 2. LLM subset ---------------------------------------------------------------------------
sub = M["test_llm_subset"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
names = list(sub)
col = [ORANGE if n == "llm_hybrid" else (BLUE if n == "transformer_rl" else GRAY) for n in names]
hbar(axes[0], [NAMES[n] for n in names], [sub[n][f"F1@{K}"] for n in names], col)
axes[0].set_title(f"F1@{K}", color=INK, loc="left")
hbar(axes[1], [NAMES[n] for n in names], [sub[n]["CatAcc"] for n in names], col)
axes[1].set_title("Video-category accuracy", color=INK, loc="left")
fig.suptitle(f"LLM comparison on a fixed random subset of {M['n_subset']} test videos", color=INK,
             x=0.01, ha="left", y=1.04)
fig.savefig(FIG / "llm_subset.png"); plt.close(fig)

# 3. efficiency vs quality ---------------------------------------------------------------
eff = M["efficiency"]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for n in ["keyword", "tfidf_lr", "transformer_bce", "transformer_rl"]:
    x, y = eff[n]["latency_ms_p50"], full[n][f"F1@{K}"]
    ax.scatter(x, y, s=70, color=BLUE if n == "transformer_rl" else GRAY, zorder=3,
               edgecolor="white", linewidth=2)
    if n == "transformer_bce":
        continue                      # sits on top of the RL point; labelled together below
    label = "Transformer, BCE and +RL" if n == "transformer_rl" else NAMES[n]
    ax.annotate(f"{label} · {eff[n]['size_mb']:.0f} MB", (x, y), xytext=(-10, -16) if n == "transformer_rl"
                else (8, -4), textcoords="offset points", fontsize=8.5, color=INK,
                ha="right" if n == "transformer_rl" else "left")
ax.set_xscale("log")
ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5], ["0.1", "0.2", "0.5", "1", "2", "5"])
ax.minorticks_off()
ax.set_xlim(0.1, 4)
ax.set_xlabel("Latency per caption, ms (CPU, 1 thread, log scale)")
ax.set_ylabel(f"F1@{K}")
ax.grid(color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
ax.set_title("Quality vs. latency (higher-left is better)", color=INK, loc="left")
fig.savefig(FIG / "efficiency.png"); plt.close(fig)

# 4. RL training curve -------------------------------------------------------------------
h = load_json(rpath("results/train_log_rl.json"))["history"]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(h["step"], h["val_setf1"], color=BLUE, lw=2, marker="o", ms=4)
ax.annotate(f"{h['val_setf1'][0]:.3f}", (h["step"][0], h["val_setf1"][0]), xytext=(6, -12),
            textcoords="offset points", color=INK, fontsize=9)
ax.annotate(f"{h['val_setf1'][-1]:.3f}", (h["step"][-1], h["val_setf1"][-1]), xytext=(-10, 8),
            textcoords="offset points", color=INK, fontsize=9)
ax.set_xlabel("RL update step"); ax.set_ylabel("Validation Set-F1 (greedy, p ≥ 0.5)")
ax.grid(color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
ax.set_title("RL fine-tuning on a set-F1 reward (REINFORCE + RLOO)", color=INK, loc="left")
fig.savefig(FIG / "rl_curve.png"); plt.close(fig)

# 5. per-category F1 ---------------------------------------------------------------------
pc = M["per_category_f1"]
cats = sorted(pc["transformer_rl"], key=lambda c: pc["transformer_rl"][c])
y = np.arange(len(cats))
fig, ax = plt.subplots(figsize=(7, 6.4))
w = 0.27
for i, (n, c) in enumerate([("keyword", GRAY), ("tfidf_lr", AQUA), ("transformer_rl", BLUE)]):
    ax.barh(y + (i - 1) * w, [pc[n][k] for k in cats], height=w - 0.03, color=c, label=NAMES[n])
ax.set_yticks(y, ["#" + c for c in cats])
ax.set_xlabel(f"F1@{K}")
ax.grid(axis="x", color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=False)
ax.set_title("F1 by kind of video (MSR-VTT category)", color=INK, loc="left")
fig.savefig(FIG / "per_category.png"); plt.close(fig)
print("figures ->", FIG)
