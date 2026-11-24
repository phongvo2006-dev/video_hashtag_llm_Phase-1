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
W = f"wnDCG@{K}"
BLUE, ORANGE, AQUA, GRAY, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#cfcec8"
INK, INK2 = "#0b0b0b", "#52514e"
NAMES = {"keyword": "Keyword rules", "tfidf_lr": "TF-IDF + LogReg", "transformer_bce": "Transformer (scratch)",
         "transformer_rl": "Transformer (scratch) + RL", "roberta_bce": "RoBERTa fine-tuned",
         "roberta_rl": "RoBERTa fine-tuned + RL", "keras_bce": "Keras Transformer (TF)",
         "keras_rl": "Keras Transformer (TF) + RL", "llm_zero_shot": "LLM zero-shot",
         "llm_hybrid": "Transformer+RL → LLM"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def color(n):
    return {"roberta_rl": BLUE, "llm_hybrid": ORANGE}.get(n, GRAY)


def hbar(ax, names, values, fmt="{:.3f}"):
    y = np.arange(len(names))[::-1]
    ax.barh(y, values, color=[color(n) for n in names], height=0.6)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.01, yi, fmt.format(v), va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [NAMES[n] for n in names])
    ax.grid(axis="x", color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(0, max(values) * 1.16)


# 1. full test -----------------------------------------------------------------------------
full = M["test_full"]; names = list(full)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
hbar(axes[0], names, [full[n][W] for n in names]); axes[0].set_title(f"{W} (primary)", color=INK, loc="left")
hbar(axes[1], names, [full[n]["SetF1@0.5"] for n in names]); axes[1].set_title("Set-F1", color=INK, loc="left")
hbar(axes[2], names, [full[n]["CatTop1"] for n in names], "{:.1%}")
axes[2].set_title("Category top-1 accuracy", color=INK, loc="left")
fig.suptitle(f"Test set: {M['n_test']:,} MSR-VTT clips, one caption each", color=INK, x=0.01, ha="left", y=1.04)
fig.savefig(FIG / "quality_full_test.png"); plt.close(fig)

# 2. LLM subset ----------------------------------------------------------------------------
sub = M["test_llm_subset"]; names = list(sub)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
hbar(axes[0], names, [sub[n][W] for n in names]); axes[0].set_title(f"{W} (primary)", color=INK, loc="left")
hbar(axes[1], names, [sub[n][f"F1@{K}"] for n in names]); axes[1].set_title(f"F1@{K}", color=INK, loc="left")
fig.suptitle(f"LLM comparison on a fixed random subset of {M['n_subset']} test clips", color=INK,
             x=0.01, ha="left", y=1.04)
fig.savefig(FIG / "llm_subset.png"); plt.close(fig)

# 3. quality vs latency ------------------------------------------------------------------
eff = M["efficiency"]
fig, ax = plt.subplots(figsize=(7, 4))
for n in ["keyword", "tfidf_lr", "transformer_rl", "keras_rl", "roberta_rl"]:
    if n not in eff:
        continue
    x, y = eff[n]["latency_ms_p50"], full[n][W]
    ax.scatter(x, y, s=70, color={"roberta_rl": BLUE, "keras_rl": ORANGE}.get(n, GRAY), zorder=3,
               edgecolor="white", linewidth=2)
    off, ha, va = {"transformer_rl": ((-8, 8), "right", "bottom"), "keras_rl": ((8, -6), "left", "top"),
                   "roberta_rl": ((-10, 8), "right", "bottom")}.get(n, ((8, -4), "left", "top"))
    ax.annotate(f"{NAMES[n]}\n{eff[n]['size_mb']:.0f} MB · {eff[n]['params'] / 1e6:.1f}M params"
                if eff[n]["params"] else NAMES[n], (x, y), xytext=off, textcoords="offset points",
                fontsize=8.5, color=INK, ha=ha, va=va)
ax.set_xscale("log")
ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 100], ["0.1", "0.3", "1", "3", "10", "30", "100"]); ax.minorticks_off()
ax.set_xlabel("Latency per caption, ms (CPU, 1 thread, log scale)"); ax.set_ylabel(W); ax.set_ylim(0.28, 0.6)
ax.grid(color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
ax.set_title("Quality vs. latency", color=INK, loc="left")
fig.savefig(FIG / "efficiency.png"); plt.close(fig)

# 4. RL curves -----------------------------------------------------------------------------
panels = [("PyTorch Transformer (scratch)", load_json(rpath("results/train_log_rl.json"))["history"]["step"],
           load_json(rpath("results/train_log_rl.json"))["history"]["val_setf1"], GRAY)]
if rpath("results/train_log_keras.json").exists():
    hk = load_json(rpath("results/train_log_keras.json"))["rl_history"]
    panels.append(("Keras Transformer (TensorFlow)", hk["step"], hk["val_setf1"], ORANGE))
if rpath("results/train_log_roberta.json").exists():
    r = load_json(rpath("results/train_log_roberta.json"))["rl"]
    panels.append(("RoBERTa (PyTorch)", [e["step"] for e in r], [e["SetF1"] for e in r], BLUE))
fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.3))
for ax, (title, xs, ys, c) in zip(np.atleast_1d(axes), panels):
    ax.plot(xs, ys, color=c, lw=2, marker="o", ms=3)
    ax.set_title(f"{title}", color=INK, loc="left", fontsize=10)
    ax.set_xlabel("RL update step"); ax.grid(color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
np.atleast_1d(axes)[0].set_ylabel("Validation Set-F1 (p ≥ 0.5)")
fig.suptitle("Validation Set-F1 during RL fine-tuning", color=INK, x=0.01, ha="left", y=1.04)
fig.savefig(FIG / "rl_curve.png"); plt.close(fig)

# 5. per category -------------------------------------------------------------------------
pc = M["per_category_wndcg"]
series = [(n, c) for n, c in [("tfidf_lr", LIGHT), ("transformer_rl", AQUA), ("roberta_rl", BLUE)] if n in pc]
best = series[-1][0]
cats = sorted(pc[best], key=lambda c: pc[best][c])
y = np.arange(len(cats)); w = 0.8 / len(series)
fig, ax = plt.subplots(figsize=(7, 6.6))
for i, (n, c) in enumerate(series):
    ax.barh(y + (i - (len(series) - 1) / 2) * w, [pc[n][k] for k in cats], height=w - 0.03, color=c, label=NAMES[n])
ax.set_yticks(y, ["#" + c for c in cats]); ax.set_xlabel(W)
ax.grid(axis="x", color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
h_, l_ = ax.get_legend_handles_labels(); ax.legend(h_[::-1], l_[::-1], loc="lower right", frameon=False)
ax.set_title("Quality by kind of video (MSR-VTT category)", color=INK, loc="left")
fig.savefig(FIG / "per_category.png"); plt.close(fig)
print("figures ->", FIG)
