"""实验结果可视化（浅色主题，支持中文）"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import FIG_DIR, LABELS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["grid.color"] = "#DDDDDD"

C_UNI = "#4C78A8"     # 单向：蓝
C_BI = "#E45756"      # 双向：红
C_GRAY = "#B0B0B0"
PALETTE = ["#4C78A8", "#E45756", "#54A24B", "#F58518", "#B279A2", "#72B7B2"]


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_length_distribution(raw_lengths, max_len):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(raw_lengths, bins=60, color=C_UNI, alpha=0.85, edgecolor="white")
    ax.axvline(max_len, color=C_BI, linestyle="--", linewidth=2,
               label=f"L = {max_len}（截断/补齐阈值）")
    tr = float((raw_lengths > max_len).mean())
    pd = float((raw_lengths < max_len).mean())
    ax.set_title(f"句子长度分布（截断比例 {tr:.1%}，补齐比例 {pd:.1%}）", fontsize=12)
    ax.set_xlabel("句子单词数")
    ax.set_ylabel("句子数量")
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, "01_length_distribution.png")


def plot_curves(histories, title, name):
    """histories: {模型名: history}"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    colors = {}
    for i, k in enumerate(histories):
        colors[k] = C_BI if k.lower().startswith("bi") else C_UNI

    for k, h in histories.items():
        ep = range(1, len(h["val_acc"]) + 1)
        axes[0].plot(ep, h["val_loss"], "-o", markersize=3, color=colors[k], label=k,
                     alpha=0.9, linewidth=1.6)
        axes[1].plot(ep, h["val_acc"], "-o", markersize=3, color=colors[k], label=k,
                     alpha=0.9, linewidth=1.6)

    axes[0].set_title("验证集损失曲线")
    axes[1].set_title("验证集准确率曲线")
    for a in axes:
        a.set_xlabel("Epoch")
        a.grid(alpha=0.3, linestyle="--")
        a.legend(fontsize=8, frameon=False)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[1].set_ylabel("Accuracy")
    fig.suptitle(title, fontsize=12)
    return _save(fig, name)


def plot_group_bars(results, metric, title, name, ylabel=None, ylim=None):
    """results: list of dict(name, value, group)"""
    names = [r["name"] for r in results]
    vals = [r["value"] for r in results]
    groups = [r["group"] for r in results]

    fig, ax = plt.subplots(figsize=(max(6.5, len(names) * 1.15), 4.0))
    x = np.arange(len(names))
    colors = [C_BI if g == "双向" else C_UNI for g in groups]
    bars = ax.bar(x, vals, color=colors, width=0.62, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    if ylim is None and vals:
        vmin, vmax = min(vals), max(vals)
        # 差异较小（<2.5 个百分点）时放大纵轴，让组间差异可见
        if vmax - vmin < 0.025:
            ylim = (max(0.0, vmin - 0.03), min(1.0, vmax + 0.012))
            title += "（纵轴已放大以便观察差异）"
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel or metric)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_UNI), plt.Rectangle((0, 0), 1, 1, color=C_BI)]
    ax.legend(handles, ["单向", "双向"], frameon=False, loc="lower right", fontsize=9)
    return _save(fig, name)


def plot_confusion(cm, title, name, normalize=True):
    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else cm.max())
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_yticklabels(LABELS, fontsize=9)
    ax.set_xlabel("预测标签", fontsize=10)
    ax.set_ylabel("真实标签", fontsize=10)
    ax.set_title(title, fontsize=11)
    thresh = (1 if normalize else cm.max()) * 0.55
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            txt = f"{cm[i, j]:.3f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > thresh else "#333333")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, name)


def plot_embedding_ab(results, name):
    """Embedding vs Word2Vec 双方案对比（分组柱状图）"""
    keys = sorted({r["model"] for r in results})
    schemes = ["nn.Embedding", "Word2Vec"]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(len(keys))
    w = 0.36
    for i, s in enumerate(schemes):
        vals = []
        for k in keys:
            v = [r["value"] for r in results if r["model"] == k and r["scheme"] == s]
            vals.append(v[0] if v else 0.0)
        b = ax.bar(x + (i - 0.5) * w, vals, width=w, label=s,
                   color=["#4C78A8", "#54A24B"][i], edgecolor="white")
        for r0, v0 in zip(b, vals):
            ax.text(r0.get_x() + r0.get_width() / 2, v0 + 0.004, f"{v0:.4f}",
                    ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=9)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("向量化方案对比：nn.Embedding vs Word2Vec 预训练", fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, name)


def plot_delta_bars(deltas, name):
    """单项指标提升幅度（双向 - 单向）"""
    names = [d[0] for d in deltas]
    vals = [d[1] for d in deltas]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colors = [C_BI if v >= 0 else C_GRAY for v in vals]
    bars = ax.barh(names, vals, color=colors, height=0.55, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(v + (0.002 if v >= 0 else -0.002), b.get_y() + b.get_height() / 2,
                f"{v:+.4f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color="#333333")
    ax.axvline(0, color="#888888", linewidth=1)
    ax.set_xlabel("双向 - 单向（Test Accuracy）")
    ax.set_title("双向结构相对单向结构的提升幅度", fontsize=12)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, name)
