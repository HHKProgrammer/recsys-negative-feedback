"""
Supplementary figure generator for Arm B, Arm D, and Master comparison.
Run from project root:  python scripts/generate_arm_figures.py

Generates:
  fig_arm_b_detection  — Arm B dislike detection HR@10 by threshold + dataset
  fig_arm_d_comparison — Arm D nDCG@10 by config, compared to Arm A + baseline
  fig_master_ndcg      — Master nDCG@10 bar chart: all methods × all datasets
  fig_arm_d_heatmap    — Arm D nDCG@10 heatmap: threshold config × dataset
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

FIGURES_DIR = Path("reports/figures")
TABLES_DIR  = Path("reports/tables")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size":        14,
    "axes.titlesize":   15,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "figure.dpi":       150,
    "font.family":      "sans-serif",
})

DATASET_LABELS = {
    "ml-1m":   "ML-1M",
    "ml-10m":  "ML-10M",
    "ml-20m":  "ML-20M",
    "spotify": "Spotify",
}
DS_ORDER = ["ml-1m", "ml-10m", "ml-20m", "spotify"]
DS_FULL  = ["MovieLens 1M", "MovieLens 10M", "MovieLens 20M", "Spotify MSSD"]

METHOD_COLORS = {
    "Baseline": "#95a5a6",
    "Arm A":    "#2ecc71",
    "Arm C":    "#f39c12",
    "Arm D":    "#e74c3c",
}

# ──────────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────────

# Arm B detection results (from tableC_arm_b_detection.csv)
arm_b_data = pd.read_csv(TABLES_DIR / "tableC_arm_b_detection.csv")

# Arm D results per config (from tableE_arm_d_headline.csv)
arm_d_data = pd.read_csv(TABLES_DIR / "tableE_arm_d_headline.csv")

# Arm D vs Arm A comparison (from tableF_arm_d_vs_arm_a.csv)
arm_d_vs_a = pd.read_csv(TABLES_DIR / "tableF_arm_d_vs_arm_a.csv")
# Normalize dataset names for consistency
arm_d_vs_a["Dataset"] = arm_d_vs_a["Dataset"].str.lower().str.replace(" ", "-").str.replace("movielens-", "ml-").str.replace("spotify-mssd", "spotify")

# Arm C headline data (from tableD_arm_c_headline.csv)
arm_c_data = pd.read_csv(TABLES_DIR / "tableD_arm_c_headline.csv")

# ──────────────────────────────────────────────────────────────────────────────
# FIG 1: Arm B Detection — HR@10 by threshold, grouped by dataset
# ──────────────────────────────────────────────────────────────────────────────

def fig_arm_b_detection():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=False)
    axes = axes.flatten()
    fig.suptitle("Arm B — Negative-item detection performance", fontsize=15, y=1.01)

    ds_map = {"ml-1m": "MovieLens 1M", "ml-10m": "MovieLens 10M",
              "ml-20m": "MovieLens 20M", "spotify": "Spotify MSSD"}
    threshold_labels = {
        "neg_le_1": "$\\leq$1", "neg_le_2": "$\\leq$2", "neg_le_3": "$\\leq$3",
        "neg_median": "median", "neg_modus": "mode",
    }
    colors = ["#2980b9", "#27ae60", "#e74c3c", "#8e44ad", "#f39c12"]

    rand_handle = None
    for ax, ds in zip(axes, DS_ORDER):
        df = arm_b_data[arm_b_data["Dataset"] == ds].copy()
        thresholds = df["Neg threshold"].tolist()
        hr_vals    = df["Detection HR@10"].tolist()
        labels     = [threshold_labels.get(t, t) for t in thresholds]
        bars = ax.bar(range(len(labels)), hr_vals,
                      color=colors[:len(labels)], width=0.6, alpha=0.85)
        line = ax.axhline(0.02, color="gray", linestyle="--", linewidth=0.8)
        if rand_handle is None:
            rand_handle = line
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=13, rotation=20, ha="right")
        ax.set_title(ds_map.get(ds, ds), fontsize=15)
        ax.set_ylabel("Detection HR@10" if ds in ("ml-1m", "ml-20m") else "")
        ax.set_ylim(0, max(hr_vals) * 1.28 + 0.02)
        for bar, val in zip(bars, hr_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=14,
                    fontweight="bold")

    fig.legend([rand_handle], ["Random-ranker HR@10 ≈ 0.020"],
               loc="upper center", bbox_to_anchor=(0.5, 1.00),
               ncol=1, fontsize=13, frameon=False)
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig_arm_b_detection.{fmt}",
                    bbox_inches="tight")
    plt.close(fig)
    print("  fig_arm_b_detection")


# ──────────────────────────────────────────────────────────────────────────────
# FIG 2: Arm D comparison — nDCG@10 by config, per dataset (grouped bars)
# ──────────────────────────────────────────────────────────────────────────────

def fig_arm_d_comparison():
    configs = ["p4_n2", "p4_n1", "p5_n1", "p3_n2"]
    config_labels = {
        "p4_n2": "$\\geq$4/$\\leq$2",
        "p4_n1": "$\\geq$4/$\\leq$1",
        "p5_n1": "$\\geq$5/$\\leq$1",
        "p3_n2": "$\\geq$3/$\\leq$2",
    }
    config_map = {(4,2): "p4_n2", (4,1): "p4_n1", (5,1): "p5_n1", (3,2): "p3_n2"}
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    datasets = DS_ORDER
    n_ds = len(datasets)
    x = np.arange(n_ds)
    width = 0.18

    # Build matrix: rows=configs, cols=datasets
    matrix = {}
    for _, row in arm_d_data.iterrows():
        ds_raw = str(row["Dataset"]).lower().strip()
        pos = int(row["Pos≥"]) if "Pos≥" in row else int(row.iloc[1])
        neg = int(row["Neg≤"]) if "Neg≤" in row else int(row.iloc[2])
        ndcg = float(row["NDCG@10"])
        key = config_map.get((pos, neg))
        if key:
            matrix.setdefault(key, {})[ds_raw] = ndcg

    # Arm A baseline per dataset (from tableF)
    arm_a_vals = {}
    baseline_vals = {}
    for _, row in arm_d_vs_a.iterrows():
        ds = str(row["Dataset"]).lower()
        arm_a_vals[ds]    = float(row["Arm A (nDCG@10)"])
        baseline_vals[ds] = float(row["SVD (nDCG@10)"])

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (cfg, clr) in enumerate(zip(configs, colors)):
        vals = [matrix.get(cfg, {}).get(ds, 0) for ds in datasets]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=config_labels[cfg],
                      color=clr, alpha=0.85)

    # Arm A as dashed line per dataset
    arm_a_line = [arm_a_vals.get(ds, 0) for ds in datasets]
    ax.plot(x, arm_a_line, "k--", linewidth=1.5, marker="D", markersize=5,
            label="Arm A (pos-only)")

    # Baseline as dotted line
    base_line = [baseline_vals.get(ds, 0) for ds in datasets]
    ax.plot(x, base_line, ":", color="gray", linewidth=1.2, marker="s", markersize=4,
            label="Full-rating SVD")

    ax.set_xticks(x)
    ax.set_xticklabels(DS_FULL, fontsize=10)
    ax.set_ylabel("nDCG@10")
    ax.set_title("Arm D — Joint binary SVD across threshold configurations")
    ax.legend(fontsize=9, loc="upper left", ncol=3)
    ax.set_ylim(0, max(max(arm_a_line), 0.35) * 1.15)
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig_arm_d_comparison.{fmt}", bbox_inches="tight")
    plt.close(fig)
    print("  fig_arm_d_comparison")


# ──────────────────────────────────────────────────────────────────────────────
# FIG 3: Master nDCG@10 — all methods × all datasets
# ──────────────────────────────────────────────────────────────────────────────

def fig_master_ndcg():
    # Hard-coded from verified results
    data = {
        "Baseline": [0.0684, 0.0872, 0.0826, 0.0176],
        "Arm A":    [0.0811, 0.2219, 0.2405, 0.3447],
        "Arm C":    [0.0707, 0.0865, 0.0826, 0.3423],
        "Arm D":    [0.1122, 0.2535, 0.2995, 0.3117],
    }
    methods = list(data.keys())
    colors  = [METHOD_COLORS[m] for m in methods]

    x = np.arange(4)
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (method, clr) in enumerate(zip(methods, colors)):
        vals   = data[method]
        offset = (i - 1.5) * width
        bars   = ax.bar(x + offset, vals, width, label=method, color=clr, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(DS_FULL, fontsize=10)
    ax.set_ylabel("nDCG@10")
    ax.set_title("nDCG@10 by Method and Dataset (test set, 501 candidates per user)")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, 0.42)
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig_master_ndcg.{fmt}", bbox_inches="tight")
    plt.close(fig)
    print("  fig_master_ndcg")


# ──────────────────────────────────────────────────────────────────────────────
# FIG 4: Arm D heatmap — nDCG@10 by (threshold config × dataset)
# ──────────────────────────────────────────────────────────────────────────────

def fig_arm_d_heatmap():
    config_order  = ["$\\geq$5/$\\leq$1", "$\\geq$4/$\\leq$1",
                     "$\\geq$4/$\\leq$2", "$\\geq$3/$\\leq$2"]
    config_map    = {(5,1): 0, (4,1): 1, (4,2): 2, (3,2): 3}

    matrix = np.zeros((len(config_order), len(DS_ORDER)))

    for _, row in arm_d_data.iterrows():
        ds_raw = str(row["Dataset"]).lower().strip()
        pos = int(row.iloc[1])
        neg = int(row.iloc[2])
        ndcg = float(row["NDCG@10"])
        ci = config_map.get((pos, neg))
        di = DS_ORDER.index(ds_raw) if ds_raw in DS_ORDER else -1
        if ci is not None and di >= 0:
            matrix[ci, di] = ndcg

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto",
                   vmin=0.05, vmax=0.35)

    ax.set_xticks(range(len(DS_ORDER)))
    ax.set_xticklabels(DS_FULL, fontsize=10)
    ax.set_yticks(range(len(config_order)))
    ax.set_yticklabels(config_order, fontsize=10)
    ax.set_title("Arm D nDCG@10 by Threshold Configuration and Dataset")

    for i in range(len(config_order)):
        for j in range(len(DS_ORDER)):
            val = matrix[i, j]
            color = "white" if val > 0.20 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, label="nDCG@10", fraction=0.03, pad=0.02)
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig_arm_d_heatmap.{fmt}", bbox_inches="tight")
    plt.close(fig)
    print("  fig_arm_d_heatmap")


# ──────────────────────────────────────────────────────────────────────────────
# FIG 5: Arm D vs Arm A per dataset — improvement bar
# ──────────────────────────────────────────────────────────────────────────────

def fig_arm_d_vs_arm_a():
    ds_labels = DS_FULL
    arm_a = [0.0811, 0.2219, 0.2405, 0.3447]
    arm_d = [0.1122, 0.2535, 0.2995, 0.3117]
    base  = [0.0684, 0.0872, 0.0826, 0.0176]

    x = np.arange(4)
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, base,  width, label="Full-rating SVD", color="#95a5a6", alpha=0.85)
    ax.bar(x,         arm_a, width, label="Arm A (pos-only)", color="#2ecc71", alpha=0.85)
    ax.bar(x + width, arm_d, width, label="Arm D (joint binary)", color="#e74c3c", alpha=0.85)

    for xi, (b, a, d) in enumerate(zip(base, arm_a, arm_d)):
        winner = "D" if d > a else "A"
        clr = "#e74c3c" if winner == "D" else "#2ecc71"
        ax.text(xi, max(b, a, d) + 0.008,
                f"Best: Arm {winner}", ha="center", fontsize=8.5, color=clr)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels, fontsize=10)
    ax.set_ylabel("nDCG@10")
    ax.set_title("Full-rating SVD vs.\ Arm A vs.\ Arm D — Best Configuration per Dataset")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 0.42)
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig_arm_d_vs_arm_a.{fmt}", bbox_inches="tight")
    plt.close(fig)
    print("  fig_arm_d_vs_arm_a")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating supplementary arm figures...")
    fig_arm_b_detection()
    fig_arm_d_comparison()
    fig_master_ndcg()
    fig_arm_d_heatmap()
    fig_arm_d_vs_arm_a()
    print(f"Done. Figures written to {FIGURES_DIR}/")
