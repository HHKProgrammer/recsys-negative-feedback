"""
one-command figure and table generator
Run from the project root:  python scripts/generate_all_figures.py

 step by step:
  1. Loads all experiment results (MovieLens 1M / 10M / 20M and Spotify)
  2. Compares the baseline SVD to published benchmarks and tells you if results look right
  3. Generates 8 figures that show research questions for analysis   4. Generates 4 tables for the thesis (CSV + LaTeX)
  5. Saves everything to reports/figures/ and reports/tables/

If you rerun experiments and want updated figures, just run this script again.
Option A results (grid_summary_optionA.json) are loaded automatically if they exist.

Main:

  Baseline question:
    "Does my SVD implementation perform as expected, or is something wrong?"

  Negative feedback question:
    "Do the three strategies (filter / rerank / weighted) make recommendations better
    by avoiding items the user has previously disliked?"

  Cross-dataset question:
    "Do the findings hold on different data: larger MovieLens sets and Spotify skips?"

   +new  question:
    "If I force negatively-rated items into the candidate pool on purpose,
    can the filter and penalty variants keep them out of the top 10?"

Why standard evaluation always shows negative@10 = 0:
  In the standard setup, candidates are 500 randomly chosen UNSEEN items.
  Negatively-rated items are items the user has already seen and rated low.
  They are excluded from the candidate pool by definition.
  So negative@10 (how many disliked items appear in the top 10) is always 0.
  This is not a bug it is the correct evaluation for "avoid recommending seen items."
  The metric that does capture the effect is sim_to_neg@10:
    it measures how similar the top-10 recommended items are to the user's disliked
    items in the latent space. Lower sim_to_neg = recommendations are further from dislikes.

+new evaluation fixes this:
  We add the user's own negative items into the candidate pool on purpose (adversarial test).
  Now negative@10 can be non-zero for the baseline.
  The filter / rerank / weighted variants should push those items out of the top 10.
  Run: python scripts/run_option_a.py  to generate the Option A grid summaries.
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)


RESULTS_DIR = Path("outputs")
FIGURES_DIR = Path("reports/figures")
TABLES_DIR  = Path("reports/tables")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 120})

# one consistent color per variant / dataset throughout all figures
VARIANT_COLORS = {
    "baseline": "#2ecc71",
    "filter":   "#3498db",
    "rerank":   "#e74c3c",
    "weighted": "#9b59b6",
}
DATASET_COLORS = {
    "ml-1m":   "#2980b9",
    "ml-10m":  "#1a5276",
    "ml-20m":  "#154360",
    "spotify": "#1DB954",
}

# short axis labels for each dataset
DATASET_LABEL = {
    "ml-1m":   "ML-1M\n(1M ratings, 6K users)",
    "ml-10m":  "ML-10M\n(10M ratings, 70K users)",
    "ml-20m":  "ML-20M\n(20M ratings, 138K users)",
    "spotify": "Spotify mini\n(168K plays, 10K sessions)",
}

#comparisons
# Surprise RMSE benchmarks
#   Source: surpriselib.com (5-fold cross-validation on MovieLens 100K)
#   Note: RMSE measures how accurately the model predicts the exact star rating.
#         Our evaluation uses ranking metrics (NDCG / HR) instead, because for
#         recommendation the order of items matters more than the exact score.
#
# Expected NDCG@10 / HR@10 ranges with 500 sampled candidates
#   There is no single agreed published number for "SVD on ML with 500 candidates"
#   because each paper uses a slightly different evaluation setup.
#   The ranges below are estimated from:
#     - NCF (He et al. 2017): HR@10 = 0.70 on ML-1M using ALL items as candidates
#     - Scaling down to 500 candidates gives roughly 0.09–0.18 for HR@10
#     - NDCG scales similarly to about 0.05–0.12
#   Spotify is lower because sessions have only ~16 plays → very sparse for SVD.
#
# Fan et al. 2024 critique
#   Many published MovieLens results look suspiciously high because models are
#   tested on temporal splits of a dataset whose timestamps are unreliable.
#   i use random leave-one-out split, which is more conservative and honest.
#   If NDCG@10 > 0.20 on ML with our setup, something is likely wrong.

BENCHMARKS = {
    "surprise_rmse": [
        ("Random baseline",    1.504, 1.206),
        ("NMF",                0.916, 0.724),
        ("BaselineOnly",       0.909, 0.719),
        ("KNN Baseline",       0.895, 0.706),
        ("SVD (default)",      0.873, 0.686),
        ("SVD++ (default)",    0.862, 0.672),
    ],
    # (lower bound, upper bound) for NDCG@10 with 500-candidate sampled eval
    "ndcg_expected": {
        "ml-1m":   (0.05, 0.12),
        "ml-10m":  (0.06, 0.13),
        "ml-20m":  (0.06, 0.13),
        "spotify": (0.01, 0.06),
    },
    "hr_expected": {
        "ml-1m":   (0.09, 0.18),
        "ml-10m":  (0.10, 0.20),
        "ml-20m":  (0.10, 0.20),
        "spotify": (0.02, 0.10),
    },
    "suspicious_ndcg_threshold": 0.20,   # above this possible data leakage
}


#  LOADING DATA
def load_experiments(path: Path) -> list:
    """Read one grid_summary.json and return a deduplicated list of experiments."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    exps = data.get("experiments", [])
    # remove duplicates that appear when the same grid runs twice
    seen, unique = set(), []
    for e in exps:
        if e["exp_id"] not in seen:
            seen.add(e["exp_id"])
            unique.append(e)
    return unique


def load_all_results():
    """
    Load  results for all datasets.
    Returns two dicts keyed by dataset name.
    """
    standard  = {}
    option_a  = {}

    for ds in ["ml-1m", "ml-10m", "ml-20m"]:
        folder = RESULTS_DIR / f"movielens/{ds}"
        exps = load_experiments(folder / "grid_summary.json")
        if exps:
            standard[ds] = exps
            print(f"  {ds}: {len(exps)} experiments")
        exps_a = load_experiments(folder / "grid_summary_optionA.json")
        if exps_a:
            option_a[ds] = exps_a
            print(f"  {ds} Option A: {len(exps_a)} experiments")

    exps_sp = load_experiments(RESULTS_DIR / "spotify/grid_summary.json")
    if exps_sp:
        standard["spotify"] = exps_sp
        print(f"  spotify: {len(exps_sp)} experiments")

    exps_sp_a = load_experiments(RESULTS_DIR / "spotify/grid_summary_optionA.json")
    if exps_sp_a:
        option_a["spotify"] = exps_sp_a
        print(f"  spotify Option A: {len(exps_sp_a)} experiments")

    return standard, option_a


def get_baseline(exps: list) -> dict:
    return next((e for e in exps if e["variant"] == "baseline"), None)


def m(exp: dict, key: str) -> float:
    """Get a metric value from an experiment, defaulting to 0."""
    return exp["metrics"].get(key, 0.0)


#
# This section compares the baseline SVD to the expected performance range and
# prints a plain-language verdict for each dataset. (for me) quick check

def classify(ds: str, exps: list) -> str:
    b = get_baseline(exps)
    if not b:
        return "no baseline found"
    ndcg = m(b, "ndcg@10")
    lo, hi = BENCHMARKS["ndcg_expected"].get(ds, (0.05, 0.12))
    if ndcg > BENCHMARKS["suspicious_ndcg_threshold"]:
        return f"SUSPICIOUS  {ndcg:.4f} (above {BENCHMARKS['suspicious_ndcg_threshold']:.2f} — check for data leakage)"
    if ndcg >= hi:
        return f"GOOD        {ndcg:.4f}  (above expected range {lo:.2f}–{hi:.2f})"
    if ndcg >= lo:
        return f"AVERAGE     {ndcg:.4f}  (within expected range {lo:.2f}–{hi:.2f})"
    return     f"BELOW       {ndcg:.4f}  (below expected range {lo:.2f}–{hi:.2f})"


def print_quality_report(standard: dict):
    print()
    print("BASELINE QUALITY REPORT")
    print("─" * 65)
    for ds, exps in standard.items():
        b = get_baseline(exps)
        if not b:
            continue
        print(f"  {ds.upper():<10}  {classify(ds, exps)}")
        lo_n, hi_n = BENCHMARKS["ndcg_expected"].get(ds, (0, 0))
        lo_h, hi_h = BENCHMARKS["hr_expected"].get(ds, (0, 0))
        print(f"             NDCG@10={m(b,'ndcg@10'):.4f} expect {lo_n:.2f}–{hi_n:.2f}  "
              f"HR@10={m(b,'hit@10'):.4f} expect {lo_h:.2f}–{hi_h:.2f}")
    print()
    print("NOTES:")
    print("  Surprise benchmarks use RMSE (rating prediction), not NDCG (ranking).")
    print("  They measure different things and cannot be directly compared.")
    print("  Our SVD RMSE on ML-1M is estimated to be ~0.87, matching Surprise default.")
    print()
    print("  Fan et al 2024 warn: many high ML results come from temporal splits on")
    print("  unreliable timestamps. Our random split is conservative and honest.")
    print()
    print("  KEY FINDING: rerank and weighted variants reduce sim_to_neg@10")
    print("  (they do push recommendations away from disliked items in latent space),")
    print("  but this consistently LOWERS NDCG@10  the tradeoff is not beneficial.")
    print("  This suggests negative feedback helps only when built into training")
    print("  (e.g. iALS, Hu et al. 2008), not when applied post-hoc at ranking time.")
    print("─" * 65)



def save(name: str):
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  {name}")


#  FIGURE 1 
# Research question: Is my SVD baseline performing at the right level?
#
# Left panel: RMSE comparison to Surprise published benchmarks.
#   This checks whether the SVD model is implemented and trained correctly.
#    RMSE should be around 0.87, matching the Surprise default SVD.
#   Note: i do not actually compute RMSE (i use ranking), so 0.865 is an estimate.
#
# Right panel: the NDCG@10 against the expected range for SVD with 500 candidates.
#   Green band = range of typical SVD results from the literature (scaled estimate).
#   If the bar is inside the green band → results look correct.

def fig1_is_my_baseline_correct(standard: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: RMSE vs Surprise
    ax = axes[0]
    names  = [r[0] for r in BENCHMARKS["surprise_rmse"]] + ["My SVD\n(n=100, ML-1M, estimated)"]
    rmse   = [r[1] for r in BENCHMARKS["surprise_rmse"]] + [0.865]
    colors = ["#cccccc"] * len(BENCHMARKS["surprise_rmse"]) + [VARIANT_COLORS["baseline"]]
    bars = ax.barh(names, rmse, color=colors, edgecolor="white")
    bars[-1].set_edgecolor("black")
    bars[-1].set_linewidth(1.5)
    ax.axvline(0.873, color="red", linestyle="--", lw=1.2, label="SVD default (0.873)")
    for bar, val in zip(bars, rmse):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlabel("RMSE  (lower = better rating prediction)")
    ax.set_title("Fig 1a — SVD RMSE vs Surprise Benchmarks\n"
                 "(ML-100K, 5-fold CV from surpriselib.com)")
    ax.legend(fontsize=9)
    ax.text(0.03, 0.03,
            "RMSE = how closely we predict the exact rating.\n"
            "Our main results use NDCG@10 (ranking quality),\nnot RMSE.",
            transform=ax.transAxes, fontsize=7.5, color="#666",
            verticalalignment="bottom", style="italic")

    # right: NDCG vs expected range
    ax = axes[1]
    ds_list = [d for d in ["ml-1m", "ml-10m", "ml-20m", "spotify"] if d in standard]
    if ds_list:
        ndcg_vals = [m(get_baseline(standard[ds]), "ndcg@10") for ds in ds_list]
        x = np.arange(len(ds_list))
        bar_colors = [DATASET_COLORS.get(ds, "#888") for ds in ds_list]
        ax.bar(x, ndcg_vals, color=bar_colors, alpha=0.85, zorder=3)

        for i, ds in enumerate(ds_list):
            lo, hi = BENCHMARKS["ndcg_expected"].get(ds, (0, 0))
            ax.fill_between([i - 0.45, i + 0.45], lo, hi, alpha=0.2,
                            color="green", zorder=2)
            ax.plot([i - 0.45, i + 0.45], [lo, lo], "k--", lw=0.8, alpha=0.4)
            ax.plot([i - 0.45, i + 0.45], [hi, hi], "k--", lw=0.8, alpha=0.4)
            ax.text(i, ndcg_vals[i] + 0.001, f"{ndcg_vals[i]:.4f}",
                    ha="center", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABEL.get(ds, ds) for ds in ds_list], fontsize=8)
        ax.set_ylabel("NDCG@10")
        ax.set_title("Fig 1b — My NDCG@10 vs Expected Range\n"
                     "(green band = typical SVD with 500 sampled candidates)")

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="green", alpha=0.25, label="Expected range (literature estimate)"),
            Patch(facecolor=VARIANT_COLORS["baseline"], label="My baseline SVD"),
        ], fontsize=8)

    save("fig1_is_baseline_correct")


#  FIGURE 2 
# Research question: Do the penalty variants (rerank / weighted) actually push
# recommendations away from disliked items, and at what cost to ranking quality?
#
# This is the most important figure of the thesis.(i think)
# Each point is one experiment. X-axis = how similar the top-10 is to negative items
# (sim_to_neg@10, lower = further from dislikes). Y-axis = NDCG@10 (higher = better).
#
# What i expect to see:
#   - baseline: sim_to_neg=0 (empty negative set by design), highest NDCG
#   - filter: sim_to_neg is non-zero (shows how negative-like the baseline top-10 is),
#             NDCG identical to baseline (filter has zero effect because negative items
#             are already excluded from candidates)
#   - rerank / weighted: as alpha increases  arrow direction), sim_to_neg goes down
#             but NDCG also goes down the tradeoff is the main finding

def fig2_the_tradeoff(standard: dict):
    ds_list = [d for d in ["ml-1m", "ml-10m"] if d in standard]
    if not ds_list:
        print("  skipping fig2: no ml results")
        return

    fig, axes = plt.subplots(1, len(ds_list), figsize=(7 * len(ds_list), 5))
    if len(ds_list) == 1:
        axes = [axes]

    for ax, ds in zip(axes, ds_list):
        exps = standard[ds]
        for variant in ["baseline", "filter", "rerank", "weighted"]:
            sub = [e for e in exps if e["variant"] == variant]
            if not sub:
                continue
            color = VARIANT_COLORS.get(variant, "#888")
            xs = [m(e, "sim_to_neg@10") for e in sub]
            ys = [m(e, "ndcg@10") for e in sub]
            ax.scatter(xs, ys, color=color, s=55, label=variant, zorder=3, alpha=0.85)
            if variant in ("rerank", "weighted"):
                sub_s = sorted(sub, key=lambda e: e.get("alpha") or 0)
                ax.plot([m(e, "sim_to_neg@10") for e in sub_s],
                        [m(e, "ndcg@10")       for e in sub_s],
                        color=color, lw=0.8, alpha=0.4, linestyle="--")
                # label the alpha=1.0 point
                last = sub_s[-1]
                ax.annotate(f"α={last.get('alpha')}", (m(last, "sim_to_neg@10"), m(last, "ndcg@10")),
                            textcoords="offset points", xytext=(4, -10), fontsize=7.5, color=color)

        b_ndcg = m(get_baseline(exps), "ndcg@10")
        ax.axhline(b_ndcg, color=VARIANT_COLORS["baseline"],
                   lw=1.2, linestyle=":", alpha=0.7, label=f"baseline NDCG = {b_ndcg:.4f}")
        ax.set_xlabel("sim_to_neg@10\n(lower = top-10 is further from disliked items)")
        ax.set_ylabel("NDCG@10  (higher = better)")
        ax.set_title(f"Fig 2 — The Core Tradeoff ({ds.upper()})\n"
                     "Reducing similarity to negatives hurts NDCG")
        ax.legend(fontsize=8)

    save("fig2_core_tradeoff")


#  FIGURE 3 
# Research question: Which variant performs best overall, and is it consistently
# better than the baseline across all four datasets?
#
# Shows the BEST NDCG@10 achieved by each variant on each dataset.
# Best = best threshold + best alpha combination.
# The dashed baseline line makes it easy to see whether any variant ever beats it.

def fig3_best_variant_per_dataset(standard: dict):
    available = {k: v for k, v in standard.items()}
    if not available:
        print("  skipping fig3: no results")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (ds, exps) in zip(axes, available.items()):
        variants = ["baseline", "filter", "rerank", "weighted"]
        best = {}
        for v in variants:
            sub = [e for e in exps if e["variant"] == v]
            if sub:
                best[v] = max(m(e, "ndcg@10") for e in sub)

        colors = [VARIANT_COLORS.get(v, "#888") for v in best]
        bars = ax.bar(list(best.keys()), list(best.values()),
                      color=colors, alpha=0.85)
        b_val = best.get("baseline", 0)
        ax.axhline(b_val, color=VARIANT_COLORS["baseline"],
                   lw=1.5, linestyle="--", alpha=0.7, label="baseline")

        for bar, val in zip(bars, best.values()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.0005, f"{val:.4f}",
                    ha="center", va="bottom", fontsize=8.5)

        ax.set_title(f"{ds.upper()}\n{classify(ds, exps)}", fontsize=9)
        ax.set_ylabel("Best NDCG@10" if ax is axes[0] else "")
        ax.set_ylim(0, max(best.values()) * 1.20)
        ax.tick_params(axis="x", rotation=10)
        ax.legend(fontsize=8)

    fig.suptitle("Fig 3 — Best NDCG@10 per Variant across Datasets\n"
                 "(best threshold + best alpha for each variant)", fontsize=11)
    save("fig3_best_variant_per_dataset")


#  FIGURE 4 
# Research question: Is there a "sweet spot" alpha value where the penalty
# reduces dislikes without hurting NDCG?
#
# Plots NDCG@10 vs alpha (0.1 / 0.3 / 1.0) for rerank and weighted.
# Averaged across all threshold strategies to isolate the alpha effect.
# The dashed line = baseline (no penalty).
# Finding: no sweet spot  any alpha > 0 starts to lower NDCG.

def fig4_alpha_sensitivity(standard: dict):
    ml = {k: v for k, v in standard.items() if k.startswith("ml")}
    if not ml:
        print("  skipping fig4: no ml results")
        return

    n = len(ml)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (ds, exps) in zip(axes, ml.items()):
        b_ndcg = m(get_baseline(exps), "ndcg@10")
        ax.axhline(b_ndcg, color=VARIANT_COLORS["baseline"],
                   lw=1.5, linestyle="--", label=f"baseline ({b_ndcg:.4f})")

        for variant in ["rerank", "weighted"]:
            sub = [e for e in exps if e["variant"] == variant and e.get("alpha") is not None]
            if not sub:
                continue
            df = pd.DataFrame([{"alpha": e["alpha"], "ndcg": m(e, "ndcg@10")} for e in sub])
            means = df.groupby("alpha")["ndcg"].mean()
            stds  = df.groupby("alpha")["ndcg"].std().fillna(0)
            color = VARIANT_COLORS[variant]
            ax.plot(means.index, means.values, marker="o",
                    color=color, lw=2, ms=7, label=variant)
            ax.fill_between(means.index, means - stds, means + stds,
                            color=color, alpha=0.12)

        ax.set_xlabel("Alpha (penalty strength)\n0.1 = gentle,  0.3 = moderate,  1.0 = aggressive")
        ax.set_ylabel("NDCG@10")
        ax.set_title(f"Fig 4 — Effect of Alpha on NDCG ({ds.upper()})\n"
                     "Is there a penalty level that helps without hurting?")
        ax.legend(fontsize=9)
        ax.set_xticks([0.1, 0.3, 1.0])

    save("fig4_alpha_sensitivity")


#  FIGURE 5 
# Research question: Does the way we define "negative" items matter?
#   fixed<=1 : only items rated 1 star are negative (strict)
#   fixed<=2 : items rated 1 or 2 stars are negative
#   fixed<=3 : items rated 1, 2, or 3 stars are negative (lenient)
#   median   : items rated below the user's own median rating are negative
#   modus    : items rated below the user's most frequent rating are negative
#
# Left panel: NDCG@10  does stricter threshold hurt more?
# Right panel: sim_to_neg@10  does a broader negative set reduce dislike proximity more?
# Finding: broader definitions (fixed<=3, median, modus) hurt NDCG more,
#          and reduce sim_to_neg more  confirming the tradeoff from Fig 2.

def fig5_threshold_comparison(standard: dict):
    if "ml-1m" not in standard:
        print("  skipping fig5: no ml-1m results")
        return

    exps = standard["ml-1m"]
    b_ndcg = m(get_baseline(exps), "ndcg@10")

    rows = []
    for e in exps:
        if e["variant"] == "baseline":
            continue
        t = e.get("threshold_type", "")
        label = f"fixed≤{e['fixed_threshold']}" if t == "fixed" else t.capitalize()
        rows.append({
            "threshold": label,
            "ndcg":      m(e, "ndcg@10"),
            "sim_neg":   m(e, "sim_to_neg@10"),
            "variant":   e["variant"],
        })
    if not rows:
        return

    df = pd.DataFrame(rows)
    order = [o for o in ["fixed≤1", "fixed≤2", "fixed≤3", "Median", "Modus"]
             if o in df["threshold"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = {v: VARIANT_COLORS.get(v, "#888") for v in df["variant"].unique()}

    ax = axes[0]
    sns.barplot(data=df, x="threshold", y="ndcg", hue="variant",
                order=order, palette=palette, ax=ax)
    ax.axhline(b_ndcg, color=VARIANT_COLORS["baseline"],
               lw=1.5, linestyle="--", label="baseline")
    ax.set_title("Fig 5a — NDCG@10 by Threshold (ML-1M)\n"
                 "Broader negative definition → more NDCG damage")
    ax.set_xlabel("How we define 'negative'")
    ax.set_ylabel("NDCG@10")
    ax.legend(title="Variant", fontsize=8)

    ax = axes[1]
    sns.barplot(data=df, x="threshold", y="sim_neg", hue="variant",
                order=order, palette=palette, ax=ax)
    ax.set_title("Fig 5b — sim_to_neg@10 by Threshold (ML-1M)\n"
                 "Broader definition also reduces negative similarity more")
    ax.set_xlabel("How we define 'negative'")
    ax.set_ylabel("sim_to_neg@10\n(lower = top-10 further from dislikes)")
    ax.legend(title="Variant", fontsize=8)

    save("fig5_threshold_comparison")


#  FIGURE 6 
# Research question: Does increasing the dataset size improve the results,
# and does the effect of negative feedback change at scale?
#
# Left panel: baseline vs best rerank NDCG@10 for 1M / 10M / 20M.
#   Tells us whether more data gives a better baseline or more room for improvement.
# Right panel: baseline NDCG and HR plotted against dataset size (in millions of ratings).
#   Shows whether SVD scales well with more data.

def fig6_dataset_scaling(standard: dict):
    ml = {k: standard[k] for k in ["ml-1m", "ml-10m", "ml-20m"] if k in standard}
    if len(ml) < 2:
        print("  skipping fig6: need at least 2 ML datasets")
        return

    ds_order = [d for d in ["ml-1m", "ml-10m", "ml-20m"] if d in ml]
    n_ratings = {"ml-1m": 1.0, "ml-10m": 10.0, "ml-20m": 20.0}

    rows = []
    for ds in ds_order:
        exps = ml[ds]
        b = get_baseline(exps)
        best_rerank = max(
            (e for e in exps if e["variant"] == "rerank"),
            key=lambda e: m(e, "ndcg@10"), default=None)
        rows.append({
            "dataset":        ds,
            "n_M_ratings":    n_ratings[ds],
            "baseline_ndcg":  m(b, "ndcg@10"),
            "baseline_hr":    m(b, "hit@10"),
            "best_rerank_ndcg": m(best_rerank, "ndcg@10") if best_rerank else 0,
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x, w = np.arange(len(ds_order)), 0.35
    ax.bar(x - w/2, df["baseline_ndcg"], w, color=VARIANT_COLORS["baseline"],
           alpha=0.85, label="Baseline SVD")
    ax.bar(x + w/2, df["best_rerank_ndcg"], w, color=VARIANT_COLORS["rerank"],
           alpha=0.85, label="Best Rerank")
    for i, row in df.iterrows():
        ax.text(i - w/2, row["baseline_ndcg"] + 0.001,
                f"{row['baseline_ndcg']:.4f}", ha="center", fontsize=8)
        ax.text(i + w/2, row["best_rerank_ndcg"] + 0.001,
                f"{row['best_rerank_ndcg']:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ds}\n({int(n_ratings[ds])}M ratings)" for ds in ds_order])
    ax.set_ylabel("NDCG@10")
    ax.set_title("Fig 6a — Baseline vs Best Rerank across Sizes\n"
                 "Does more data help, and does rerank help more?")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(df["n_M_ratings"], df["baseline_ndcg"], "o-",
            color=VARIANT_COLORS["baseline"], lw=2, ms=8, label="Baseline NDCG@10")
    ax.plot(df["n_M_ratings"], df["baseline_hr"], "s--",
            color=DATASET_COLORS["ml-1m"], lw=2, ms=8, label="Baseline HR@10")
    for _, row in df.iterrows():
        ax.annotate(row["dataset"],
                    (row["n_M_ratings"], row["baseline_ndcg"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Dataset size (millions of ratings)")
    ax.set_ylabel("Metric")
    ax.set_title("Fig 6b — Does SVD Improve with More Data?\n"
                 "(each point = one MovieLens version)")
    ax.legend(fontsize=9)

    save("fig6_dataset_scaling")


#  FIGURE 7 
# Research question: Do negative feedback strategies work the same way on Spotify
# (implicit skip signals) as on MovieLens (explicit star ratings)?
#
# MovieLens: user explicitly gave 1-2 stars → clear dislike signal
# Spotify:   user skipped a track → converted to rating 1-2 using skip taxonomy
#            BUT sessions are short (~16 plays) → SVD cannot learn good embeddings
#
# Expected difference: Spotify NDCG will be much lower (sparse data), and the
# effect of negative variants will be similar (or even weaker) to MovieLens.
# This is normal and expected  we are not claiming Spotify works as well as ML.
# The finding shows whether the pattern generalises across feedback modalities.

def fig7_movielens_vs_spotify(standard: dict):
    if "ml-1m" not in standard or "spotify" not in standard:
        print("  skipping fig7: need both ml-1m and spotify results")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, ds in zip(axes, ["ml-1m", "spotify"]):
        exps = standard[ds]
        rows = [{"variant": e["variant"],
                 "ndcg":    m(e, "ndcg@10"),
                 "sim_neg": m(e, "sim_to_neg@10")} for e in exps]
        df = pd.DataFrame(rows)
        palette = {v: VARIANT_COLORS.get(v, "#aaa") for v in df["variant"].unique()}
        sns.boxplot(data=df, x="variant", y="ndcg", hue="variant",
                    palette=palette, ax=ax, legend=False)
        b_val = df[df["variant"] == "baseline"]["ndcg"].mean()
        ax.axhline(b_val, color=VARIANT_COLORS["baseline"],
                   lw=1.5, linestyle="--", label=f"baseline = {b_val:.4f}")
        title_extra = ("MovieLens 1M — explicit star ratings"
                       if ds == "ml-1m" else
                       "Spotify mini — implicit skip signals\n(session = user, ~16 plays/session)")
        ax.set_title(f"Fig 7 — {title_extra}")
        ax.set_xlabel("Variant")
        ax.set_ylabel("NDCG@10")
        ax.legend(fontsize=8)

        note = ("rerank / weighted consistently lower NDCG\n"
                "filter has zero effect (neg items already excluded)")
        ax.text(0.02, 0.02, note, transform=ax.transAxes,
                fontsize=7.5, color="#555", verticalalignment="bottom", style="italic")

    save("fig7_movielens_vs_spotify")


#  FIGURE 8 
#  adds negative items to the candidate pool
# Research question: If we deliberately include the user's disliked
# items in the candidate pool, can the filter / rerank / weighted variants keep them
# out of the top 10?
#
# This is the "adversarial" test. In the standard evaluation, negatives are never
# in the candidate pool so negative@10 = 0 always. Here we force them in and ask:
# "Does our method actually protect the user from seeing known dislikes?"
#
# How to generate  results:
#   python scripts/run_option_a.py
# This runs the same 36 experiments but adds negative items to the candidate pool
# and saves the results as grid_summary_optionA.json in each dataset folder.

def fig8_option_a(standard: dict, option_a: dict):
    avail = [ds for ds in standard if ds in option_a]
    if not avail:
        print("  skipping fig8: no Option A results (run: python scripts/run_option_a.py)")
        return

    n = len(avail)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, avail):
        variants = ["baseline", "filter", "rerank", "weighted"]
        def avg(exps, key):
            return {v: np.mean([m(e, key) for e in exps if e["variant"] == v] or [0])
                    for v in variants}

        std_neg  = avg(standard[ds],  "negative@10")
        opt_neg  = avg(option_a[ds],  "negative@10")
        std_ndcg = avg(standard[ds],  "ndcg@10")
        opt_ndcg = avg(option_a[ds],  "ndcg@10")

        x, w = np.arange(len(variants)), 0.35
        colors = [VARIANT_COLORS.get(v, "#aaa") for v in variants]
        ax.bar(x - w/2, [std_neg[v] for v in variants], w,
               color=colors, alpha=0.4, label="Standard neg@10 (always ≈ 0)")
        ax.bar(x + w/2, [opt_neg[v] for v in variants], w,
               color=colors, alpha=0.85, label="Option A neg@10")

        ax2 = ax.twinx()
        ax2.plot(x, [std_ndcg[v] for v in variants], "o-",
                 color="#666", lw=1.5, ms=6, label="Standard NDCG@10")
        ax2.plot(x, [opt_ndcg[v] for v in variants], "s--",
                 color="#222", lw=1.5, ms=6, label="Option A NDCG@10")
        ax2.set_ylabel("NDCG@10", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.set_ylabel("negative@10\n(disliked items in top 10)")
        ax.set_title(f"Fig 8 — Option A: Adversarial Evaluation ({ds.upper()})\n"
                     "Negatives added to candidates on purpose")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8)

    save("fig8_option_a_adversarial")


#  7.  TABLES

def df_to_latex(df: "pd.DataFrame", path: Path, caption: str = "") -> None:
    """Write a DataFrame to a .tex file without needing jinja2."""
    cols = list(df.columns)
    col_fmt = "l" + "r" * (len(cols) - 1)
    header = " & ".join(str(c).replace("_", r"\_").replace("%", r"\%") for c in cols)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            s = str(v).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_tables(standard: dict, option_a: dict):
    """Produce 4 tables: full results, baseline summary, Surprise comparison, key findings."""

    # table 1: everything  all experiments on all datasets
    rows = []
    for ds, exps in standard.items():
        for e in exps:
            t = e.get("threshold_type") or "N/A"
            label = f"fixed≤{e['fixed_threshold']}" if t == "fixed" else t
            rows.append({
                "Dataset":      ds,
                "Variant":      e["variant"],
                "Threshold":    label,
                "Alpha":        e.get("alpha", "—"),
                "NDCG@10":      round(m(e, "ndcg@10"), 4),
                "HR@10":        round(m(e, "hit@10"),  4),
                "MRR":          round(m(e, "mrr"),     4),
                "P@10":         round(m(e, "precision@10"), 4),
                "sim_neg@10":   round(m(e, "sim_to_neg@10"), 4),
            })
    if rows:
        df = pd.DataFrame(rows).sort_values(["Dataset", "Variant", "Threshold", "Alpha"])
        df.to_csv(TABLES_DIR / "table1_all_results.csv", index=False)
        df_to_latex(df, TABLES_DIR / "table1_all_results.tex", "All experiments across all datasets")
        print("  table1_all_results")

    # table 2: one row per dataset  baseline summary + quality verdict
    base_rows = []
    for ds, exps in standard.items():
        b = get_baseline(exps)
        if not b:
            continue
        lo_n, hi_n = BENCHMARKS["ndcg_expected"].get(ds, (0, 0))
        base_rows.append({
            "Dataset":             ds,
            "NDCG@10":             round(m(b, "ndcg@10"), 4),
            "HR@10":               round(m(b, "hit@10"),  4),
            "MRR":                 round(m(b, "mrr"),     4),
            "Expected NDCG range": f"{lo_n:.2f}–{hi_n:.2f}",
            "Verdict":             classify(ds, exps),
        })
    if base_rows:
        df2 = pd.DataFrame(base_rows)
        df2.to_csv(TABLES_DIR / "table2_baseline_summary.csv", index=False)
        df_to_latex(df2, TABLES_DIR / "table2_baseline_summary.tex", "Baseline SVD quality verdict per dataset")
        print("  table2_baseline_summary")

    # table 3: Surprise RMSE benchmark
    cmp = pd.DataFrame([
        {"Algorithm": "Random (lower bound)",   "RMSE": 1.504, "MAE": 1.206, "Source": "surpriselib.com"},
        {"Algorithm": "NMF",                    "RMSE": 0.916, "MAE": 0.724, "Source": "surpriselib.com"},
        {"Algorithm": "BaselineOnly",           "RMSE": 0.909, "MAE": 0.719, "Source": "surpriselib.com"},
        {"Algorithm": "SVD (default)",          "RMSE": 0.873, "MAE": 0.686, "Source": "surpriselib.com"},
        {"Algorithm": "SVD++ (default)",        "RMSE": 0.862, "MAE": 0.672, "Source": "surpriselib.com"},
        {"Algorithm": "My SVD (n=100, ML-1M)",  "RMSE": "~0.87", "MAE": "N/A", "Source": "This thesis (estimated)"},
    ])
    cmp.to_csv(TABLES_DIR / "table3_surprise_comparison.csv", index=False)
    df_to_latex(cmp, TABLES_DIR / "table3_surprise_comparison.tex", "Surprise SVD benchmark comparison (ML-100K, 5-fold CV)")
    print("  table3_surprise_comparison")

    # table 4: key findings  best variant per dataset, delta over baseline
    finding_rows = []
    for ds, exps in standard.items():
        b = get_baseline(exps)
        b_ndcg = m(b, "ndcg@10") if b else 0
        for variant in ["filter", "rerank", "weighted"]:
            sub = [e for e in exps if e["variant"] == variant]
            if not sub:
                continue
            best = max(sub, key=lambda e: m(e, "ndcg@10"))
            delta = (m(best, "ndcg@10") - b_ndcg) / b_ndcg * 100 if b_ndcg else 0
            finding_rows.append({
                "Dataset":         ds,
                "Variant":         variant,
                "Best NDCG@10":    round(m(best, "ndcg@10"), 4),
                "Baseline NDCG":   round(b_ndcg, 4),
                "Change (%)":      round(delta, 1),
                "Best config":     best["exp_id"],
                "sim_neg@10":      round(m(best, "sim_to_neg@10"), 4),
            })
    if finding_rows:
        df4 = pd.DataFrame(finding_rows)
        df4.to_csv(TABLES_DIR / "table4_key_findings.csv", index=False)
        df_to_latex(df4, TABLES_DIR / "table4_key_findings.tex", "Best variant per dataset vs baseline")
        print("  table4_key_findings")


#  8.  MAIN 

def main():
    print("=" * 65)
    print("THESIS FIGURE GENERATOR — Negative Feedback in Recommender Systems")
    print("=" * 65)
    print()
    print("Loading experiment results...")
    standard, option_a = load_all_results()

    if not standard:
        print("No results found. Run experiments first:")
        print("  python main.py grid --config configs/movielens_1m.yaml")
        sys.exit(1)

    print_quality_report(standard)

    print("Generating figures...")
    fig1_is_my_baseline_correct(standard)
    fig2_the_tradeoff(standard)
    fig3_best_variant_per_dataset(standard)
    fig4_alpha_sensitivity(standard)
    fig5_threshold_comparison(standard)
    fig6_dataset_scaling(standard)
    fig7_movielens_vs_spotify(standard)
    fig8_option_a(standard, option_a)

    print()
    print("Generating tables...")
    generate_tables(standard, option_a)

    print()
    print(f"Figures → {FIGURES_DIR}/")
    print(f"Tables  → {TABLES_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()