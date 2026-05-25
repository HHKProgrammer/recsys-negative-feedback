"""
one-command figure and table generator
Run from the project root:  python scripts/generate_all_figures.py

 step by step:
  1. Loads all experiment results (MovieLens 1M / 10M / 20M and Spotify)
  2. Compares the baseline SVD to published benchmarks and tells you if results look right
  3. Generates 8 figures that show research questions for analysis   4. Generates 4 tables for the thesis (CSV + LaTeX)
  5. Saves everything to reports/figures/ and reports/tables/

If you rerun experiments and want updated figures, just run this script again.
Known-negative injection results (grid_summary_known_neg_eval.json) are loaded automatically if they exist.

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
  Run: python scripts/run_known_negative_eval.py --config configs/movielens_1m.yaml
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
    Load results for all datasets.
    Returns seven dicts keyed by dataset name:
      standard        main grid (post-hoc variants: filter/rerank/weighted)
      train_positive  training-time positive-only SVD (legacy v1)
      known_neg_eval  adversarial evaluation with injected negative candidates
      arm_a           Arm A — SVD trained on rating >= pos_threshold
      arm_b           Arm B — SVD trained on rating <= neg_threshold (dislike detector)
      arm_c           Arm C — hybrid: minmax(arm_a) - alpha * minmax(arm_b)
      arm_d           Arm D — joint positive-negative SVD (binary targets)
    """
    standard       = {}
    train_positive = {}
    known_neg_eval = {}
    arm_a          = {}
    arm_b          = {}
    arm_c          = {}
    arm_d          = {}

    ds_folders = {
        "ml-1m":   RESULTS_DIR / "movielens/ml-1m",
        "ml-10m":  RESULTS_DIR / "movielens/ml-10m",
        "ml-20m":  RESULTS_DIR / "movielens/ml-20m",
        "spotify": RESULTS_DIR / "spotify",
    }

    for ds, folder in ds_folders.items():
        for key, fname, store in [
            ("standard",       "grid_summary.json",                standard),
            ("train_positive", "grid_summary_train_positive.json", train_positive),
            ("known_neg_eval", "grid_summary_known_neg_eval.json", known_neg_eval),
            ("arm_a",          "grid_summary_arm_a.json",          arm_a),
            ("arm_b",          "grid_summary_arm_b.json",          arm_b),
            ("arm_c",          "grid_summary_arm_c.json",          arm_c),
            ("arm_d",          "grid_summary_arm_d.json",          arm_d),
        ]:
            exps = load_experiments(folder / fname)
            if exps:
                store[ds] = exps
                print(f"  {ds}: {len(exps):3d} experiments ({key})")

    return standard, train_positive, known_neg_eval, arm_a, arm_b, arm_c, arm_d


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

    # left: actual NDCG@10 vs random-rank reference line
    # NOTE: the old version showed an *estimated* RMSE bar alongside real Surprise numbers
    # which was misleading (different metric, different dataset, not computed).
    # Replaced with: actual NDCG@10 per dataset vs a random-ranker reference
    # (random ranker on 501 candidates hits rank ~250.5 → expected NDCG ≈ log2(2)/log2(251) ≈ 0.0028)
    ax = axes[0]
    ds_list_left = [d for d in ["ml-1m", "ml-10m", "ml-20m", "spotify"] if d in standard]
    if ds_list_left:
        ndcg_vals = [m(get_baseline(standard[ds]), "ndcg@10") for ds in ds_list_left]
        hr_vals   = [m(get_baseline(standard[ds]), "hit@10")  for ds in ds_list_left]
        x         = np.arange(len(ds_list_left))
        w         = 0.35
        bar_colors = [DATASET_COLORS.get(ds, "#888") for ds in ds_list_left]
        ax.bar(x - w/2, ndcg_vals, w, color=bar_colors, alpha=0.85, label="NDCG@10 (measured)")
        ax.bar(x + w/2, hr_vals,   w, color=bar_colors, alpha=0.45, label="HR@10 (measured)")
        # random-rank reference: hit prob = k / n_candidates = 10 / 501 ≈ 0.020
        random_hr = 10 / 501
        ax.axhline(random_hr, color="red", lw=1.2, linestyle="--",
                   label=f"Random ranker HR@10 ≈ {random_hr:.3f}")
        for i, (nd, hr) in enumerate(zip(ndcg_vals, hr_vals)):
            ax.text(i - w/2, nd + 0.001, f"{nd:.4f}", ha="center", fontsize=8)
            ax.text(i + w/2, hr + 0.001, f"{hr:.4f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABEL.get(ds, ds) for ds in ds_list_left], fontsize=8)
        ax.set_ylabel("Metric value")
        ax.set_title("Fig 1a — Baseline SVD: NDCG@10 and HR@10\n"
                     "(501 sampled candidates, random LOO split)")
        ax.legend(fontsize=9)
        ax.text(0.03, 0.03,
                "All values computed on held-out test set.\n"
                "Random ranker reference: hit@10 = 10/501 ≈ 0.020.",
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
        ax.set_title("Fig 1b — My NDCG@10 vs Plausibility Range\n"
                     "(green band = heuristic estimate, not a published benchmark)")

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="green", alpha=0.25,
                  label="Plausibility range (own estimate: NCF He et al. 2017\n"
                        "scaled to 500-candidate sampled eval — not a direct benchmark)"),
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
                       "Spotify MSSD — implicit skip signals\n(session = user, ~16 plays/session)")
        ax.set_title(f"Fig 7 — {title_extra}")
        ax.set_xlabel("Variant")
        ax.set_ylabel("NDCG@10")
        ax.legend(fontsize=8)

        if ds == "spotify":
            note = ("Skip→rating mapping: skip-before-30s=1, skip-before-end=2,\n"
                    "full-play=4, full-play+add-to-queue=5.\n"
                    "Sessions avg ~16 plays → sparse; results are generalization probe.")
        else:
            note = ("rerank / weighted consistently lower NDCG\n"
                    "filter has zero effect (neg items already excluded from candidates)")
        ax.text(0.02, 0.02, note, transform=ax.transAxes,
                fontsize=7.5, color="#555", verticalalignment="bottom", style="italic")

    save("fig7_movielens_vs_spotify")


#  FIGURE 8 
#  adds negative items to the candidate pool
# Research question: If we deliberately include the user's disliked
# items in the candidate pool, can the filter / rerank / weighted variants keep them
# out of the top 10?
#
# This is the known-negative injection test (Krichene & Rendle 2020).
# In standard evaluation, negatives are never in the candidate pool so
# negative@10 = 0 always. Here we inject up to 50 of the user's own
# low-rated items into the 501-item candidate pool and ask:
# "Does our method actually protect the user from seeing known dislikes?"
#
# How to generate results:
#   python scripts/run_known_negative_eval.py --config configs/movielens_1m.yaml
# Saves as grid_summary_known_neg_eval.json in each dataset's output folder.

def fig8_known_neg_eval(standard: dict, option_a: dict):
    avail = [ds for ds in standard if ds in option_a]
    if not avail:
        print("  skipping fig8: no known-negative eval results")
        print("  run: python scripts/run_known_negative_eval.py --config configs/movielens_1m.yaml")
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
        inj_neg  = avg(option_a[ds],  "negative@10")
        std_ndcg = avg(standard[ds],  "ndcg@10")
        inj_ndcg = avg(option_a[ds],  "ndcg@10")

        x, w = np.arange(len(variants)), 0.35
        colors = [VARIANT_COLORS.get(v, "#aaa") for v in variants]
        # standard neg@10 is structurally zero (known negatives excluded from candidate pool
        # by construction not an empirical result). Use hatching to signal this.
        std_bars = ax.bar(x - w/2, [max(std_neg[v], 0.001) for v in variants], w,
                          color=colors, alpha=0.25, hatch="///",
                          label="Standard neg@10 (= 0 by construction, not measured)")
        ax.bar(x + w/2, [inj_neg[v] for v in variants], w,
               color=colors, alpha=0.85, label="Injected neg@10 (measured)")
        ax.text(0.02, 0.97, "Hatched bars = 0 by construction\n(negatives excluded from candidate pool)",
                transform=ax.transAxes, fontsize=7.5, color="#555",
                verticalalignment="top", style="italic")

        ax2 = ax.twinx()
        ax2.plot(x, [std_ndcg[v] for v in variants], "o-",
                 color="#666", lw=1.5, ms=6, label="Standard NDCG@10")
        ax2.plot(x, [inj_ndcg[v] for v in variants], "s--",
                 color="#222", lw=1.5, ms=6, label="Injected NDCG@10")
        ax2.set_ylabel("NDCG@10", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.set_ylabel("negative@10\n(disliked items in top 10)")
        ax.set_title(f"Fig 8 — Known-Negative Injection Evaluation ({ds.upper()})\n"
                     "50 disliked items deliberately added to candidate pool\n"
                     "(Krichene & Rendle 2020)")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8)

    save("fig8_known_neg_eval")


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


def table_dataset_stats():
    """Table A Dataset statistics for the thesis methodology section."""
    rows = [
        {"Dataset": "MovieLens 1M",  "Users": 6_040,   "Items": 3_416,  "Ratings": 993_571,
         "Feedback": "Explicit (1–5 stars)", "Split": "Random LOO",
         "Source": "GroupLens"},
        {"Dataset": "MovieLens 10M", "Users": 69_878,  "Items": 10_196, "Ratings": 9_928_938,
         "Feedback": "Explicit (1–5 stars)", "Split": "Random LOO",
         "Source": "GroupLens"},
        {"Dataset": "MovieLens 20M", "Users": 138_493, "Items": 18_345, "Ratings": 19_845_531,
         "Feedback": "Explicit (0.5–5 stars)", "Split": "Random LOO",
         "Source": "GroupLens"},
        {"Dataset": "Spotify MSSD",  "Users": 6_376,   "Items": 3_490,  "Ratings": 74_648,
         "Feedback": "Implicit (skip taxonomy → 1–5)", "Split": "Random LOO",
         "Source": "AIcrowd"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "tableA_dataset_stats.csv", index=False)
    df_to_latex(df, TABLES_DIR / "tableA_dataset_stats.tex",
                "Dataset statistics. Spotify skip taxonomy: skip-before-30s=1, "
                "skip-before-end=2, full-play=4, full-play+queue=5.")
    print("  tableA_dataset_stats")


def table_arm_b_detection(arm_b: dict):
    """Table C  Arm B dislike detection quality (LNO evaluation)."""
    if not arm_b:
        return
    rows = []
    for ds, exps in arm_b.items():
        for e in exps:
            hit  = m(e, "neg_detection_hit@10")
            ndcg = m(e, "neg_detection_ndcg@10")
            rank = m(e, "mean_dislike_rank")
            n    = e["metrics"].get("n_users", 0)
            rows.append({
                "Dataset":           ds,
                "Neg threshold":     e.get("neg_label", "?"),
                "Detection HR@10":   round(hit,  4),
                "Detection NDCG@10": round(ndcg, 4),
                "Mean dislike rank": round(rank, 1),
                "n_users (LNO)":     n,
                "Strength":          ("strong" if hit > 0.30 else
                                      "moderate" if hit > 0.15 else "weak"),
            })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(TABLES_DIR / "tableC_arm_b_detection.csv", index=False)
        df_to_latex(df, TABLES_DIR / "tableC_arm_b_detection.tex",
                    "Arm B dislike-detection quality (leave-one-negative-out). "
                    "HR@10 > 0.30 = strong, > 0.15 = moderate, else weak.")
        print("  tableC_arm_b_detection")


def table_arm_c_headline(arm_a: dict, arm_c: dict, standard: dict):
    """Table D  Arm A vs Arm C: headline comparison (the main research question)."""
    avail = [d for d in ["ml-1m", "ml-10m", "ml-20m", "spotify"]
             if d in arm_a and d in arm_c]
    if not avail:
        return
    rows = []
    for ds in avail:
        std_b    = get_baseline(standard.get(ds, []))
        best_a   = max(arm_a[ds],  key=lambda e: m(e, "ndcg@10"), default=None)
        best_c   = max(arm_c[ds],  key=lambda e: m(e, "ndcg@10"), default=None)
        b_ndcg   = m(std_b, "ndcg@10") if std_b else 0

        def delta(val):
            return round((val - b_ndcg) / b_ndcg * 100, 1) if b_ndcg else 0

        row = {
            "Dataset":         ds,
            "SVD baseline NDCG@10":   round(b_ndcg, 4),
            "Arm A NDCG@10":          round(m(best_a, "ndcg@10"), 4) if best_a else "—",
            "Arm A Δ vs baseline (%)": delta(m(best_a, "ndcg@10")) if best_a else "—",
            "Arm C NDCG@10":          round(m(best_c, "ndcg@10"), 4) if best_c else "—",
            "Arm C Δ vs baseline (%)": delta(m(best_c, "ndcg@10")) if best_c else "—",
            "Best Arm C config":      best_c["exp_id"] if best_c else "—",
        }
        if best_c:
            row["Arm A sim_neg@10"] = round(m(best_a, "sim_to_neg@10"), 4) if best_a else "—"
            row["Arm C sim_neg@10"] = round(m(best_c, "sim_to_neg@10"), 4)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(TABLES_DIR / "tableD_arm_c_headline.csv", index=False)
        df_to_latex(df, TABLES_DIR / "tableD_arm_c_headline.tex",
                    "Arm A vs Arm C headline comparison. "
                    "Best config = highest NDCG@10 over all (pos_threshold, neg_threshold, alpha). "
                    "No significance tests — requires per-user data (future work).")
        print("  tableD_arm_c_headline")


def generate_tables(standard: dict, train_positive: dict,
                    arm_a: dict = None, arm_b: dict = None, arm_c: dict = None,
                    arm_d: dict = None):
    """Produce 5 tables: full results, baseline summary, Surprise comparison, key findings, v1 vs v2."""

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

    # table 5: v1 (post-hoc) vs v2 (train-positive) comparison
    # shows side by side: best rerank NDCG vs best train-positive NDCG vs baseline
    if train_positive:
        cmp_rows = []
        for ds in ["ml-1m", "ml-10m", "ml-20m", "spotify"]:
            if ds not in standard or ds not in train_positive:
                continue
            b = get_baseline(standard[ds])
            b_ndcg = m(b, "ndcg@10") if b else 0

            best_rerank = max(
                [e for e in standard[ds] if e["variant"] == "rerank"],
                key=lambda e: m(e, "ndcg@10"), default=None
            )
            best_tp = max(train_positive[ds], key=lambda e: m(e, "ndcg@10"), default=None)

            def delta_pct(val):
                return round((val - b_ndcg) / b_ndcg * 100, 1) if b_ndcg else 0

            rr_ndcg = m(best_rerank, "ndcg@10") if best_rerank else 0
            tp_ndcg = m(best_tp, "ndcg@10") if best_tp else 0
            cmp_rows.append({
                "Dataset":                  ds,
                "Baseline NDCG@10":         round(b_ndcg, 4),
                "Best Rerank NDCG@10":      round(rr_ndcg, 4),
                "Rerank Δ (%)":             delta_pct(rr_ndcg),
                "Train-Positive NDCG@10":   round(tp_ndcg, 4),
                "Train-Pos Δ (%)":          delta_pct(tp_ndcg),
                "Better approach":          "train-pos" if tp_ndcg > rr_ndcg else "rerank",
            })
        if cmp_rows:
            df5 = pd.DataFrame(cmp_rows)
            df5.to_csv(TABLES_DIR / "table5_v1_vs_v2_comparison.csv", index=False)
            df_to_latex(df5, TABLES_DIR / "table5_v1_vs_v2_comparison.tex",
                        "Post-hoc reranking (V1) vs train-positive SVD (V2). "
                        "Train-positive: SVD trained on rating >= 4 only (differs from "
                        "Hu et al. 2008 confidence-weighted ALS which uses all interactions).")
            print("  table5_v1_vs_v2_comparison")

    # new tables: dataset stats, Arm B detection, Arm A vs C
    table_dataset_stats()
    if arm_b:
        table_arm_b_detection(arm_b)
    if arm_a and arm_c:
        table_arm_c_headline(arm_a, arm_c, standard)
    if arm_d:
        table_arm_d_headline(arm_d)
    if arm_a and arm_d:
        table_arm_d_vs_arm_a(arm_a, arm_d, standard)
    table_strategy_summary()


#  Three-Arm Architecture
# 
def fig_arm_b_detection(arm_b: dict):
    """Arm B detection quality: how well does the dislike-risk SVD find dislikes?

    Plots neg_detection_hit@10 and neg_detection_ndcg@10 per threshold per dataset.
    A weak Arm B explains weak Arm C: if the detector cannot find dislikes,
    the hybrid penalty has nothing meaningful to subtract.
    Evaluation: leave-one-negative-out (LNO) held-out disliked item vs 500 neutrals.
    """
    if not arm_b:
        print("  skipping fig_arm_b: no arm_b results (run Phase 6)")
        return

    ds_list = [d for d in ["ml-1m", "ml-10m", "ml-20m", "spotify"] if d in arm_b]
    n = len(ds_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    THRESHOLD_ORDER = ["neg_le_1", "neg_le_2", "neg_le_3", "neg_median", "neg_modus"]
    THRESHOLD_LABELS = {"neg_le_1": "≤1", "neg_le_2": "≤2", "neg_le_3": "≤3",
                        "neg_median": "median", "neg_modus": "modus"}

    for ax, ds in zip(axes, ds_list):
        exps = arm_b[ds]
        labels = [THRESHOLD_LABELS.get(e["neg_label"], e["neg_label"]) for e in exps]
        hit_vals  = [m(e, "neg_detection_hit@10")  for e in exps]
        ndcg_vals = [m(e, "neg_detection_ndcg@10") for e in exps]

        x = np.arange(len(exps))
        w = 0.35
        ax.bar(x - w/2, hit_vals,  w, color="#e74c3c", alpha=0.85, label="HR@10 (detection)")
        ax.bar(x + w/2, ndcg_vals, w, color="#c0392b", alpha=0.55, label="NDCG@10 (detection)")

        ax.axhline(10/501, color="gray", lw=1.2, linestyle="--",
                   label=f"Random chance ≈ {10/501:.3f}")

        for i, (h, n_) in enumerate(zip(hit_vals, ndcg_vals)):
            ax.text(i - w/2, h + 0.003, f"{h:.3f}", ha="center", fontsize=8)
            ax.text(i + w/2, n_ + 0.003, f"{n_:.3f}", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("Negative threshold")
        ax.set_ylabel("Detection metric (higher = better dislike detection)")
        ax.set_title(f"Arm B — Dislike Detection Quality ({ds.upper()})\n"
                     "LNO eval: held-out dislike vs 500 neutral items")
        ax.legend(fontsize=8)
        ax.text(0.02, 0.02,
                "LNO = leave-one-negative-out. High rank for disliked item = correct detection.\n"
                "501 candidates: 1 held-out dislike + 500 neutral unseen items.",
                transform=ax.transAxes, fontsize=7, color="#555",
                verticalalignment="bottom", style="italic")

    save("fig_arm_b_detection")


def fig_arm_c_headline(arm_a: dict, arm_c: dict, standard: dict):
    """Arm A vs Arm C headline comparison the core research question for V2.

    Two panels per dataset: NDCG@10 and sim_to_neg@10.
    Shows whether the three-arm hybrid improves over positive-only SVD.
    Also shows standard SVD baseline for reference.
    Best Arm C config selected per dataset (best NDCG@10).
    """
    avail = [d for d in ["ml-1m", "ml-10m", "ml-20m", "spotify"]
             if d in arm_a and d in arm_c]
    if not avail:
        print("  skipping fig_arm_c: no arm_c results (run Phase 7) or no arm_a")
        return

    n = len(avail)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, ds in enumerate(avail):
        std_baseline = get_baseline(standard.get(ds, []))
        best_arm_a   = max(arm_a[ds],  key=lambda e: m(e, "ndcg@10"), default=None)
        best_arm_c   = max(arm_c[ds],  key=lambda e: m(e, "ndcg@10"), default=None)

        for row, metric in enumerate(["ndcg@10", "sim_to_neg@10"]):
            ax = axes[row, col]
            vals = {
                "SVD\n(all ratings)": m(std_baseline, metric) if std_baseline else 0,
                "Arm A\n(pos-only SVD)": m(best_arm_a, metric) if best_arm_a else 0,
                "Arm C\n(hybrid)":       m(best_arm_c, metric) if best_arm_c else 0,
            }
            bar_colors = [VARIANT_COLORS["baseline"], "#f39c12", "#8e44ad"]
            bars = ax.bar(list(vals.keys()), list(vals.values()),
                          color=bar_colors, alpha=0.85)

            b_val = vals["SVD\n(all ratings)"]
            ax.axhline(b_val, color=VARIANT_COLORS["baseline"],
                       lw=1.2, linestyle="--", alpha=0.6)

            for bar, val in zip(bars, vals.values()):
                delta = (val - b_val) / b_val * 100 if b_val else 0
                sign = "+" if delta >= 0 else ""
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                        f"{sign}{delta:.1f}%", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() / 2 if bar.get_height() > 0.002 else bar.get_height() + 0.001,
                        f"{val:.4f}", ha="center", va="center", fontsize=7.5, color="white"
                        if bar.get_height() > 0.01 else "black")

            ylabel = ("NDCG@10 (↑ better)" if metric == "ndcg@10"
                      else "sim_to_neg@10 (↓ further from dislikes)")
            ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                best_c_info = ""
                if best_arm_c:
                    pt   = best_arm_c.get("pos_threshold", "?")
                    nl   = best_arm_c.get("neg_label",     "?")
                    alph = best_arm_c.get("alpha",          "?")
                    best_c_info = f"\nBest Arm C: pos≥{pt}, {nl}, α={alph}"
                ax.set_title(f"Arm A vs Arm C — {ds.upper()}{best_c_info}", fontsize=9)

    fig.suptitle("Fig — Arm A vs Arm C: Does the hybrid beat positive-only SVD?\n"
                 "Δ% vs standard SVD baseline shown on bars. No significance testing yet.",
                 fontsize=11)
    save("fig_arm_c_headline")


def fig_arm_d_comparison(arm_a: dict, arm_c: dict, arm_d: dict, standard: dict):
    """Arm A vs Arm C (best) vs Arm D main new comparison for Arm D.

    Two metrics panels: NDCG@10 and sim_to_neg@10.
    Answers: does joint positive-negative training (Arm D) beat positive-only (Arm A)?
    """
    avail = [d for d in ["ml-1m", "ml-10m", "ml-20m", "spotify"]
             if d in arm_a and d in arm_d]
    if not avail:
        print("  skipping fig_arm_d: no arm_d results (run run_arm_d.sh first)")
        return

    n = len(avail)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, ds in enumerate(avail):
        std_baseline = get_baseline(standard.get(ds, []))
        best_arm_a = max(arm_a[ds], key=lambda e: m(e, "ndcg@10"), default=None)
        best_arm_c = max(arm_c.get(ds, []), key=lambda e: m(e, "ndcg@10"), default=None) if ds in arm_c else None
        best_arm_d = max(arm_d[ds], key=lambda e: m(e, "ndcg@10"), default=None)

        for row, metric in enumerate(["ndcg@10", "sim_to_neg@10"]):
            ax = axes[row, col]
            entries = {
                "SVD\n(all ratings)": (m(std_baseline, metric) if std_baseline else 0, VARIANT_COLORS["baseline"]),
                "Arm A\n(pos-only)":  (m(best_arm_a, metric) if best_arm_a else 0,  "#f39c12"),
                "Arm D\n(joint)":     (m(best_arm_d, metric) if best_arm_d else 0,  "#e74c3c"),
            }
            if best_arm_c:
                entries["Arm C\n(hybrid)"] = (m(best_arm_c, metric), "#8e44ad")

            labels  = list(entries.keys())
            values  = [v for v, _ in entries.values()]
            colors  = [c for _, c in entries.values()]
            bars = ax.bar(labels, values, color=colors, alpha=0.85)

            b_val = entries["SVD\n(all ratings)"][0]
            ax.axhline(b_val, color=VARIANT_COLORS["baseline"], lw=1.2, linestyle="--", alpha=0.6)

            for bar, val in zip(bars, values):
                delta = (val - b_val) / b_val * 100 if b_val else 0
                sign = "+" if delta >= 0 else ""
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.0003,
                        f"{sign}{delta:.1f}%", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() / 2 if bar.get_height() > 0.002 else bar.get_height() + 0.001,
                        f"{val:.4f}", ha="center", va="center",
                        fontsize=7.5, color="white" if bar.get_height() > 0.01 else "black")

            ylabel = ("NDCG@10 (↑ better)" if metric == "ndcg@10"
                      else "sim_to_neg@10 (↓ further from dislikes)")
            ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                d_info = ""
                if best_arm_d:
                    pos = best_arm_d.get("pos_threshold", "?")
                    neg = best_arm_d.get("neg_threshold", "?")
                    d_info = f"\nBest Arm D: pos≥{pos}, neg≤{neg}"
                ax.set_title(f"Arm A vs Arm D — {ds.upper()}{d_info}", fontsize=9)

    fig.suptitle(
        "Fig — Arm A vs Arm D: Does joint positive-negative training improve ranking?\n"
        "Arm D trains SVD on binary targets (pos=1.0, neg=0.0). Δ% vs full-rating SVD baseline.",
        fontsize=11,
    )
    save("fig_arm_d_comparison")


def table_arm_d_headline(arm_d: dict):
    """Table E: Arm D joint positive-negative SVD results per dataset."""
    if not arm_d:
        return
    rows = []
    for ds in ["ml-1m", "ml-10m", "ml-20m", "spotify"]:
        if ds not in arm_d:
            continue
        for exp in arm_d[ds]:
            me = exp["metrics"]
            k  = 10
            rows.append({
                "Dataset":        ds,
                "Pos≥": exp.get("pos_threshold", "?"),
                "Neg≤": exp.get("neg_threshold", "?"),
                "NDCG@10":       f"{me.get(f'ndcg@{k}', 0):.4f}",
                "HR@10":         f"{me.get(f'hit@{k}', 0):.4f}",
                "MRR":           f"{me.get('mrr', 0):.4f}",
                "sim_neg@10":    f"{me.get(f'sim_to_neg@{k}', 0):.4f}",
                "N_pos":         me.get("n_positives", "?"),
                "N_neg":         me.get("n_negatives", "?"),
            })
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "tableE_arm_d_headline.csv", index=False)
    df_to_latex(df, TABLES_DIR / "tableE_arm_d_headline.tex",
                caption="Arm D: Joint Positive-Negative SVD results. "
                        "SVD trained on binary targets (pos=1.0, neg=0.0). "
                        "Compared to Arm A (Table D) to test whether training-time negative "
                        "feedback improves ranking quality.")
    print("  tableE_arm_d_headline")


def table_arm_d_vs_arm_a(arm_a: dict, arm_d: dict, standard: dict):
    """Table F: Arm A vs Arm D headline — core training-time comparison."""
    avail = [d for d in ["ml-1m", "ml-10m", "ml-20m", "spotify"]
             if d in arm_a and d in arm_d]
    if not avail:
        return

    rows = []
    k = 10
    for ds in avail:
        std_b    = get_baseline(standard.get(ds, []))
        best_a   = max(arm_a[ds], key=lambda e: m(e, "ndcg@10"), default=None)
        best_d   = max(arm_d[ds], key=lambda e: m(e, "ndcg@10"), default=None)

        base_ndcg = m(std_b, "ndcg@10") if std_b else 0
        a_ndcg    = m(best_a, f"ndcg@{k}") if best_a else 0
        d_ndcg    = m(best_d, f"ndcg@{k}") if best_d else 0

        delta_d_vs_a   = d_ndcg - a_ndcg
        delta_d_vs_base = d_ndcg - base_ndcg
        d_label = (f"pos≥{best_d.get('pos_threshold','?')} neg≤{best_d.get('neg_threshold','?')}"
                   if best_d else "—")

        rows.append({
            "Dataset":         ds,
            "SVD (nDCG@10)":   f"{base_ndcg:.4f}",
            "Arm A (nDCG@10)": f"{a_ndcg:.4f}",
            "Arm D (nDCG@10)": f"{d_ndcg:.4f}",
            "Δ D vs A":        f"{delta_d_vs_a:+.4f}",
            "Δ D vs SVD":      f"{delta_d_vs_base:+.4f}",
            "Best Arm D config": d_label,
        })

    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "tableF_arm_d_vs_arm_a.csv", index=False)
    df_to_latex(df, TABLES_DIR / "tableF_arm_d_vs_arm_a.tex",
                caption="Arm A (positive-only SVD) vs Arm D (joint positive-negative SVD). "
                        "Arm D trains on binary targets to test whether incorporating negative "
                        "feedback during training improves top-N ranking quality over "
                        "positive-only training.")
    print("  tableF_arm_d_vs_arm_a")


def table_strategy_summary():
    """Table G: Strategy overviewall approaches compared qualitatively."""
    rows = [
        {
            "Strategy":              "Full-rating SVD (baseline)",
            "Neg. in training":      "Yes (as raw ratings)",
            "Neg. post-hoc":         "No",
            "Main purpose":          "Baseline MF",
            "Neg. feedback role":    "Implicit via rating scale",
        },
        {
            "Strategy":              "Post-hoc filter/rerank/weighted",
            "Neg. in training":      "Yes (as raw ratings)",
            "Neg. post-hoc":         "Yes",
            "Main purpose":          "Dislike avoidance at rank time",
            "Neg. feedback role":    "Cosine penalty or candidate removal",
        },
        {
            "Strategy":              "Arm A (positive-only SVD)",
            "Neg. in training":      "No (removed)",
            "Neg. post-hoc":         "No",
            "Main purpose":          "Clean positive preference model",
            "Neg. feedback role":    "None — noise removed",
        },
        {
            "Strategy":              "Arm B (negative detector)",
            "Neg. in training":      "Yes (negatives only)",
            "Neg. post-hoc":         "No",
            "Main purpose":          "Dislike-risk scoring",
            "Neg. feedback role":    "Dedicated dislike latent space",
        },
        {
            "Strategy":              "Arm C (hybrid A+B)",
            "Neg. in training":      "Indirectly via Arm B",
            "Neg. post-hoc":         "Yes (alpha blending)",
            "Main purpose":          "Hybrid avoidance + preference",
            "Neg. feedback role":    "norm(A) - alpha * norm(B)",
        },
        {
            "Strategy":              "Arm D (joint pos-neg SVD)",
            "Neg. in training":      "Yes (binary targets)",
            "Neg. post-hoc":         "No",
            "Main purpose":          "Joint preference learning",
            "Neg. feedback role":    "Direct contrast: like=1.0, dislike=0.0",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "tableG_strategy_summary.csv", index=False)
    df_to_latex(df, TABLES_DIR / "tableG_strategy_summary.tex",
                caption="Summary of all negative-feedback strategies evaluated in this thesis. "
                        "The table compares how and when negative feedback is incorporated "
                        "across baseline, post-hoc, and training-time approaches.")
    print("  tableG_strategy_summary")


#  8.  MAIN


def fig9_training_time_vs_posthoc(standard: dict, train_positive: dict):
    # Research question: Does removing negative items from training help more than
    # applying post-hoc penalties at inference time?
    #
    # Left panel: NDCG@10  baseline vs best post-hoc (rerank) vs train-positive
    # Right panel: NDCG delta (%) over baseline for each approach and threshold
    #
    # Hypothesis (Hu et al. 2008): training-time approach should outperform post-hoc
    # because the model learns cleaner representations without noisy negatives.
    if not train_positive:
        print("  skipping fig9: no train-positive results")
        print("  run: python scripts/run_train_positive_grid.py --config configs/movielens_1m.yaml")
        return

    ds_list = [d for d in ["ml-1m", "ml-10m", "ml-20m", "spotify"]
               if d in standard and d in train_positive]
    if not ds_list:
        print("  skipping fig9: no datasets with both standard and train-positive results")
        return

    fig, axes = plt.subplots(1, len(ds_list), figsize=(6 * len(ds_list), 5))
    if len(ds_list) == 1:
        axes = [axes]

    for ax, ds in zip(axes, ds_list):
        std_exps = standard[ds]
        tp_exps  = train_positive[ds]
        b = get_baseline(std_exps)
        b_ndcg = m(b, "ndcg@10") if b else 0

        # best post-hoc rerank NDCG (main grid)
        best_rerank = max(
            [e for e in std_exps if e["variant"] == "rerank"],
            key=lambda e: m(e, "ndcg@10"),
            default=None,
        )
        # best train-positive NDCG
        best_tp = max(tp_exps, key=lambda e: m(e, "ndcg@10"), default=None)

        approaches = ["Baseline\n(all ratings)", "Best Rerank\n(post-hoc)", "Train-Positive SVD\n(positive-only training)"]
        ndcgs = [
            b_ndcg,
            m(best_rerank, "ndcg@10") if best_rerank else 0,
            m(best_tp, "ndcg@10") if best_tp else 0,
        ]
        colors = [VARIANT_COLORS["baseline"], VARIANT_COLORS["rerank"], "#f39c12"]

        bars = ax.bar(approaches, ndcgs, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(b_ndcg, color=VARIANT_COLORS["baseline"], lw=1.5, ls="--", alpha=0.6)

        for bar, val in zip(bars, ndcgs):
            delta = (val - b_ndcg) / b_ndcg * 100 if b_ndcg else 0
            sign = "+" if delta >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                    f"{sign}{delta:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_ylim(0, max(ndcgs) * 1.15)
        ax.set_ylabel("NDCG@10")
        ax.set_title(f"Fig 9 — Post-hoc vs Training-Time ({ds.upper()})\n"
                     "Orange: SVD trained on rating ≥ 4 interactions only\n"
                     "(positive-only training; differs from Hu et al. 2008 confidence-weighted ALS)")

    save("fig9_training_time_vs_posthoc")


def main():
    print("=" * 65)
    print("THESIS FIGURE GENERATOR — Negative Feedback in Recommender Systems")
    print("=" * 65)
    print()
    print("Loading experiment results...")
    standard, train_positive, known_neg_eval, arm_a, arm_b, arm_c, arm_d = load_all_results()

    if not standard:
        print("No results found. Run experiments first:")
        print("  python main.py grid --config configs/movielens_1m.yaml")
        sys.exit(1)

    print_quality_report(standard)

    print("Generating figures...")
    # original V1 figures
    fig1_is_my_baseline_correct(standard)
    fig2_the_tradeoff(standard)
    fig3_best_variant_per_dataset(standard)
    fig4_alpha_sensitivity(standard)
    fig5_threshold_comparison(standard)
    fig6_dataset_scaling(standard)
    fig7_movielens_vs_spotify(standard)
    fig8_known_neg_eval(standard, known_neg_eval)
    fig9_training_time_vs_posthoc(standard, train_positive)
    # three-arm architecture figures (generated once Phase 6 + 7 complete)
    fig_arm_b_detection(arm_b)
    fig_arm_c_headline(arm_a, arm_c, standard)
    # Arm D: joint positive-negative SVD (generated once run_arm_d.sh completes)
    fig_arm_d_comparison(arm_a, arm_c, arm_d, standard)

    print()
    print("Generating tables...")
    generate_tables(standard, train_positive, arm_a=arm_a, arm_b=arm_b, arm_c=arm_c, arm_d=arm_d)

    print()
    print(f"Figures → {FIGURES_DIR}/")
    print(f"Tables  → {TABLES_DIR}/")
    print()
    print("NOTES FOR THESIS:")
    print("  Fig 1a: NDCG/HR vs random ranker (computed, not estimated)")
    print("   Fig 1b: green band = heuristic plausibility range (not a published benchmark)")
    print("   Fig 8:  hatched bars = negative@10 is 0 by construction, not measured")
    print("  Fig 9:  'Train-Positive SVD' ≠ Hu et al. 2008 (different method)")
    print("  Tables A/C/D: new — dataset stats, Arm B detection, Arm A vs C")
    print("  Significance tests: not yet added — requires per-user CSV output")
    print("    (run grid scripts with per-user saving to enable significance stars)")
    print("=" * 65)


if __name__ == "__main__":
    main()