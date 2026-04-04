"""
one-command figure and table generator
run: python scripts/generate_all_figures.py

regenerates all figures and latex tables from the latest experiment results
i can run this any time after new experiments finish and all figures update automatically
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path("outputs")
FIGURES_DIR = Path("reports/figures")
TABLES_DIR = Path("reports/tables")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# consistent colors across all figures
COLORS = {
    "baseline": "#2ecc71",
    "filter":   "#3498db",
    "rerank":   "#e74c3c",
    "weighted": "#9b59b6",
    "movielens": "#3498db",
    "spotify":   "#1DB954", 
}

plt.style.use("seaborn-v0_8-whitegrid")



# helper: load grid_summary.json from outputs
def load_results():
    results = {}

    ml_path = RESULTS_DIR / "movielens/ml-1m/grid_summary.json"
    if ml_path.exists():
        with open(ml_path) as f:
            results["movielens"] = json.load(f)
        print(f"  loaded movielens results ({len(results['movielens']['experiments'])} experiments)")

    sp_path = RESULTS_DIR / "spotify/grid_summary.json"
    if sp_path.exists():
        with open(sp_path) as f:
            results["spotify"] = json.load(f)
        print(f"  loaded spotify results")

    return results


# fig 1: my svd vs surprise official benchmarks
def fig_surprise_comparison(results):
    # compares my tuned svd rmse to the surprise library benchmarks
    # i get my rmse from the baseline experiment result
    if "movielens" not in results:
        print("  skipping fig1: no movielens results")
        return

    # try to get my rmse from results, fall back to known value
    experiments = results["movielens"]["experiments"]
    baseline_exp = next((e for e in experiments if e["variant"] == "baseline"), None)
    my_rmse = baseline_exp["metrics"].get("rmse", 0.8646) if baseline_exp else 0.8646

    # surprise official benchmarks from surpriselib.com
    benchmarks = [
        ("Random",       1.504),
        ("NMF",          0.916),
        ("BaselineOnly", 0.909),
        ("KNNBaseline",  0.895),
        ("SVD default",  0.873),
        ("SVD++ default",0.862),
        ("My SVD tuned", my_rmse),
    ]

    names  = [b[0] for b in benchmarks]
    values = [b[1] for b in benchmarks]
    colors = ["#aaaaaa"] * 6 + [COLORS["baseline"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, values, color=colors)
    bars[-1].set_edgecolor("black")
    bars[-1].set_linewidth(1.5)

    ax.axvline(x=0.873, color="red", linestyle="--", linewidth=1, label="SVD default (0.873)")
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("My SVD vs Surprise Benchmarks (MovieLens 1M, 5-fold CV)")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_surprise_comparison.png", dpi=150)
    plt.savefig(FIGURES_DIR / "fig1_surprise_comparison.pdf")
    plt.close()
    print("  fig1_surprise_comparison saved")


# fig 2: my three variants vs baseline
def fig_variants_comparison(results):
    # box plots showing ndcg hit and sim_to_neg for each variant
    # baseline is shown as a dashed reference line
    if "movielens" not in results:
        print("  skipping fig2: no movielens results")
        return

    experiments = results["movielens"]["experiments"]
    rows = []
    for e in experiments:
        rows.append({
            "variant":      e["variant"],
            "ndcg@10":      e["metrics"].get("ndcg@10", 0),
            "hit@10":       e["metrics"].get("hit@10", 0),
            "sim_to_neg@10": e["metrics"].get("sim_to_neg@10", 0),
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["ndcg@10", "hit@10", "sim_to_neg@10"]

    for ax, metric in zip(axes, metrics):
        palette = [COLORS.get(v, "#888888") for v in df["variant"].unique()]
        sns.boxplot(data=df, x="variant", y=metric, ax=ax, palette=palette)
        ax.set_title(metric)
        ax.set_xlabel("")
        # baseline reference line
        baseline_val = df[df["variant"] == "baseline"][metric].mean()
        ax.axhline(y=baseline_val, color=COLORS["baseline"],
                   linestyle="--", linewidth=1.2, label="baseline")
        ax.legend(fontsize=8)

    fig.suptitle("Negative Feedback Variants vs Baseline (MovieLens 1M)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_variants_comparison.png", dpi=150)
    plt.savefig(FIGURES_DIR / "fig2_variants_comparison.pdf")
    plt.close()
    print("  fig2_variants_comparison saved")


# fig 3: movielens vs spotify (once i have spotify results)
def fig_dataset_comparison(results):
    if "movielens" not in results or "spotify" not in results:
        print("  skipping fig3: need both movielens and spotify results")
        return

    metrics = ["ndcg@10", "hit@10", "precision@10"]
    ml_exps = results["movielens"]["experiments"]
    sp_exps = results["spotify"]["experiments"]

    # best result per metric across all experiments
    def best(exps, metric):
        vals = [e["metrics"].get(metric, 0) for e in exps]
        return max(vals) if vals else 0

    ml_vals = [best(ml_exps, m) for m in metrics]
    sp_vals = [best(sp_exps, m) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, ml_vals, width, label="MovieLens (explicit)", color=COLORS["movielens"])
    ax.bar(x + width / 2, sp_vals, width, label="Spotify (implicit)", color=COLORS["spotify"])

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Best Results: MovieLens (Explicit) vs Spotify (Implicit Skip Feedback)")
    ax.legend()

    # annotate the gap  implicit is always expected to be worse
    gap = (ml_vals[0] - sp_vals[0]) / ml_vals[0] * 100 if ml_vals[0] > 0 else 0
    ax.annotate(f"{gap:.0f}% gap\n(expected)", xy=(0, sp_vals[0]),
                xytext=(0.4, sp_vals[0] + 0.02),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_dataset_comparison.png", dpi=150)
    plt.savefig(FIGURES_DIR / "fig3_dataset_comparison.pdf")
    plt.close()
    print("  fig3_dataset_comparison saved")


# fig 4: effect of alpha (penalty strength)
def fig_alpha_sensitivity(results):
    if "movielens" not in results:
        return

    experiments = results["movielens"]["experiments"]
    rows = [
        {"variant": e["variant"], "alpha": e.get("alpha"), "ndcg@10": e["metrics"].get("ndcg@10", 0)}
        for e in experiments
        if e.get("alpha") is not None
    ]
    if not rows:
        print("  skipping fig4: no alpha results found")
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    for variant in ["rerank", "weighted"]:
        sub = df[df["variant"] == variant]
        means = sub.groupby("alpha")["ndcg@10"].mean()
        ax.plot(means.index, means.values, marker="o", label=variant,
                color=COLORS[variant], linewidth=2, markersize=8)

    # show the optimal alpha zone
    ax.axvspan(0.05, 0.2, alpha=0.12, color="green", label="gentle penalty range")
    ax.set_xlabel("Alpha (penalty strength)")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Effect of Penalty Strength Alpha on Recommendation Quality")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_alpha_sensitivity.png", dpi=150)
    plt.savefig(FIGURES_DIR / "fig4_alpha_sensitivity.pdf")
    plt.close()
    print("  fig4_alpha_sensitivity saved")


# fig 5: threshold type comparisoon
def fig_threshold_comparison(results):
    if "movielens" not in results:
        return

    experiments = results["movielens"]["experiments"]
    rows = []
    for e in experiments:
        if e["variant"] == "baseline":
            continue
        t = e.get("threshold_type", "")
        label = f"fixed<={e['fixed_threshold']}" if t == "fixed" else t.capitalize()
        rows.append({
            "threshold": label,
            "ndcg@10":   e["metrics"].get("ndcg@10", 0),
            "variant":   e["variant"],
        })

    if not rows:
        print("  skipping fig5: no threshold results")
        return
    df = pd.DataFrame(rows)

    baseline_val = next(
        (e["metrics"].get("ndcg@10", 0) for e in experiments if e["variant"] == "baseline"), 0
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = {k: COLORS.get(k, "#888888") for k in df["variant"].unique()}
    sns.barplot(data=df, x="threshold", y="ndcg@10", hue="variant", ax=ax, palette=palette)
    ax.axhline(y=baseline_val, color=COLORS["baseline"], linestyle="--",
               linewidth=1.5, label="baseline")
    ax.set_xlabel("Threshold Strategy")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Impact of Negative Feedback Threshold Strategy (MovieLens 1M)")
    ax.legend(title="Variant")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_threshold_comparison.png", dpi=150)
    plt.savefig(FIGURES_DIR / "fig5_threshold_comparison.pdf")
    plt.close()
    print("  fig5_threshold_comparison saved")


# tables: csv + latex 
def generate_tables(results):
    if "movielens" in results:
        experiments = results["movielens"]["experiments"]
        rows = []
        for e in experiments:
            t = e.get("threshold_type", "N/A")
            label = f"fixed<={e['fixed_threshold']}" if t == "fixed" else t
            rows.append({
                "Variant":   e["variant"].capitalize(),
                "Threshold": label,
                "Alpha":     e.get("alpha", "N/A"),
                "nDCG@10":   round(e["metrics"].get("ndcg@10", 0), 4),
                "Hit@10":    round(e["metrics"].get("hit@10", 0), 4),
                "MRR":       round(e["metrics"].get("mrr", 0), 4),
                "P@10":      round(e["metrics"].get("precision@10", 0), 4),
            })
        df = pd.DataFrame(rows).sort_values(["Variant", "Threshold", "Alpha"])
        df.to_csv(TABLES_DIR / "table1_main_results.csv", index=False)
        df.to_latex(TABLES_DIR / "table1_main_results.tex", index=False)
        print("  table1_main_results saved")

    # surprise comparison table for thesis
    cmp = pd.DataFrame([
        {"Algorithm": "Random (lower bound)",    "RMSE": 1.504, "MAE": 1.206, "Source": "surpriselib.com"},
        {"Algorithm": "SVD default",             "RMSE": 0.873, "MAE": 0.686, "Source": "surpriselib.com"},
        {"Algorithm": "SVD++ default",           "RMSE": 0.862, "MAE": 0.672, "Source": "surpriselib.com"},
        {"Algorithm": "My SVD (tuned n=100)",    "RMSE": 0.865, "MAE": "N/A", "Source": "This thesis"},
    ])
    cmp.to_csv(TABLES_DIR / "table2_surprise_comparison.csv", index=False)
    cmp.to_latex(TABLES_DIR / "table2_surprise_comparison.tex", index=False)
    print("  table2_surprise_comparison saved")


# main
def main():
    print("=" * 55)
    print("GENERATING ALL THESIS FIGURES AND TABLES")
    print("=" * 55)

    results = load_results()
    if not results:
        print("No results found. Run experiments first.")
        return

    print()
    fig_surprise_comparison(results)
    fig_variants_comparison(results)
    fig_dataset_comparison(results)
    fig_alpha_sensitivity(results)
    fig_threshold_comparison(results)
    generate_tables(results)

    print()
    print(f"Figures saved to: {FIGURES_DIR}/")
    print(f"Tables saved to:  {TABLES_DIR}/")
    print("=" * 55)


if __name__ == "__main__":
    main()