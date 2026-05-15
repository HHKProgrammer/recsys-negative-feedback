"""
Training-Time Negative Feedback Grid


The post-hoc variants (filter / rerank / weighted) all share the same SVD model
that was trained on EVERY rating including 1-star ones. They patch the ranking at
inference time AFTER the model has already learned from dislikes.

This script tests a fundamentally different hypothesis (Hu et al. 2008, ICDM):

  "If you remove or down-weight negatively rated items from the training data,
   does SVD learn better representations, leading to higher NDCG?"

How it works

For each threshold configuration, here:
  1. Identify each user's negative items (same logic as filter/rerank/weighted)
  2. REMOVE those (user, item) pairs from the training set
  3. Train a fresh SVD on the cleaned, positive-only training data
  4. Evaluate using the standard protocol (no post-hoc adjustments)

Reference
Hu, Koren & Volinsky (2008) "Collaborative Filtering for Implicit Feedback Datasets" (ICDM)
https://ieeexplore.ieee.org/document/4781121
  → Section 3: "confidence weights"negative/irrelevant interactions get low confidence
  →my approach: confidence 0 for known dislikes, 1 for everything else (hard variant)

Key differences from post-hoc variants
  Post-hoc (filter/rerank/weighted): one SVD trained on all ratings, ranking patched later
  Training-time (here script):       one SVD per threshold, trained only on positive ratings

Expected behaviour
Removing noisy negative-rated items from training should:
  + Increase NDCG (cleaner training signal, model focuses on positives)
  - Reduce training data size (fewer rows → less signal overall)
  Net effect depends on how many negatives exist and how noisy they are.

Threshold configurations (same 5 as main grid)
  fixed_1:  keep ratings >= 2  (remove only 1-star items)
  fixed_2:  keep ratings >= 3  (remove 1- and 2-star items)
  fixed_3:  keep ratings >= 4  (keep only 4- and 5-star items most aggressive)
  median:   keep ratings >= user's median rating
  modus:    keep ratings >= user's most common rating

Output
  outputs/<dataset>/grid_summary_train_positive.json

Usage
  python scripts/run_train_positive_grid.py --config configs/movielens_1m.yaml
  python scripts/run_train_positive_grid.py --config configs/movielens_10m.yaml
  python scripts/run_train_positive_grid.py --config configs/movielens_20m.yaml
  python scripts/run_train_positive_grid.py --config configs/spotify.yaml
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.threshold_utils import get_all_user_negative_items
from src.eval.ranking_metrics import evaluate_ranking
from src.models.svd_baseline import SVDBaseline
from src.utils.config import ExperimentConfig
from src.utils.io import load_parquet
from src.utils.seed import set_global_seed


def build_positive_train_df(
    train_df: pd.DataFrame,
    user_negative_items: Dict[int, set],
) -> pd.DataFrame:
    """
    Remove each user's negative items from the training set.

    Uses a fast vectorized merge rather than row-by-row iteration so it
    scales to 10M+ row datasets in under a second.
    """
    if not user_negative_items:
        return train_df

    # Build a (userId, movieId) DataFrame of pairs to exclude
    rows = [
        (uid, item)
        for uid, items in user_negative_items.items()
        for item in items
    ]
    if not rows:
        return train_df

    excl = pd.DataFrame(rows, columns=["userId", "movieId"])
    excl["_excl"] = True

    merged = train_df.merge(excl, on=["userId", "movieId"], how="left")
    positive_df = train_df[merged["_excl"].isna()].copy()
    return positive_df


def run_one_config(
    config: ExperimentConfig,
    threshold_type: str,
    fixed_threshold: Optional[int],
    max_users: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    set_global_seed(seed)

    proc = config.data.processed_path
    train_df        = load_parquet(proc + config.splits.train_file)
    test_df         = load_parquet(proc + config.splits.test_file)
    user_thresholds = load_parquet(proc + config.splits.user_thresholds_file)

    # identify negatives using same logic as post-hoc variants
    user_neg = get_all_user_negative_items(
        train_df, threshold_type, fixed_threshold, user_thresholds
    )

    n_neg_interactions = sum(len(v) for v in user_neg.values())
    pos_train_df = build_positive_train_df(train_df, user_neg)

    pct_removed = (1 - len(pos_train_df) / len(train_df)) * 100
    print(f"    Removed {len(train_df) - len(pos_train_df):,} negative interactions "
          f"({pct_removed:.1f}%) — {len(pos_train_df):,} remain")

    # train a fresh SVD on the cleaned data
    model = SVDBaseline(
        n_factors=config.model.n_factors,
        n_epochs=config.model.n_epochs,
        lr_all=config.model.lr_all,
        reg_all=config.model.reg_all,
        random_state=seed,
    )
    model.fit(pos_train_df)

    all_items = set(pos_train_df["movieId"].unique())
    dummy_neg = {uid: set() for uid in test_df["userId"].unique()}

    aggregated, _ = evaluate_ranking(
        model=model,
        test_df=test_df,
        train_df=pos_train_df,
        user_negative_items=dummy_neg,
        all_items=all_items,
        k=config.eval.k,
        n_candidates=config.eval.n_candidates,
        seed=config.eval.random_seed,
        max_users=max_users,
        similarity_fn=model.get_similarity,
    )
    aggregated["n_neg_interactions_removed"] = n_neg_interactions
    aggregated["pct_train_removed"] = round(pct_removed, 2)
    return aggregated


# same 5 threshold configs as the main grid (no alph no inference-time penalty)
def build_threshold_grid(config: ExperimentConfig) -> List[Dict]:
    grid = []
    for t in config.negative_feedback.fixed_thresholds:
        grid.append({"threshold_type": "fixed", "fixed_threshold": t})
    if config.negative_feedback.use_median:
        grid.append({"threshold_type": "median", "fixed_threshold": None})
    if config.negative_feedback.use_modus:
        grid.append({"threshold_type": "modus", "fixed_threshold": None})
    return grid


def exp_id(threshold_type: str, fixed_threshold: Optional[int]) -> str:
    if threshold_type == "fixed":
        return f"train_positive_fixed_{fixed_threshold}"
    return f"train_positive_{threshold_type}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_users", type=int, default=None,
                        help="Limit to first N users for a quick test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    output_path = Path(config.output_dir).parent / "grid_summary_train_positive.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # load existing results so we can resume if interrupted
    if output_path.exists():
        with open(output_path) as f:
            summary = json.load(f)
        done = {e["exp_id"] for e in summary.get("experiments", [])}
    else:
        summary = {"experiments": [], "meta": {}}
        done = set()

    grid = build_threshold_grid(config)

    print(f"\nTraining-Time Positive-Only Grid")
    print(f"  config:    {args.config}")
    print(f"  output:    {output_path}")
    print(f"  reference: Hu, Koren & Volinsky 2008 (ICDM) — confidence-weighted training")
    print(f"  approach:  remove user-specific negative (low-rated) items before fitting SVD")
    print()

    total = len(grid)
    for i, cfg in enumerate(grid, 1):
        eid = exp_id(cfg["threshold_type"], cfg["fixed_threshold"])
        if eid in done:
            print(f"  [{i}/{total}] SKIP (done): {eid}")
            continue

        print(f"  [{i}/{total}] Running: {eid}")
        try:
            metrics = run_one_config(
                config=config,
                threshold_type=cfg["threshold_type"],
                fixed_threshold=cfg["fixed_threshold"],
                max_users=args.max_users,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"    ERROR: {exc}")
            raise

        k = config.eval.k
        print(f"    NDCG@{k}={metrics.get(f'ndcg@{k}', 0):.4f}  "
              f"HR@{k}={metrics.get(f'hit@{k}', 0):.4f}  "
              f"MRR={metrics.get('mrr', 0):.4f}")

        record = {
            "exp_id":          eid,
            "variant":         "train_positive",
            "threshold_type":  cfg["threshold_type"],
            "fixed_threshold": cfg["fixed_threshold"],
            "alpha":           None,
            "metrics":         metrics,
        }
        summary["experiments"].append(record)
        done.add(eid)

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    summary["meta"] = {
        "approach":    "training_time_negative_removal",
        "reference":   "Hu Koren Volinsky 2008 ICDM — confidence-weighted collaborative filtering",
        "description": "SVD trained only on positive-rated items per user (negatives removed before fit)",
        "completed_at": datetime.now().isoformat(),
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results in {output_path}")
    print("Run generate_all_figures.py to include these in the comparison figures.")


if __name__ == "__main__":
    main()
