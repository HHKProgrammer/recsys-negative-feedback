"""
Hyperparameter Tuning for SVD Arms


Research context

This thesis frames recommendation as a TOP-N RANKING task, not a rating-regression task.
Ratings are raw material: high ratings → positive preference signal, low ratings → negative
preference signal. The goal is not to predict the exact star value (RMSE/MAE) but to rank
items so that relevant positives appear high and disliked items are avoided.

Because the three arms train on fundamentally different data distributions
(all interactions / positive-only / negative-only), each arm needs its own
hyperparameter search. Using a single shared config would give unfair advantage
to whichever arm the default params were originally tuned for.

Arm A :Positive Preference Model
  Trains SVD on interactions with rating >= positive_threshold.
  Objective: maximize nDCG@10 on the validation set (positive leave-one-out).

Arm B: Negative Preference (Dislike-Risk) Model
  Trains SVD on interactions with rating <= negative_threshold.
  Arm B is NOT a recommender it is a dislike-risk DETECTOR.
  Objective: maximize negative_detection_hit@10  how often does the model
             correctly rank a held-out disliked item in its own top-10?
  Evaluation uses a leave-one-negative-out split created on-the-fly from
  the negative training interactions of train_inner.

Search space
  n_factors:  Arm A [10,20,30,50,75,100,150,200,300,500], Arm B [10,20,30,50,75,100,150,200]
  n_epochs:   [20, 30, 50, 75, 100]
  lr_all:     [0.001, 0.002, 0.005, 0.01, 0.02]
  reg_all:    Arm A [0.005,0.01,0.03,0.05,0.1,0.2], Arm B [0.03,0.05,0.1,0.2,0.5]
  biased:     [True, False]

Uses Optuna (Bayesian TPE) with n_trials=200. Falls back to random search if
Optuna is not available.

Output files
  outputs/<dataset>/tuning/arm_a/pos_ge_<threshold>/best_params.json
  outputs/<dataset>/tuning/arm_b/neg_le_<threshold>/best_params.json  (also: median, modus)
  outputs/<dataset>/tuning/arm_a/pos_ge_<threshold>/study.pkl  (Optuna study, if available)

Usage
  # Arm A, positive threshold 4 (standard):
  python scripts/run_hyperparameter_tuning.py --config configs/movielens_1m.yaml --arm a --threshold 4

  # Arm A, strict positive:
  python scripts/run_hyperparameter_tuning.py --config configs/movielens_1m.yaml --arm a --threshold 5

  # Arm B, negative threshold 2:
  python scripts/run_hyperparameter_tuning.py --config configs/movielens_1m.yaml --arm b --threshold 2

  # Arm B, per-user median:
  python scripts/run_hyperparameter_tuning.py --config configs/movielens_1m.yaml --arm b --threshold median

  # Quick sanity check (5 trials, 500 users):
  python scripts/run_hyperparameter_tuning.py --config configs/movielens_1m.yaml --arm a --threshold 4 --n_trials 5 --max_users 500
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.threshold_utils import get_all_user_negative_items
from src.eval.metrics import (
    ndcg_at_k,
    negative_detection_hit_at_k,
    negative_detection_ndcg_at_k,
)
from src.models.svd_baseline import SVDBaseline
from src.utils.config import ExperimentConfig
from src.utils.io import load_parquet
from src.utils.seed import set_global_seed

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

SEARCH_SPACE = {
    "a": {
        "n_factors":  [10, 20, 30, 50, 75, 100, 150, 200],
        "n_epochs":   [20, 30, 50, 75, 100],
        "lr_all":     [0.001, 0.002, 0.005, 0.01, 0.02],
        "reg_all":    [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
        "biased":     [True, False],
    },
    "b": {
        "n_factors":  [10, 20, 30, 50, 75, 100, 150, 200],
        "n_epochs":   [20, 30, 50, 75, 100],
        "lr_all":     [0.001, 0.002, 0.005, 0.01, 0.02],
        "reg_all":    [0.03, 0.05, 0.1, 0.2, 0.5],  
        "biased":     [True, False],
    },
}


# Data preparation
def filter_positive(train_df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Keep only interactions with rating >= threshold (Arm A training data)."""
    return train_df[train_df["rating"] >= threshold].copy()


def filter_negative(
    train_df: pd.DataFrame,
    threshold,
    user_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only interactions with rating <= threshold (Arm B training data).
    threshold can be an integer or 'median' / 'modus'.
    """
    if isinstance(threshold, int):
        return train_df[train_df["rating"] <= threshold].copy()

    # adaptive: per-user threshold from user_thresholds
    col = "median_rating" if threshold == "median" else "modus_rating"
    merged = train_df.merge(user_thresholds[["userId", col]], on="userId", how="left")
    mask = merged["rating"] < merged[col]  
    return train_df[mask].copy()


def build_arm_a_val(
    train_inner: pd.DataFrame,
    val: pd.DataFrame,
    pos_threshold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (positive_train, standard_val)."""
    pos_train = filter_positive(train_inner, pos_threshold)
    return pos_train, val


def build_arm_b_neg_val(
    train_inner: pd.DataFrame,
    threshold,
    user_thresholds: pd.DataFrame,
    rng: random.Random,
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """Return (negative_train_without_holdouts, {userId: held_out_negative_item}).

    For each user who has ≥2 negative interactions in train_inner, one negative
    item is held out for evaluation (leave-one-negative-out).
    Users with only 1 negative item cannot be evaluated and are excluded.
    """
    neg_df = filter_negative(train_inner, threshold, user_thresholds)
    if len(neg_df) == 0:
        return neg_df, {}

    neg_val: Dict[int, int] = {}
    holdout_pairs = []

    for uid, grp in neg_df.groupby("userId"):
        items = grp["movieId"].tolist()
        if len(items) < 2:
            continue  # can't hold out and still train
        held_out = rng.choice(items)
        neg_val[uid] = held_out
        holdout_pairs.append((uid, held_out))

    if not holdout_pairs:
        return neg_df, {}

    excl = pd.DataFrame(holdout_pairs, columns=["userId", "movieId"])
    excl["_x"] = True
    merged = neg_df.merge(excl, on=["userId", "movieId"], how="left")
    neg_train = neg_df[merged["_x"].isna()].copy()
    return neg_train, neg_val


# evaluatio

def evaluate_arm_a(
    model: SVDBaseline,
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    all_items: set,
    k: int,
    n_candidates: int,
    max_users: Optional[int],
    seed: int,
) -> float:
    """Evaluate Arm A with standard nDCG@10 (positive leave-one-out)."""
    rng = random.Random(seed)
    user_seen = train_df.groupby("userId")["movieId"].apply(set).to_dict()

    subset = val_df.head(max_users) if max_users else val_df
    hits = []
    for _, row in subset.iterrows():
        uid = int(row["userId"])
        target = int(row["movieId"])
        seen = user_seen.get(uid, set())

        unseen = list(all_items - seen - {target})
        candidates = rng.sample(unseen, min(n_candidates, len(unseen)))
        if target not in candidates:
            candidates.append(target)

        ranked = [item for item, _ in model.rank_items_for_user(uid, candidates)]
        hits.append(ndcg_at_k(ranked, {target}, k))

    return float(np.mean(hits)) if hits else 0.0


def evaluate_arm_b(
    model: SVDBaseline,
    neg_val: Dict[int, int],
    all_items: set,
    neg_train: pd.DataFrame,
    k: int,
    n_candidates: int,
    max_users: Optional[int],
    seed: int,
) -> float:
    """Evaluate Arm B as a dislike detector: negative_detection_hit@10."""
    if not neg_val:
        return 0.0

    rng = random.Random(seed)
    user_seen = neg_train.groupby("userId")["movieId"].apply(set).to_dict()

    users = list(neg_val.keys())
    if max_users:
        users = users[:max_users]

    hits = []
    for uid in users:
        target_neg = neg_val[uid]
        seen = user_seen.get(uid, set())

        unseen_non_neg = list(all_items - seen - {target_neg})
        candidates = rng.sample(unseen_non_neg, min(n_candidates, len(unseen_non_neg)))
        if target_neg not in candidates:
            candidates.append(target_neg)

        ranked = [item for item, _ in model.rank_items_for_user(uid, candidates)]
        hits.append(negative_detection_hit_at_k(ranked, target_neg, k))

    return float(np.mean(hits)) if hits else 0.0


# Optuna
def make_objective(
    arm: str,
    train_inner: pd.DataFrame,
    val_df: pd.DataFrame,
    user_thresholds: pd.DataFrame,
    threshold,
    all_items: set,
    config: ExperimentConfig,
    max_users: Optional[int],
    seed: int,
):
    space = SEARCH_SPACE[arm]
    rng = random.Random(seed)

    if arm == "b":
        neg_train, neg_val = build_arm_b_neg_val(train_inner, threshold, user_thresholds, rng)
        neg_all_items = set(neg_train["movieId"].unique()) if len(neg_train) > 0 else all_items
    else:
        pos_train, std_val = build_arm_a_val(train_inner, val_df, threshold)

    def objective(trial):
        params = {}
        for name, choices in space.items():
            if name == "biased":
                params[name] = trial.suggest_categorical(name, choices)
            else:
                params[name] = trial.suggest_categorical(name, choices)

        model = SVDBaseline(
            n_factors=params["n_factors"],
            n_epochs=params["n_epochs"],
            lr_all=params["lr_all"],
            reg_all=params["reg_all"],
            biased=params["biased"],
            random_state=seed,
        )

        if arm == "a":
            if len(pos_train) < 10:
                return 0.0
            model.fit(pos_train)
            return evaluate_arm_a(
                model, std_val, pos_train, all_items,
                config.eval.k, config.eval.n_candidates, max_users, seed,
            )
        else:
            if len(neg_train) < 10:
                return 0.0
            model.fit(neg_train)
            return evaluate_arm_b(
                model, neg_val, neg_all_items, neg_train,
                config.eval.k, config.eval.n_candidates, max_users, seed,
            )

    return objective


def random_search_objective(arm, train_inner, val_df, user_thresholds, threshold,
                             all_items, config, max_users, seed, space):
    """Single random trial — returns (params, score)."""
    rng_params = random.Random(seed)
    params = {k: rng_params.choice(v) for k, v in space.items()}

    model = SVDBaseline(
        n_factors=params["n_factors"],
        n_epochs=params["n_epochs"],
        lr_all=params["lr_all"],
        reg_all=params["reg_all"],
        biased=params["biased"],
        random_state=seed,
    )

    rng = random.Random(seed)
    if arm == "a":
        pos_train = filter_positive(train_inner, threshold)
        if len(pos_train) < 10:
            return params, 0.0
        model.fit(pos_train)
        score = evaluate_arm_a(model, val_df, pos_train, all_items,
                                config.eval.k, config.eval.n_candidates, max_users, seed)
    else:
        neg_train, neg_val = build_arm_b_neg_val(train_inner, threshold, user_thresholds, rng)
        if len(neg_train) < 10:
            return params, 0.0
        model.fit(neg_train)
        neg_items = set(neg_train["movieId"].unique())
        score = evaluate_arm_b(model, neg_val, neg_items, neg_train,
                                config.eval.k, config.eval.n_candidates, max_users, seed)
    return params, score

#main
def threshold_label(arm: str, threshold) -> str:
    if arm == "a":
        return f"pos_ge_{threshold}"
    if isinstance(threshold, int):
        return f"neg_le_{threshold}"
    return f"neg_{threshold}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True, choices=["a", "b"],
                        help="a = Arm A positive preference model, b = Arm B dislike-risk detector")
    parser.add_argument("--threshold", required=True,
                        help="Arm A: integer (e.g. 4 means train on rating>=4). "
                             "Arm B: integer (e.g. 2 means train on rating<=2) or 'median'/'modus'.")
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--max_users", type=int, default=None,
                        help="Limit evaluation to first N users (for quick dev runs)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # parse threshold
    threshold = args.threshold
    if threshold not in ("median", "modus"):
        threshold = int(threshold)

    config = ExperimentConfig.from_yaml(args.config)
    set_global_seed(args.seed)

    label = threshold_label(args.arm, threshold)
    out_dir = Path(config.output_dir).parent / "tuning" / f"arm_{args.arm}" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    best_params_path = out_dir / "best_params.json"

    print(f"\nHyperparameter Tuning — SVD Arm {args.arm.upper()}")
    print(f"  Dataset:   {config.data.name}")
    print(f"  Arm:       {'Positive Preference' if args.arm == 'a' else 'Negative Dislike-Risk'}")
    print(f"  Threshold: {label}")
    print(f"  Trials:    {args.n_trials}")
    print(f"  Engine:    {'Optuna TPE' if HAS_OPTUNA else 'Random Search (install optuna for Bayesian)'}")
    print(f"  Output:    {best_params_path}")
    print()

    proc = config.data.processed_path
    train_inner    = load_parquet(proc + config.splits.train_inner_file)
    val_df         = load_parquet(proc + config.splits.val_file)
    user_thresholds = load_parquet(proc + config.splits.user_thresholds_file)
    train_full     = load_parquet(proc + config.splits.train_file)
    all_items      = set(train_full["movieId"].unique())

    space = SEARCH_SPACE[args.arm]

    if HAS_OPTUNA:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        )
        objective = make_objective(
            args.arm, train_inner, val_df, user_thresholds, threshold,
            all_items, config, args.max_users, args.seed,
        )
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_value = study.best_value

        # save Optuna study for later inspection
        import pickle
        with open(out_dir / "study.pkl", "wb") as f:
            pickle.dump(study, f)

    else:
        print(f"  Running {args.n_trials} random trials...")
        best_params, best_value = None, -1.0
        for trial_i in range(args.n_trials):
            trial_seed = args.seed + trial_i
            params, score = random_search_objective(
                args.arm, train_inner, val_df, user_thresholds, threshold,
                all_items, config, args.max_users, trial_seed, space,
            )
            if score > best_value:
                best_value = score
                best_params = params
            if (trial_i + 1) % 10 == 0:
                metric = "nDCG@10" if args.arm == "a" else "neg_hit@10"
                print(f"  [{trial_i+1}/{args.n_trials}] best {metric}={best_value:.4f}")

    metric_name = "nDCG@10 (val)" if args.arm == "a" else "negative_detection_hit@10 (neg-LOO val)"
    print(f"\nBest {metric_name}: {best_value:.4f}")
    print(f"Best params: {best_params}")

    result = {
        "arm": args.arm,
        "threshold": str(threshold),
        "threshold_label": label,
        "dataset": config.data.name,
        "best_value": best_value,
        "metric": metric_name,
        "n_trials": args.n_trials,
        "best_params": best_params,
        "search_space": {k: [str(v) for v in vals] for k, vals in space.items()},
        "note": (
            "Task is top-N ranking, not rating regression. "
            "Arm A optimized for positive ranking quality (nDCG@10). "
            "Arm B optimized for negative dislike detection (negative_hit@10)."
        ),
    }
    with open(best_params_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {best_params_path}")


if __name__ == "__main__":
    main()