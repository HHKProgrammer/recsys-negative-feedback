"""
Hyperparameter Tuning for SVD Arms

Research context

This thesis frames recommendation as a TOP-N RANKING task, not a rating-regression task.
Ratings are raw material: high ratings → positive preference signal, low ratings → negative
preference signal. The goal is not to predict the exact star value (RMSE/MAE) but to rank
items so that relevant positives appear high and disliked items are avoided.

Because the three arms train on fundamentally different data distributions
(all interactions / positive-only / negative-only), each arm needs its own
hyperparameter search.

Arm A: Positive Preference Model
  Trains SVD on interactions with rating >= positive_threshold.
  Objective: maximize nDCG@10 on the validation set (positive leave-one-out).

Arm B: Negative Preference (Dislike-Risk) Model
  Trains SVD on interactions with rating <= negative_threshold.
  Objective: maximize negative_detection_hit@10 (leave-one-negative-out).

Arm D: Joint Positive-Negative SVD
  Trains SVD on binary targets: rating>=pos → 1.0, rating<=neg → 0.0.
  Objective: maximize nDCG@10 on the validation set.

Search space
  n_factors: Arm A/D [10,20,30,50,75,100,150,200], Arm B same (smaller sets)
  n_epochs:  [20, 30, 50, 75, 100]
  lr_all:    [0.001, 0.002, 0.005, 0.01, 0.02]
  reg_all:   Arm A/D [0.005,0.01,0.03,0.05,0.1,0.2], Arm B [0.03,0.05,0.1,0.2,0.5]
  biased:    [True, False]

Speed optimizations (do not change the scientific experiment)
  - Training data built ONCE per job, reused across all 200 trials
  - Validation candidate sets built ONCE, saved to parquet cache, reused across trials
  - evaluate_from_cache(): pure scoring using predict_batch()+ no resampling per trial
  - Random search: full resume from trials.csv  skips completed trials on restart
  - Optuna: resume-safe via SQLite storage (load_if_exists=True)
  - Per-trial ETA logging so you know exactly how long is left

Uses Optuna (Bayesian TPE) with n_trials=200. Falls back to random search if
Optuna is not available.

Output files
  outputs/<dataset>/tuning/arm_<a|b|d>/<label>/best_params.json
  outputs/<dataset>/tuning/arm_<a|b|d>/<label>/trials.csv
  outputs/<dataset>/tuning/arm_<a|b|d>/<label>/candidates_n<N>_<u>_s<seed>.parquet
  outputs/<dataset>/tuning/arm_<a|b|d>/<label>/study.db  (Optuna SQLite, if Optuna installed)

Usage
  python scripts/run_hyperparameter_tuning.py --config configs/movielens_1m.yaml --arm a --threshold 4
  python scripts/run_hyperparameter_tuning.py --config configs/movielens_1m.yaml --arm b --threshold 2
  python scripts/run_hyperparameter_tuning.py --config configs/movielens_1m.yaml --arm b --threshold median
  python scripts/run_hyperparameter_tuning.py --config configs/movielens_1m.yaml --arm d --threshold 4 --neg_threshold 2
  python scripts/run_hyperparameter_tuning.py --config configs/movielens_1m.yaml --arm a --threshold 4 --n_trials 5 --max_users 500
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics import ndcg_at_k, negative_detection_hit_at_k
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
        "n_factors": [10, 20, 30, 50, 75, 100, 150, 200],
        "n_epochs":  [20, 30, 50, 75, 100],
        "lr_all":    [0.001, 0.002, 0.005, 0.01, 0.02],
        "reg_all":   [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
        "biased":    [True, False],
    },
    "b": {
        "n_factors": [10, 20, 30, 50, 75, 100, 150, 200],
        "n_epochs":  [20, 30, 50, 75, 100],
        "lr_all":    [0.001, 0.002, 0.005, 0.01, 0.02],
        "reg_all":   [0.03, 0.05, 0.1, 0.2, 0.5],
        "biased":    [True, False],
    },
    "d": {
        "n_factors": [10, 20, 30, 50, 75, 100, 150, 200],
        "n_epochs":  [20, 30, 50, 75, 100],
        "lr_all":    [0.001, 0.002, 0.005, 0.01, 0.02],
        "reg_all":   [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
        "biased":    [True, False],
    },
}


# Data preparation
def filter_positive(train_df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    return train_df[train_df["rating"] >= threshold].copy()


def filter_negative(
    train_df: pd.DataFrame,
    threshold,
    user_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(threshold, int):
        return train_df[train_df["rating"] <= threshold].copy()
    col = "median_rating" if threshold == "median" else "modus_rating"
    merged = train_df.merge(user_thresholds[["userId", col]], on="userId", how="left")
    mask = (merged["rating"] < merged[col]).values
    return train_df[mask].copy()


def build_arm_d_train(
    train_df: pd.DataFrame,
    pos_threshold: int,
    neg_threshold: int,
) -> pd.DataFrame:
    positives = train_df[train_df["rating"] >= pos_threshold][["userId", "movieId"]].copy()
    positives["rating"] = 1.0
    negatives = train_df[train_df["rating"] <= neg_threshold][["userId", "movieId"]].copy()
    negatives["rating"] = 0.0
    joint = pd.concat([positives, negatives], ignore_index=True)
    return joint.sample(frac=1.0, random_state=42).reset_index(drop=True)


def build_lno_split(
    neg_df: pd.DataFrame,
    rng: random.Random,
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """Leave-one-negative-out: returns (neg_train_without_holdouts, {userId: held_out_item})."""
    neg_val: Dict[int, int] = {}
    holdout_pairs = []
    for uid, grp in neg_df.groupby("userId"):
        items = grp["movieId"].tolist()
        if len(items) < 2:
            continue
        held = rng.choice(items)
        neg_val[int(uid)] = int(held)
        holdout_pairs.append((int(uid), int(held)))
    if not holdout_pairs:
        return neg_df, {}
    excl = pd.DataFrame(holdout_pairs, columns=["userId", "movieId"])
    excl["_x"] = True
    merged = neg_df.merge(excl, on=["userId", "movieId"], how="left")
    neg_train = neg_df[merged["_x"].isna().values].copy()
    return neg_train, neg_val



def _build_cands_arm_a_d(
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    all_items: set,
    n_candidates: int,
    max_users: Optional[int],
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    user_seen = train_df.groupby("userId")["movieId"].apply(set).to_dict()
    subset = val_df.head(max_users) if max_users else val_df
    rows = []
    for _, row in subset.iterrows():
        uid = int(row["userId"])
        target = int(row["movieId"])
        seen = user_seen.get(uid, set())
        pool = list(all_items - seen - {target})
        cands = rng.sample(pool, min(n_candidates, len(pool)))
        cands.append(target)
        for item in cands:
            rows.append({"userId": uid, "itemId": item,
                         "targetItemId": target, "isTarget": int(item == target)})
    return pd.DataFrame(rows)


def _build_cands_arm_b(
    neg_val: Dict[int, int],
    neg_train: pd.DataFrame,
    all_items: set,
    n_candidates: int,
    max_users: Optional[int],
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    user_seen = neg_train.groupby("userId")["movieId"].apply(set).to_dict()
    users = list(neg_val.keys())
    if max_users:
        users = users[:max_users]
    rows = []
    for uid in users:
        target = neg_val[uid]
        seen = user_seen.get(uid, set())
        pool = list(all_items - seen - {target})
        cands = rng.sample(pool, min(n_candidates, len(pool)))
        cands.append(target)
        for item in cands:
            rows.append({"userId": uid, "itemId": item,
                         "targetItemId": target, "isTarget": int(item == target)})
    return pd.DataFrame(rows)


def load_or_build_candidate_dict(
    arm: str,
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    neg_val: Optional[Dict[int, int]],
    all_items: set,
    n_candidates: int,
    max_users: Optional[int],
    seed: int,
    cache_path: Path,
) -> Dict[int, Dict]:
    """Load or build fixed candidate sets.

    Returns {userId: {"items": [item_ids], "target": item_id}}.
    Candidates are deterministic (fixed seed) so all trials evaluate exactly
    the same users on the same pools only the trained model changes per trial.
    """
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"    Candidate cache loaded  ({df['userId'].nunique():,} users) ← {cache_path.name}")
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if arm == "b":
            df = _build_cands_arm_b(neg_val, train_df, all_items, n_candidates, max_users, seed)
        else:
            df = _build_cands_arm_a_d(val_df, train_df, all_items, n_candidates, max_users, seed)
        df.to_parquet(cache_path, index=False)
        print(f"    Candidate cache built   ({df['userId'].nunique():,} users) → {cache_path.name}")

    cdict: Dict[int, Dict] = {}
    for uid, grp in df.groupby("userId"):
        items = grp["itemId"].tolist()
        hits = grp[grp["isTarget"] == 1]
        if hits.empty:
            continue
        cdict[int(uid)] = {"items": items, "target": int(hits.iloc[0]["itemId"])}
    return cdict


def evaluate_from_cache(
    model: SVDBaseline,
    candidate_dict: Dict[int, Dict],
    k: int,
    metric: str,
) -> float:
    """Score pre-built candidate sets  no resampling, no user_seen rebuild.

    metric: "ndcg" for Arm A/D, "neg_hit" for Arm B.
    Uses vectorized predict_batch() (one matmul per user).
    """
    scores = []
    for uid, data in candidate_dict.items():
        items = data["items"]
        target = data["target"]
        preds = model.predict_batch(uid, items)
        order = np.argsort(-preds)
        ranked = [items[i] for i in order]
        if metric == "ndcg":
            scores.append(ndcg_at_k(ranked, {target}, k))
        else:
            scores.append(float(negative_detection_hit_at_k(ranked, target, k)))
    return float(np.mean(scores)) if scores else 0.0



def append_trial_row(path: Path, row: dict) -> None:
    """Append one trial row to trials.csv (one write per trial = crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_completed_trials(
    path: Path,
    score_col: str = "score",
) -> Tuple[set, float, List[float], Optional[dict]]:
    """Returns (completed_set, best_score, durations, best_params_dict)."""
    if not path.exists():
        return set(), -1.0, [], None
    df = pd.read_csv(path)
    if df.empty:
        return set(), -1.0, [], None
    completed = set(df["trial"].astype(int).tolist())
    best_score = float(df[score_col].max())
    durations = df["duration_s"].tolist() if "duration_s" in df.columns else []
    best_row = df.loc[df[score_col].idxmax()].to_dict()
    return completed, best_score, durations, best_row


def _cast_params(params_dict: dict, space: dict) -> dict:
    """Re-cast params read from CSV back to their correct Python types."""
    out = {}
    for k, choices in space.items():
        if k not in params_dict:
            continue
        v = params_dict[k]
        sample = choices[0]
        if isinstance(sample, bool):
            out[k] = str(v).lower() == "true"
        elif isinstance(sample, int):
            out[k] = int(float(v))
        elif isinstance(sample, float):
            out[k] = float(v)
        else:
            out[k] = v
    return out



def threshold_label(arm: str, threshold, neg_threshold=None) -> str:
    if arm == "a":
        return f"pos_ge_{threshold}"
    if arm == "d":
        return f"pos_ge_{threshold}_neg_le_{neg_threshold}"
    if isinstance(threshold, int):
        return f"neg_le_{threshold}"
    return f"neg_{threshold}"


# 

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True, choices=["a", "b", "d"],
                        help="a = Arm A positive-only, b = Arm B dislike detector, "
                             "d = Arm D joint positive-negative")
    parser.add_argument("--threshold", required=True,
                        help="Arm A/D: positive threshold (e.g. 4). "
                             "Arm B: integer (e.g. 2) or 'median'/'modus'.")
    parser.add_argument("--neg_threshold", default=None,
                        help="Arm D only: negative threshold integer (e.g. 2).")
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--max_users", type=int, default=None,
                        help="Limit evaluation to first N users (HPT only).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Re-run even if best_params.json already exists.")
    args = parser.parse_args()

    threshold = args.threshold
    if threshold not in ("median", "modus"):
        threshold = int(threshold)

    neg_threshold = None
    if args.arm == "d":
        if args.neg_threshold is None:
            parser.error("--neg_threshold is required for --arm d")
        neg_threshold = int(args.neg_threshold)

    config = ExperimentConfig.from_yaml(args.config)
    set_global_seed(args.seed)

    label = threshold_label(args.arm, threshold, neg_threshold)
    out_dir = Path(config.output_dir).parent / "tuning" / f"arm_{args.arm}" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    best_params_path = out_dir / "best_params.json"
    trials_path = out_dir / "trials.csv"

    if best_params_path.exists() and not args.overwrite:
        print(f"SKIP: {best_params_path} already exists (use --overwrite to re-tune)")
        sys.exit(0)

    arm_desc = {
        "a": "Positive Preference (positive-only SVD)",
        "b": "Negative Dislike-Risk Detector",
        "d": "Joint Positive-Negative SVD (binary targets)",
    }
    eval_metric = "neg_hit" if args.arm == "b" else "ndcg"
    metric_short = "neg_hit@10" if args.arm == "b" else "nDCG@10"
    rating_scale = (0, 1) if args.arm == "d" else (1, 5)
    space = SEARCH_SPACE[args.arm]

    print(f"\nHyperparameter Tuning — SVD Arm {args.arm.upper()}")
    print(f"  Dataset:  {config.data.name}")
    print(f"  Arm:      {arm_desc[args.arm]}")
    print(f"  Label:    {label}")
    print(f"  Trials:   {args.n_trials}")
    print(f"  Engine:   {'Optuna TPE' if HAS_OPTUNA else 'Random Search (install optuna for Bayesian)'}")
    print(f"  Output:   {best_params_path}")

    proc = config.data.processed_path
    train_inner = load_parquet(proc + config.splits.train_inner_file)
    val_df = load_parquet(proc + config.splits.val_file)
    user_thresholds = load_parquet(proc + config.splits.user_thresholds_file)
    train_full = load_parquet(proc + config.splits.train_file)
    all_items = set(train_full["movieId"].unique())

    k = config.eval.k
    n_candidates = config.eval.n_candidates
    seed_eval = config.eval.random_seed

    print("\n  Preparing training data (once)...")
    neg_val: Dict[int, int] = {}
    if args.arm == "a":
        train_df = filter_positive(train_inner, threshold)
        print(f"  Positive rows (rating>={threshold}): {len(train_df):,}")
    elif args.arm == "d":
        train_df = build_arm_d_train(train_inner, int(threshold), int(neg_threshold))
        n_pos = (train_df["rating"] == 1.0).sum()
        n_neg = (train_df["rating"] == 0.0).sum()
        print(f"  Joint train: {n_pos:,} positives + {n_neg:,} negatives = {len(train_df):,} rows")
    else:  # arm b
        neg_df = filter_negative(train_inner, threshold, user_thresholds)
        rng_lno = random.Random(args.seed)
        train_df, neg_val = build_lno_split(neg_df, rng_lno)
        print(f"  Negative rows: {len(train_df):,}  LNO hold-out users: {len(neg_val):,}")

    if len(train_df) < 10:
        print("ERROR: too few training rows — abort")
        sys.exit(1)

    max_u_tag = f"u{args.max_users}" if args.max_users else "uALL"
    cache_path = out_dir / f"candidates_n{n_candidates}_{max_u_tag}_s{seed_eval}.parquet"
    print("\n  Preparing candidate cache (once)...")
    candidate_dict = load_or_build_candidate_dict(
        arm=args.arm,
        val_df=val_df,
        train_df=train_df,
        neg_val=neg_val if args.arm == "b" else None,
        all_items=all_items,
        n_candidates=n_candidates,
        max_users=args.max_users,
        seed=seed_eval,
        cache_path=cache_path,
    )
    if not candidate_dict:
        print("ERROR: empty candidate dict — abort")
        sys.exit(1)
    print(f"    Evaluation users: {len(candidate_dict):,}")
    print()

    

    if HAS_OPTUNA:
        study_db = out_dir / "study.db"
        storage = f"sqlite:///{study_db}"

        already_done = 0
        try:
            tmp = optuna.load_study(study_name=label, storage=storage)
            already_done = sum(
                1 for t in tmp.trials if t.state == optuna.trial.TrialState.COMPLETE
            )
        except Exception:
            pass
        if already_done > 0:
            print(f"  Resuming Optuna study ({already_done}/{args.n_trials} trials done)")

        study = optuna.create_study(
            study_name=label,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        )

        trial_durations: List[float] = []

        def objective(trial):
            params = {name: trial.suggest_categorical(name, choices)
                      for name, choices in space.items()}
            model = SVDBaseline(
                n_factors=params["n_factors"],
                n_epochs=params["n_epochs"],
                lr_all=params["lr_all"],
                reg_all=params["reg_all"],
                biased=params["biased"],
                random_state=args.seed,
                rating_scale=rating_scale,
            )
            model.fit(train_df)
            return evaluate_from_cache(model, candidate_dict, k, eval_metric)

        def optuna_callback(
            study: "optuna.Study",
            trial: "optuna.trial.FrozenTrial",
        ) -> None:
            dur = trial.duration.total_seconds() if trial.duration else 0.0
            trial_durations.append(dur)
            n_done = sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            )
            remaining = args.n_trials - n_done
            avg_dur = float(np.mean(trial_durations))
            eta_h = remaining * avg_dur / 3600
            row = {"trial": n_done, **trial.params,
                   "score": trial.value, "duration_s": round(dur, 1)}
            append_trial_row(trials_path, row)
            print(
                f"  [{n_done}/{args.n_trials}] "
                f"{metric_short}={trial.value:.4f}  best={study.best_value:.4f}  "
                f"trial_time={dur:.1f}s  ETA={eta_h:.2f}h",
                flush=True,
            )

        remaining = max(0, args.n_trials - already_done)
        if remaining > 0:
            study.optimize(objective, n_trials=remaining, callbacks=[optuna_callback])

        best_params = study.best_params
        best_value = study.best_value

        import pickle
        with open(out_dir / "study.pkl", "wb") as f:
            pickle.dump(study, f)

    else:
        # Random search with resume support
        completed_set, best_value, past_durations, best_row = load_completed_trials(trials_path)
        best_params = _cast_params(best_row, space) if best_row else None
        durations = list(past_durations)

        if completed_set:
            print(f"  Resuming random search: {len(completed_set)} trials done, "
                  f"best={best_value:.4f}")
        else:
            print(f"  Running {args.n_trials} random trials...")

        n_total_done = len(completed_set)

        for trial_i in range(1, args.n_trials + 1):
            if trial_i in completed_set:
                continue

            trial_seed = args.seed + trial_i
            rng_params = random.Random(trial_seed)
            params = {k_: rng_params.choice(v_) for k_, v_ in space.items()}

            t0 = time.time()
            model = SVDBaseline(
                n_factors=params["n_factors"],
                n_epochs=params["n_epochs"],
                lr_all=params["lr_all"],
                reg_all=params["reg_all"],
                biased=params["biased"],
                random_state=trial_seed,
                rating_scale=rating_scale,
            )
            model.fit(train_df)
            score = evaluate_from_cache(model, candidate_dict, k, eval_metric)
            duration = time.time() - t0
            durations.append(duration)
            n_total_done += 1

            row = {"trial": trial_i, **params,
                   "score": score, "duration_s": round(duration, 1)}
            append_trial_row(trials_path, row)

            if score > best_value:
                best_value = score
                best_params = params

            remaining = args.n_trials - n_total_done
            avg_dur = float(np.mean(durations))
            eta_h = remaining * avg_dur / 3600
            print(
                f"  [{n_total_done}/{args.n_trials}] "
                f"{metric_short}={score:.4f}  best={best_value:.4f}  "
                f"trial_time={duration:.1f}s  avg={avg_dur:.1f}s  ETA={eta_h:.2f}h",
                flush=True,
            )

    metric_full = {
        "a": "nDCG@10 (val)",
        "b": "negative_detection_hit@10 (neg-LOO val)",
        "d": "nDCG@10 (val, binary-target model)",
    }[args.arm]

    print(f"\n  Best {metric_full}: {best_value:.4f}")
    print(f"  Best params: {best_params}")

    result = {
        "arm":             args.arm,
        "threshold":       str(threshold),
        "neg_threshold":   str(neg_threshold) if neg_threshold is not None else None,
        "threshold_label": label,
        "dataset":         config.data.name,
        "best_value":      best_value,
        "metric":          metric_full,
        "n_trials":        args.n_trials,
        "best_params":     best_params,
        "search_space":    {k_: [str(v_) for v_ in vals] for k_, vals in space.items()},
        "note": (
            "Task is top-N ranking, not rating regression. "
            "Arm A/D optimized for positive ranking quality (nDCG@10). "
            "Arm B optimized for negative dislike detection (negative_hit@10). "
            "Arm D uses binary targets (1.0=like, 0.0=dislike) with rating_scale=(0,1)."
        ),
    }
    with open(best_params_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Saved → {best_params_path}")


if __name__ == "__main__":
    main()