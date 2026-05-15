"""
Arm B  Negative Dislike-Risk Detector SVD Grid

Arm B is the dislike-risk detector in the three-arm architecture:

  Arm A: Positive Preference Model   trains SVD on ratings >= pos_threshold
  Arm B: Negative Dislike Detector   trains SVD on ratings <= neg_threshold
  Arm C: Hybrid combiner             norm(Arm A) - alpha * norm(Arm B)

Arm B is NOT a recommender.  It is trained to recognize what users dislike.
A high rank in Arm B output means "this item looks like something the user
dislikes", which is used by Arm C to penalise those items.

This framing is supported by the literature on negative preference modelling:

  Cena, Console, Gena, Goy, Levi & Tasso (2004)
  "Integrating and adapting knowledge for personalisation"
explicit dislikes as constraints on recommendations

  Paudel, Bonta, Bernstein & Kuflik (2016)
  "Fewer flops at the top: Accuracy and Diversity in Recommender Systems"
  filtering via known-negative avoidance

Evaluation leave-one-negative-out (LNO)
For each user with >= 2 negative interactions in training, one negative item
is held out.  Arm B is evaluated by whether it places the held-out disliked
item in its top-k output when competing against n_candidates neutral items.
This inverts the standard ranking assumption: a HIGH rank for a disliked item
means Arm B correctly detected it.

Metrics
  neg_detection_hit@k    1 if the held-out negative is in top-k, else 0
  neg_detection_ndcg@k   nDCG where the held-out negative is "relevant"
  mean_dislike_rank      mean 1-based rank of the held-out negative (lower = better)

Negative threshold configurations
  neg_le_1:    rating <= 1  (strictly bad)
  neg_le_2:    rating <= 2  (bad + mediocre)
  neg_le_3:    rating <= 3  (anything below average)
  neg_median:  rating <  user_median  (adaptive, strict less-than)
  neg_modus:   rating <  user_mode    (adaptive, strict less-than)

Output
  outputs/<dataset>/grid_summary_arm_b.json
  outputs/<dataset>/models/arm_b/<neg_label>/model.pkl   (loaded by Arm C)

Usage
  python scripts/run_arm_b_negative_svd_grid.py --config configs/movielens_1m.yaml
  python scripts/run_arm_b_negative_svd_grid.py --config configs/movielens_1m.yaml --max_users 500
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics import (
    mean_dislike_rank,
    negative_detection_hit_at_k,
    negative_detection_ndcg_at_k,
)
from src.models.svd_baseline import SVDBaseline
from src.utils.config import ExperimentConfig
from src.utils.io import load_parquet
from src.utils.seed import set_global_seed

# Five negative threshold configurations (consistent with post-hoc grid)
NEG_CONFIGS = [
    {"type": "fixed",  "value": 1,        "label": "neg_le_1"},
    {"type": "fixed",  "value": 2,        "label": "neg_le_2"},
    {"type": "fixed",  "value": 3,        "label": "neg_le_3"},
    {"type": "median", "value": "median", "label": "neg_median"},
    {"type": "modus",  "value": "modus",  "label": "neg_modus"},
]


def filter_negative(
    train_df: pd.DataFrame,
    threshold_type: str,
    threshold_value,
    user_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Return only rows whose rating qualifies as negative for that user."""
    if threshold_type == "fixed":
        return train_df[train_df["rating"] <= threshold_value].copy()
    col = "median_rating" if threshold_type == "median" else "modus_rating"
    merged = train_df.merge(user_thresholds[["userId", col]], on="userId", how="left")
    mask = merged["rating"] < merged[col]      # strict less-than (adaptive)
    return train_df[mask].copy()


def build_lno_split(
    neg_df: pd.DataFrame,
    rng: random.Random,
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """Leave-one-negative-out split.

    Returns (neg_train_without_holdouts, {userId: held_out_movieId}).
    Users with only 1 negative interaction are excluded (can't train and evaluate).
    """
    neg_val: Dict[int, int] = {}
    holdout_pairs: List[Tuple[int, int]] = []

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
    neg_train = neg_df[merged["_x"].isna()].copy()
    return neg_train, neg_val


def evaluate_detection(
    model: SVDBaseline,
    neg_val: Dict[int, int],
    neg_train: pd.DataFrame,
    all_items: Set[int],
    k: int,
    n_candidates: int,
    seed: int,
    max_users: Optional[int],
) -> Dict:
    """Evaluate Arm B as a dislike-risk detector (leave-one-negative-out).

    Candidate pool per user: n_candidates neutral items (not in the user's
    negative training set) + the held-out negative target.
    A good detector ranks the held-out target in its top-k.
    """
    if not neg_val:
        return {f"neg_detection_hit@{k}": 0.0,
                f"neg_detection_ndcg@{k}": 0.0,
                "mean_dislike_rank": float(n_candidates + 1),
                "n_users": 0}

    rng = random.Random(seed)
    user_neg_seen = neg_train.groupby("userId")["movieId"].apply(set).to_dict()

    users = list(neg_val.keys())
    if max_users:
        users = users[:max_users]

    results = []
    for uid in tqdm(users, desc="Arm B LNO eval", leave=False):
        target = neg_val[uid]
        neg_seen = user_neg_seen.get(uid, set())

        # neutral candidates: exclude items already in neg_train for this user
        pool = list(all_items - neg_seen - {target})
        candidates = rng.sample(pool, min(n_candidates, len(pool)))
        candidates.append(target)

        ranked = [item for item, _ in model.rank_items_for_user(uid, candidates)]

        results.append({
            f"neg_detection_hit@{k}":  negative_detection_hit_at_k(ranked, target, k),
            f"neg_detection_ndcg@{k}": negative_detection_ndcg_at_k(ranked, target, k),
            "mean_dislike_rank":        mean_dislike_rank(ranked, {target}),
        })

    df = pd.DataFrame(results)
    agg = {col: float(df[col].mean()) for col in df.columns}
    agg["n_users"] = len(df)
    return agg


def load_best_params(tuning_dir: Path, label: str, config: ExperimentConfig) -> dict:
    path = tuning_dir / "arm_b" / label / "best_params.json"
    if path.exists():
        with open(path) as f:
            result = json.load(f)
        params = result["best_params"]
        print(f"    Loaded tuned params ({result['metric']}={result['best_value']:.4f})")
        return params
    params = {
        "n_factors": config.model.n_factors,
        "n_epochs":  config.model.n_epochs,
        "lr_all":    config.model.lr_all,
        "reg_all":   config.model.reg_all,
        "biased":    True,
    }
    print(f"    No tuning found at {path}, using config defaults")
    return params


def run_one(
    config: ExperimentConfig,
    neg_cfg: dict,
    tuning_dir: Path,
    model_dir: Path,
    max_users: Optional[int],
    seed: int,
) -> Dict:
    set_global_seed(seed)
    rng = random.Random(seed)

    label  = neg_cfg["label"]
    t_type = neg_cfg["type"]
    t_val  = neg_cfg["value"]
    proc   = config.data.processed_path

    train_df        = load_parquet(proc + config.splits.train_file)
    user_thresholds = load_parquet(proc + config.splits.user_thresholds_file)

    neg_df = filter_negative(train_df, t_type, t_val, user_thresholds)
    pct_neg = len(neg_df) / len(train_df) * 100
    print(f"    {len(neg_df):,} negative interactions ({pct_neg:.1f}% of training data)")

    if len(neg_df) < 100:
        raise ValueError(f"Too few negative interactions for config '{label}'")

    neg_train, neg_val = build_lno_split(neg_df, rng)
    print(f"    LNO split: {len(neg_val):,} users have a held-out negative")

    if len(neg_train) < 50:
        raise ValueError(f"Too few training rows after LNO split for '{label}'")

    params = load_best_params(tuning_dir, label, config)

    model = SVDBaseline(
        n_factors=int(params["n_factors"]),
        n_epochs=int(params["n_epochs"]),
        lr_all=float(params["lr_all"]),
        reg_all=float(params["reg_all"]),
        biased=bool(params.get("biased", True)),
        random_state=seed,
    )
    model.fit(neg_train)

    save_path = model_dir / label / "model.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"    Model saved → {save_path}")

    # detection evaluatio all items as candidate universe
    all_items = set(train_df["movieId"].unique())
    det = evaluate_detection(
        model=model,
        neg_val=neg_val,
        neg_train=neg_train,
        all_items=all_items,
        k=config.eval.k,
        n_candidates=config.eval.n_candidates,
        seed=config.eval.random_seed,
        max_users=max_users,
    )

    det["n_neg_interactions"] = len(neg_df)
    det["pct_neg_of_train"]   = round(pct_neg, 2)
    det["params"]             = params
    return det


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",    required=True)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    config   = ExperimentConfig.from_yaml(args.config)
    base     = Path(config.output_dir).parent
    out_path = base / "grid_summary_arm_b.json"
    tuning   = base / "tuning"
    models   = base / "models" / "arm_b"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        with open(out_path) as f:
            summary = json.load(f)
        done = {e["exp_id"] for e in summary.get("experiments", [])}
    else:
        summary = {"experiments": [], "meta": {}}
        done = set()

    print(f"\nArm B — Negative Dislike-Risk Detector SVD Grid")
    print(f"  Dataset:    {config.data.name}")
    print(f"  Thresholds: {[c['label'] for c in NEG_CONFIGS]}")
    print(f"  Evaluation: leave-one-negative-out (LNO)")
    print(f"  Output:     {out_path}")
    print()

    for i, neg_cfg in enumerate(NEG_CONFIGS, 1):
        eid = f"arm_b_{neg_cfg['label']}"
        if eid in done:
            print(f"  [{i}/{len(NEG_CONFIGS)}] SKIP (done): {eid}")
            continue

        print(f"  [{i}/{len(NEG_CONFIGS)}] {eid}")
        metrics = run_one(config, neg_cfg, tuning, models, args.max_users, args.seed)

        k = config.eval.k
        print(f"    neg_hit@{k}={metrics.get(f'neg_detection_hit@{k}', 0):.4f}  "
              f"neg_nDCG@{k}={metrics.get(f'neg_detection_ndcg@{k}', 0):.4f}  "
              f"mean_rank={metrics.get('mean_dislike_rank', 0):.1f}")

        summary["experiments"].append({
            "exp_id":           eid,
            "variant":          "arm_b",
            "neg_threshold_type":  neg_cfg["type"],
            "neg_threshold_value": str(neg_cfg["value"]),
            "neg_label":        neg_cfg["label"],
            "metrics":          metrics,
        })
        done.add(eid)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

    summary["meta"] = {
        "approach":    "arm_b_negative_svd_detector",
        "references":  [
            "Cena Console Gena Goy Levi Tasso 2004  negative constraints on recommendations",
            "Paudel Bonta Bernstein Kuflik 2016 avoiding known dislikes in top-N",
        ],
        "evaluation":  "leave-one-negative-out: held-out negative item must appear in model top-k",
        "note":        "High rank for a disliked item = good detection (inverted semantics vs Arm A)",
        "completed_at": datetime.now().isoformat(),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results in {out_path}")


if __name__ == "__main__":
    main()