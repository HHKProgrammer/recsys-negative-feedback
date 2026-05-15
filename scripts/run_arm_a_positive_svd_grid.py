"""
Arm A Positive Preference SVD Grid

Arm A is the positive-preference model in the three-arm architecture:

  Arm A: Positive Preference Model trains SVD on ratings >= pos_threshold
  Arm B: Negative Dislike Detector  trains SVD on ratings <= neg_threshold
  Arm C: Hybrid combiner          norm(Arm A) - alpha * norm(Arm B)

Hypothesis
Training SVD exclusively on high-rated interactions removes the noisy
signal from low ratings that distorts the shared latent space.  The model
learns to encode what users actually enjoy, without being pulled toward
disliked content.  This is the positive-only training approach used in BPR
(Rendle et al. 2009, UAI) and NCF (He et al. 2017, WWW):

  Rendle, Freudenthaler, Gantner & Schmidt-Thieme (2009)
  "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI)
  https://arxiv.org/abs/1205.2618

  He, Liao, Zhang, Nie, Hu & Chua (2017)
  "Neural Collaborative Filtering" (WWW)
  https://dl.acm.org/doi/10.1145/3038912.3052569

Positive threshold configurations
  pos_ge_3:  rating >= 3  (broad positive, includes neutral ratings)
  pos_ge_4:  rating >= 4  (standard positive)
  pos_ge_5:  rating >= 5  (strict, 5-stars only)

If hyperparameter tuning output exists (run_hyperparameter_tuning.py --arm a),
best params are loaded automatically.  Config defaults are used otherwise.

Output
  outputs/<dataset>/grid_summary_arm_a.json
  outputs/<dataset>/models/arm_a/pos_ge_<t>/model.pkl   (loaded by Arm C)

Usage
  python scripts/run_arm_a_positive_svd_grid.py --config configs/movielens_1m.yaml
  python scripts/run_arm_a_positive_svd_grid.py --config configs/movielens_1m.yaml --max_users 500
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.ranking_metrics import evaluate_ranking
from src.models.svd_baseline import SVDBaseline
from src.utils.config import ExperimentConfig
from src.utils.io import load_parquet
from src.utils.seed import set_global_seed

POS_THRESHOLDS = [3, 4, 5]


def pos_label(t: int) -> str:
    return f"pos_ge_{t}"


def load_best_params(tuning_dir: Path, label: str, config: ExperimentConfig) -> dict:
    path = tuning_dir / "arm_a" / label / "best_params.json"
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
    pos_threshold: int,
    tuning_dir: Path,
    model_dir: Path,
    max_users: Optional[int],
    seed: int,
) -> Dict:
    set_global_seed(seed)

    label = pos_label(pos_threshold)
    proc  = config.data.processed_path

    train_df        = load_parquet(proc + config.splits.train_file)
    test_df         = load_parquet(proc + config.splits.test_file)

    pos_train = train_df[train_df["rating"] >= pos_threshold].copy()
    n_removed = len(train_df) - len(pos_train)
    pct_removed = n_removed / len(train_df) * 100
    print(f"    {len(pos_train):,} rows kept (removed {n_removed:,} / {pct_removed:.1f}%)")

    if len(pos_train) < 100:
        raise ValueError(f"Too few positive interactions for threshold {pos_threshold}")

    params = load_best_params(tuning_dir, label, config)

    model = SVDBaseline(
        n_factors=int(params["n_factors"]),
        n_epochs=int(params["n_epochs"]),
        lr_all=float(params["lr_all"]),
        reg_all=float(params["reg_all"]),
        biased=bool(params.get("biased", True)),
        random_state=seed,
    )
    model.fit(pos_train)

    save_path = model_dir / label / "model.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"    Model saved → {save_path}")

    all_items = set(train_df["movieId"].unique())
    dummy_neg = {int(uid): set() for uid in test_df["userId"].unique()}

    agg, _ = evaluate_ranking(
        model=model,
        test_df=test_df,
        train_df=train_df,          # full train for seen-item exclusion
        user_negative_items=dummy_neg,
        all_items=all_items,
        k=config.eval.k,
        n_candidates=config.eval.n_candidates,
        seed=config.eval.random_seed,
        max_users=max_users,
        similarity_fn=None,
    )

    agg["pos_threshold"]           = pos_threshold
    agg["n_interactions_removed"]  = n_removed
    agg["pct_train_removed"]       = round(pct_removed, 2)
    agg["params"]                  = params
    return agg


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",    required=True)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    config   = ExperimentConfig.from_yaml(args.config)
    base     = Path(config.output_dir).parent
    out_path = base / "grid_summary_arm_a.json"
    tuning   = base / "tuning"
    models   = base / "models" / "arm_a"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        with open(out_path) as f:
            summary = json.load(f)
        done = {e["exp_id"] for e in summary.get("experiments", [])}
    else:
        summary = {"experiments": [], "meta": {}}
        done = set()

    print(f"\nArm A — Positive Preference SVD Grid")
    print(f"  Dataset:    {config.data.name}")
    print(f"  Thresholds: pos >= {POS_THRESHOLDS}")
    print(f"  References: Rendle et al. 2009 (BPR), He et al. 2017 (NCF)")
    print(f"  Output:     {out_path}")
    print()

    for i, pt in enumerate(POS_THRESHOLDS, 1):
        eid = f"arm_a_{pos_label(pt)}"
        if eid in done:
            print(f"  [{i}/{len(POS_THRESHOLDS)}] SKIP (done): {eid}")
            continue

        print(f"  [{i}/{len(POS_THRESHOLDS)}] {eid}  (rating >= {pt})")
        metrics = run_one(config, pt, tuning, models, args.max_users, args.seed)

        k = config.eval.k
        print(f"    nDCG@{k}={metrics[f'ndcg@{k}']:.4f}  "
              f"HR@{k}={metrics[f'hit@{k}']:.4f}  "
              f"MRR={metrics['mrr']:.4f}")

        summary["experiments"].append({
            "exp_id":        eid,
            "variant":       "arm_a",
            "pos_threshold": pt,
            "metrics":       metrics,
        })
        done.add(eid)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

    summary["meta"] = {
        "approach":    "arm_a_positive_svd",
        "references":  [
            "Rendle Freudenthaler Gantner Schmidt-Thieme 2009 UAI — BPR positive-only training",
            "He Liao Zhang Nie Hu Chua 2017 WWW — NCF positive-only training",
        ],
        "description": "SVD trained on positive-rated interactions only (rating >= pos_threshold)",
        "completed_at": datetime.now().isoformat(),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results in {out_path}")


if __name__ == "__main__":
    main()