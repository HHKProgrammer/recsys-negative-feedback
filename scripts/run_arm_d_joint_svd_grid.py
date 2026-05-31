"""
Arm D — Joint Positive-Negative SVD Grid

Arm D tests whether negative feedback is most effective when incorporated
directly into model training, rather than as a post-hoc penalty or as a
separate dislike detector.

Training data transformation:
  rating >= pos_threshold  -> target = 1.0  (explicit like)
  rating <= neg_threshold  -> target = 0.0  (explicit dislike)
  middle ratings           -> dropped       (ambiguous signal)

This script supports:
  - old flat best_params.json files
  - new nested best_params.json files
  - rerunning one specific config with --only_pos / --only_neg
  - forcing recomputation with --force
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


ARM_D_GRID = [
    {"pos": 4, "neg": 2},   # main: 4-5=like, 1-2=dislike, 3=dropped
    {"pos": 5, "neg": 1},   # strict both ends
    {"pos": 4, "neg": 1},   # strict negatives only
    {"pos": 3, "neg": 2},   # broad positives
]


def arm_d_label(pos: int, neg: int) -> str:
    return f"pos_ge_{pos}_neg_le_{neg}"


def build_joint_train(
    train_df: pd.DataFrame,
    pos_threshold: int,
    neg_threshold: int,
) -> pd.DataFrame:
    """
    Transform ratings into binary targets for Arm D.

    Positive:
        rating >= pos_threshold -> 1.0

    Negative:
        rating <= neg_threshold -> 0.0

    Middle ratings are dropped because they are ambiguous.
    """
    positives = train_df[train_df["rating"] >= pos_threshold][["userId", "movieId"]].copy()
    positives["rating"] = 1.0

    negatives = train_df[train_df["rating"] <= neg_threshold][["userId", "movieId"]].copy()
    negatives["rating"] = 0.0

    joint = pd.concat([positives, negatives], ignore_index=True)
    joint = joint.sample(frac=1.0, random_state=42).reset_index(drop=True)

    return joint


def normalize_params(params: dict) -> dict:
    """
    Normalize parameter types before passing them into SVDBaseline.
    """
    required = ["n_factors", "n_epochs", "lr_all", "reg_all"]
    missing = [key for key in required if key not in params]

    if missing:
        raise ValueError(f"Missing required params: {missing}. Got keys: {list(params.keys())}")

    return {
        "n_factors": int(params["n_factors"]),
        "n_epochs": int(params["n_epochs"]),
        "lr_all": float(params["lr_all"]),
        "reg_all": float(params["reg_all"]),
        "biased": bool(params.get("biased", True)),
    }


def load_best_params(tuning_dir: Path, label: str, config: ExperimentConfig) -> dict:
    """
    Load best parameters for Arm D.

    Supports two formats:

    New format:
        {
          "best_value": ...,
          "metric": ...,
          "best_params": {
              "n_factors": ...,
              "n_epochs": ...,
              ...
          }
        }

    Old format:
        {
          "n_factors": ...,
          "n_epochs": ...,
          "lr_all": ...,
          "reg_all": ...,
          "biased": ...
        }
    """
    path = tuning_dir / "arm_d" / label / "best_params.json"

    if path.exists():
        with open(path) as f:
            result = json.load(f)

        # New nested format
        if "best_params" in result:
            params = normalize_params(result["best_params"])
            metric = result.get("metric", "metric")
            best_value = result.get("best_value", None)

            if best_value is not None:
                print(f"    Loaded tuned params ({metric}={best_value:.4f})")
            else:
                print(f"    Loaded tuned params ({metric})")

            return params

        # Old flat format
        if {"n_factors", "n_epochs", "lr_all", "reg_all"}.issubset(result.keys()):
            params = normalize_params(result)
            print(f"    Loaded tuned params from flat JSON: {path}")
            return params

        raise ValueError(
            f"Unsupported best_params.json format at {path}. "
            f"Keys found: {list(result.keys())}"
        )

    params = {
        "n_factors": int(config.model.n_factors),
        "n_epochs": int(config.model.n_epochs),
        "lr_all": float(config.model.lr_all),
        "reg_all": float(config.model.reg_all),
        "biased": True,
    }

    print(f"    No tuning found at {path}, using config defaults")
    return params


def run_one(
    config: ExperimentConfig,
    pos_threshold: int,
    neg_threshold: int,
    tuning_dir: Path,
    model_dir: Path,
    max_users: Optional[int],
    seed: int,
) -> Dict:
    set_global_seed(seed)

    label = arm_d_label(pos_threshold, neg_threshold)
    proc = config.data.processed_path

    train_df = load_parquet(proc + config.splits.train_file)
    test_df = load_parquet(proc + config.splits.test_file)

    joint_train = build_joint_train(train_df, pos_threshold, neg_threshold)

    n_pos = int((joint_train["rating"] == 1.0).sum())
    n_neg = int((joint_train["rating"] == 0.0).sum())
    pct_used = len(joint_train) / len(train_df) * 100

    print(
        f"    Joint train: {n_pos:,} positives + {n_neg:,} negatives "
        f"= {len(joint_train):,} rows ({pct_used:.1f}% of original)"
    )

    if len(joint_train) < 100:
        raise ValueError(f"Too few joint interactions for {label}")

    params = load_best_params(tuning_dir, label, config)

    model = SVDBaseline(
        n_factors=params["n_factors"],
        n_epochs=params["n_epochs"],
        lr_all=params["lr_all"],
        reg_all=params["reg_all"],
        biased=params.get("biased", True),
        random_state=seed,
        rating_scale=(0, 1),
    )

    model.fit(joint_train)

    save_path = model_dir / label / "model.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"    Model saved -> {save_path}")

    all_items = set(train_df["movieId"].unique())
    dummy_neg = {int(uid): set() for uid in test_df["userId"].unique()}

    agg, _ = evaluate_ranking(
        model=model,
        test_df=test_df,
        train_df=train_df,
        user_negative_items=dummy_neg,
        all_items=all_items,
        k=config.eval.k,
        n_candidates=config.eval.n_candidates,
        seed=config.eval.random_seed,
        max_users=max_users,
        similarity_fn=None,
    )

    agg["pos_threshold"] = int(pos_threshold)
    agg["neg_threshold"] = int(neg_threshold)
    agg["n_positives"] = int(n_pos)
    agg["n_negatives"] = int(n_neg)
    agg["pct_train_used"] = round(float(pct_used), 2)
    agg["params"] = params

    return agg


def remove_existing_experiment(summary: dict, exp_id: str) -> dict:
    """
    Remove an existing experiment from grid_summary when --force is used.
    """
    experiments = summary.get("experiments", [])
    summary["experiments"] = [
        exp for exp in experiments
        if exp.get("exp_id") != exp_id
    ]
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--only_pos",
        type=int,
        default=None,
        help="Run only one positive threshold, e.g. --only_pos 3",
    )
    parser.add_argument(
        "--only_neg",
        type=int,
        default=None,
        help="Run only one negative threshold, e.g. --only_neg 2",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if experiment already exists in grid_summary_arm_d.json",
    )

    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    base = Path(config.output_dir).parent
    out_path = base / "grid_summary_arm_d.json"
    tuning = base / "tuning"
    models = base / "models" / "arm_d"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        with open(out_path) as f:
            summary = json.load(f)
    else:
        summary = {"experiments": [], "meta": {}}

    done = {e["exp_id"] for e in summary.get("experiments", [])}

    grid = ARM_D_GRID
    if args.only_pos is not None or args.only_neg is not None:
        if args.only_pos is None or args.only_neg is None:
            raise ValueError("Use both --only_pos and --only_neg together.")
        grid = [
            g for g in ARM_D_GRID
            if g["pos"] == args.only_pos and g["neg"] == args.only_neg
        ]
        if not grid:
            raise ValueError(
                f"No Arm D config found for pos={args.only_pos}, neg={args.only_neg}"
            )

    print("\nArm D — Joint Positive-Negative SVD Grid")
    print(f"  Dataset:      {config.data.name}")
    print(f"  Grid:         {[(g['pos'], g['neg']) for g in grid]}")
    print("  Approach:     binary targets (1.0=like, 0.0=dislike, middle=dropped)")
    print("  rating_scale: (0, 1)")
    print(f"  Output:       {out_path}")
    print(f"  Force rerun:  {args.force}")
    print()

    k = config.eval.k

    for i, cfg in enumerate(grid, 1):
        pos, neg = cfg["pos"], cfg["neg"]
        label = arm_d_label(pos, neg)
        eid = f"arm_d_{label}"

        if eid in done and not args.force:
            print(f"  [{i}/{len(grid)}] SKIP (done): {eid}")
            continue

        if eid in done and args.force:
            print(f"  [{i}/{len(grid)}] FORCE rerun: {eid}")
            summary = remove_existing_experiment(summary, eid)
            done.discard(eid)

        print(
            f"  [{i}/{len(grid)}] {eid} "
            f"(rating>={pos}->1.0, rating<={neg}->0.0, middle=dropped)"
        )

        try:
            metrics = run_one(
                config=config,
                pos_threshold=pos,
                neg_threshold=neg,
                tuning_dir=tuning,
                model_dir=models,
                max_users=args.max_users,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"    ERROR: {exc}")
            continue

        print(
            f"    nDCG@{k}={metrics[f'ndcg@{k}']:.4f}  "
            f"HR@{k}={metrics[f'hit@{k}']:.4f}  "
            f"MRR={metrics['mrr']:.4f}  "
            f"sim_neg@{k}={metrics.get(f'sim_to_neg@{k}', 0):.4f}"
        )

        summary["experiments"].append({
            "exp_id": eid,
            "variant": "arm_d",
            "pos_threshold": pos,
            "neg_threshold": neg,
            "label": label,
            "metrics": metrics,
        })

        done.add(eid)

        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

    summary["meta"] = {
        "approach": "arm_d_joint_positive_negative_svd",
        "description": (
            "SVD trained on binary targets: rating>=pos_threshold -> 1.0, "
            "rating<=neg_threshold -> 0.0, middle ratings dropped. "
            "Arm D tests whether joint training on positive and negative feedback "
            "improves top-N ranking quality compared to positive-only Arm A."
        ),
        "rating_scale": [0, 1],
        "note": (
            "Core comparison: Arm A positive-only vs Arm D joint pos-neg. "
            "Arm D > Arm A means training-time negative feedback improves ranking. "
            "Arm D approximately equal to Arm A but lower sim_to_neg means better avoidance."
        ),
        "completed_at": datetime.now().isoformat(),
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results in {out_path}")


if __name__ == "__main__":
    main()
