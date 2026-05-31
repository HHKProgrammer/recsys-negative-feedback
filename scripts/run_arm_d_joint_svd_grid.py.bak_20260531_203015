"""
Arm D  Joint Positive-Negative SVD Grid

Arm D tests whether negative feedback is most effective when incorporated
directly into model training, rather than as a post-hoc penalty (post-hoc grid)
or a separate dislike detector (Arm B).

Training data transformation:
  rating >= pos_threshold  →  target = 1.0  (explicit like)
  rating <= neg_threshold  →  target = 0.0  (explicit dislike)
  middle ratings           →  dropped        (ambiguous signal)

SVD is then trained on this binary preference signal using rating_scale=(0, 1).
High predicted value = likely liked, low predicted value = likely disliked.
This gives the model a contrast signal: push liked items up, disliked items down.

Difference from Arm B:
  Arm B  trained only on negatives → dislike detector only
  Arm D  trained on both positives and negatives → joint preference model

Core research question answered here:
  Does negative feedback improve recommendations when learned directly in
  the latent factors, rather than applied as a post-hoc adjustment?

Main comparison: Arm A (positive-only) vs Arm D (joint pos-neg)
  Arm D > Arm A in nDCG@10:  training-time negatives improve ranking
  Arm D ≈ Arm A but lower sim_to_neg@10:  negatives improve avoidance, not accuracy
  Arm D < Arm A:  binary signal loses information from the full rating scale

Grid: 4 configurations (pos_threshold, neg_threshold)
  pos_ge_4 + neg_le_2  (main, clean: 4-5=like, 1-2=dislike, 3=dropped)
  pos_ge_5 + neg_le_1  (strict both ends)
  pos_ge_4 + neg_le_1  (strict negatives only)
  pos_ge_3 + neg_le_2  (broad positives)

Output
  outputs/<dataset>/grid_summary_arm_d.json
  outputs/<dataset>/models/arm_d/<label>/model.pkl  (for potential Arm E)

Usage
  python scripts/run_arm_d_joint_svd_grid.py --config configs/movielens_1m.yaml
  python scripts/run_arm_d_joint_svd_grid.py --config configs/movielens_10m.yaml --max_users 5000
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

# Grid: (pos_threshold, neg_threshold) pairs
# Each pair defines what counts as "positive" and "negative" for joint training.
ARM_D_GRID = [
    {"pos": 4, "neg": 2},   # main: 4-5=like, 1-2=dislike, 3=dropped
    {"pos": 5, "neg": 1},   # strict both ends
    {"pos": 4, "neg": 1},   # strict negatives only
    {"pos": 3, "neg": 2},   # broad positives
]


def arm_d_label(pos: int, neg: int) -> str:
    return f"pos_ge_{pos}_neg_le_{neg}"


def build_joint_train(train_df: pd.DataFrame, pos_threshold: int, neg_threshold: int) -> pd.DataFrame:
    """Transform ratings to binary targets for joint positive-negative SVD training.

    The rating column is repurposed as the binary target (1.0 or 0.0) so that
    SVDBaseline.fit() can train normally with rating_scale=(0, 1).
    Middle ratings between thresholds are excluded as ambiguous signal.
    """
    positives = train_df[train_df["rating"] >= pos_threshold][["userId", "movieId"]].copy()
    positives["rating"] = 1.0

    negatives = train_df[train_df["rating"] <= neg_threshold][["userId", "movieId"]].copy()
    negatives["rating"] = 0.0

    joint = pd.concat([positives, negatives], ignore_index=True)
    joint = joint.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return joint


def load_best_params(tuning_dir: Path, label: str, config: ExperimentConfig) -> dict:
    path = tuning_dir / "arm_d" / label / "best_params.json"
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
    neg_threshold: int,
    tuning_dir: Path,
    model_dir: Path,
    max_users: Optional[int],
    seed: int,
) -> Dict:
    set_global_seed(seed)

    label = arm_d_label(pos_threshold, neg_threshold)
    proc  = config.data.processed_path

    train_df = load_parquet(proc + config.splits.train_file)
    test_df  = load_parquet(proc + config.splits.test_file)

    joint_train = build_joint_train(train_df, pos_threshold, neg_threshold)
    n_pos = (joint_train["rating"] == 1.0).sum()
    n_neg = (joint_train["rating"] == 0.0).sum()
    pct_used = len(joint_train) / len(train_df) * 100
    print(f"    Joint train: {n_pos:,} positives + {n_neg:,} negatives "
          f"= {len(joint_train):,} rows ({pct_used:.1f}% of original)")

    if len(joint_train) < 100:
        raise ValueError(f"Too few joint interactions for {label}")

    params = load_best_params(tuning_dir, label, config)

    model = SVDBaseline(
        n_factors=int(params["n_factors"]),
        n_epochs=int(params["n_epochs"]),
        lr_all=float(params["lr_all"]),
        reg_all=float(params["reg_all"]),
        biased=bool(params.get("biased", True)),
        random_state=seed,
        rating_scale=(0, 1),   # binary targets: like=1.0, dislike=0.0
    )
    model.fit(joint_train)

    save_path = model_dir / label / "model.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"    Model saved → {save_path}")

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

    agg["pos_threshold"]  = pos_threshold
    agg["neg_threshold"]  = neg_threshold
    agg["n_positives"]    = int(n_pos)
    agg["n_negatives"]    = int(n_neg)
    agg["pct_train_used"] = round(pct_used, 2)
    agg["params"]         = params
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
    out_path = base / "grid_summary_arm_d.json"
    tuning   = base / "tuning"
    models   = base / "models" / "arm_d"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        with open(out_path) as f:
            summary = json.load(f)
        done = {e["exp_id"] for e in summary.get("experiments", [])}
    else:
        summary = {"experiments": [], "meta": {}}
        done = set()

    print(f"\nArm D — Joint Positive-Negative SVD Grid")
    print(f"  Dataset:      {config.data.name}")
    print(f"  Grid:         {[(g['pos'], g['neg']) for g in ARM_D_GRID]}")
    print(f"  Approach:     binary targets (1.0=like, 0.0=dislike, middle=dropped)")
    print(f"  rating_scale: (0, 1)")
    print(f"  References:   Cena et al. 2004, Paudel et al. 2016")
    print(f"  Output:       {out_path}")
    print()

    k = config.eval.k

    for i, cfg in enumerate(ARM_D_GRID, 1):
        pos, neg = cfg["pos"], cfg["neg"]
        label = arm_d_label(pos, neg)
        eid   = f"arm_d_{label}"

        if eid in done:
            print(f"  [{i}/{len(ARM_D_GRID)}] SKIP (done): {eid}")
            continue

        print(f"  [{i}/{len(ARM_D_GRID)}] {eid}  "
              f"(rating>={pos}→1.0, rating<={neg}→0.0, middle=dropped)")

        try:
            metrics = run_one(config, pos, neg, tuning, models, args.max_users, args.seed)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            continue

        print(f"    nDCG@{k}={metrics[f'ndcg@{k}']:.4f}  "
              f"HR@{k}={metrics[f'hit@{k}']:.4f}  "
              f"MRR={metrics['mrr']:.4f}  "
              f"sim_neg@{k}={metrics.get(f'sim_to_neg@{k}', 0):.4f}")

        summary["experiments"].append({
            "exp_id":        eid,
            "variant":       "arm_d",
            "pos_threshold": pos,
            "neg_threshold": neg,
            "label":         label,
            "metrics":       metrics,
        })
        done.add(eid)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

    summary["meta"] = {
        "approach":    "arm_d_joint_positive_negative_svd",
        "description": (
            "SVD trained on binary targets: rating>=pos_threshold → 1.0, "
            "rating<=neg_threshold → 0.0, middle ratings dropped. "
            "Arm D tests whether joint training on positive AND negative feedback "
            "improves top-N ranking quality compared to positive-only (Arm A)."
        ),
        "rating_scale": [0, 1],
        "references":   [
            "Cena Console Gena Goy Levi Tasso 2004 — explicit dislikes as constraints",
            "Paudel Bonta Bernstein Kuflik 2016 — known-dislike avoidance in top-N",
        ],
        "note": (
            "Core comparison: Arm A (positive-only) vs Arm D (joint pos-neg). "
            "Arm D > Arm A → training-time negative feedback helps. "
            "Arm D ≈ Arm A with lower sim_to_neg → negatives improve avoidance only."
        ),
        "completed_at": datetime.now().isoformat(),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results in {out_path}")


if __name__ == "__main__":
    main()