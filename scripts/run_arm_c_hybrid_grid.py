"""
Arm C  Hybrid Recommender Grid

Arm C combines Arm A (positive preference) and Arm B (dislike-risk detector)
into a single ranking signal:

  final_score(u, i) = norm_pos(u, i) - alpha * norm_neg(u, i)

where norm_pos and norm_neg are min-max normalised per user per candidate list,
so that alpha is a true [0, 1] blending weight independent of the raw score
scale of the underlying models.

  Epsilon handling: if all scores for a user are identical (flat range),
  normalisation falls back to 0.5 for all items, making Arm C degrade
  gracefully to pure Arm A ranking for that user.

The grid explores:
  pos_threshold in [3, 4, 5]          which Arm A model to use
  neg_threshold in [1, 2, 3, median, modus]   which Arm B model to use
  alpha         in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

Total: 3 * 5 * 6 = 90 combinations per dataset.
Missing model files are skipped with a warning.

Theoretical framing
Combining explicit positive and negative preference models follows the
dislike-avoidance recommendation literature:

  Paudel, Bonta, Bernstein & Kuflik (2016)
  "Fewer flops at the top: Accuracy and Diversity in Recommender Systems"
  alpha-weighted blend of preference and dislike scores

  Sinha, Sodhani, Pineau & Hamilton (2019)
  "Negative interactions and the value of dispreference information"

Evaluation
Standard top-N ranking on the held-out test set (same protocol as baseline
and post-hoc variants):
  nDCG@k, HR@k, MRR, precision@k, recall@k
  sim_to_neg@k measures how close top-k recommendations are to the user's
                 disliked items in Arm A's latent space (lower = better)

Output
  outputs/<dataset>/grid_summary_arm_c.json

Usage
  python scripts/run_arm_c_hybrid_grid.py --config configs/movielens_1m.yaml
  python scripts/run_arm_c_hybrid_grid.py --config configs/movielens_1m.yaml --max_users 500
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

from src.data.threshold_utils import get_all_user_negative_items
from src.eval.metrics import (
    hit_at_k,
    ndcg_at_k,
    negative_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    sim_to_neg_at_k,
)
from src.models.svd_baseline import SVDBaseline
from src.utils.config import ExperimentConfig
from src.utils.io import load_parquet
from src.utils.seed import set_global_seed

POS_THRESHOLDS = [3, 4, 5]
NEG_CONFIGS = [
    {"type": "fixed",  "value": 1,        "label": "neg_le_1"},
    {"type": "fixed",  "value": 2,        "label": "neg_le_2"},
    {"type": "fixed",  "value": 3,        "label": "neg_le_3"},
    {"type": "median", "value": "median", "label": "neg_median"},
    {"type": "modus",  "value": "modus",  "label": "neg_modus"},
]
ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
EPS = 1e-8


def pos_label(t: int) -> str:
    return f"pos_ge_{t}"


def exp_id(pt: int, neg_label: str, alpha: float) -> str:
    return f"arm_c_{pos_label(pt)}_{neg_label}_alpha_{alpha:.2f}"


def minmax_norm(arr: np.ndarray, eps: float = EPS) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < eps:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def rank_arm_c(
    arm_a: SVDBaseline,
    arm_b: SVDBaseline,
    user_id: int,
    candidates: List[int],
    alpha: float,
) -> List[Tuple[int, float]]:
    pos_scores = arm_a.predict_batch(user_id, candidates).astype(np.float64)
    neg_scores = arm_b.predict_batch(user_id, candidates).astype(np.float64)

    norm_pos = minmax_norm(pos_scores)
    norm_neg = minmax_norm(neg_scores)

    final = norm_pos - alpha * norm_neg
    return sorted(zip(candidates, final.tolist()), key=lambda x: x[1], reverse=True)


def evaluate_arm_c(
    arm_a: SVDBaseline,
    arm_b: SVDBaseline,
    alpha: float,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    user_neg_items: Dict[int, set],
    all_items: Set[int],
    k: int,
    n_candidates: int,
    seed: int,
    max_users: Optional[int],
) -> Dict:
    rng = random.Random(seed)
    user_seen = train_df.groupby("userId")["movieId"].apply(set).to_dict()

    subset = test_df.head(max_users) if max_users else test_df

    # pre-generate candidate lists for reproducibility
    user_rows = []
    for _, row in subset.iterrows():
        uid  = int(row["userId"])
        item = int(row["movieId"])
        seen = user_seen.get(uid, set())
        pool = list(all_items - seen)
        cands = rng.sample(pool, min(n_candidates, len(pool)))
        if item not in cands:
            cands.append(item)
        user_rows.append((uid, item, cands))

    results = []
    for uid, test_item, cands in tqdm(user_rows, desc=f"alpha={alpha:.2f}", leave=False):
        ranked_pairs = rank_arm_c(arm_a, arm_b, uid, cands, alpha)
        ranked = [item for item, _ in ranked_pairs]

        relevant = {test_item}
        neg_set  = user_neg_items.get(uid, set())

        results.append({
            f"ndcg@{k}":       ndcg_at_k(ranked, relevant, k),
            f"hit@{k}":        hit_at_k(ranked, test_item, k),
            "mrr":             reciprocal_rank(ranked, test_item),
            f"precision@{k}":  precision_at_k(ranked, relevant, k),
            f"recall@{k}":     recall_at_k(ranked, relevant, k),
            f"negative@{k}":   negative_at_k(ranked, neg_set, k),
            f"sim_to_neg@{k}": sim_to_neg_at_k(ranked, neg_set, arm_a.get_similarity, k),
        })

    df  = pd.DataFrame(results)
    agg = {col: float(df[col].mean()) for col in df.columns}
    agg["n_users"] = len(df)
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
    out_path = base / "grid_summary_arm_c.json"
    models   = base / "models"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        with open(out_path) as f:
            summary = json.load(f)
        done = {e["exp_id"] for e in summary.get("experiments", [])}
    else:
        summary = {"experiments": [], "meta": {}}
        done = set()

    # pre-load data (shared across all combinations)
    proc            = config.data.processed_path
    train_df        = load_parquet(proc + config.splits.train_file)
    test_df         = load_parquet(proc + config.splits.test_file)
    user_thresholds = load_parquet(proc + config.splits.user_thresholds_file)
    all_items       = set(train_df["movieId"].unique())

    k            = config.eval.k
    n_candidates = config.eval.n_candidates
    seed_eval    = config.eval.random_seed

    # build grid: pos_threshold x neg_config x alpha
    grid = [
        (pt, neg_cfg, alpha)
        for pt in POS_THRESHOLDS
        for neg_cfg in NEG_CONFIGS
        for alpha in ALPHAS
    ]

    print(f"\nArm C — Hybrid Recommender Grid")
    print(f"  Dataset:    {config.data.name}")
    print(f"  Pos:        {POS_THRESHOLDS}")
    print(f"  Neg:        {[c['label'] for c in NEG_CONFIGS]}")
    print(f"  Alpha:      {ALPHAS}")
    print(f"  Total:      {len(grid)} combinations")
    print(f"  Output:     {out_path}")
    print()

    # cache loaded models to avoid re-loading the same pkl repeatedly
    _model_cache: Dict[str, Optional[SVDBaseline]] = {}

    def load_model(arm_subdir: str, label: str) -> Optional[SVDBaseline]:
        key = f"{arm_subdir}/{label}"
        if key not in _model_cache:
            path = models / arm_subdir / label / "model.pkl"
            if not path.exists():
                print(f"    WARNING: model not found at {path} — skip")
                _model_cache[key] = None
            else:
                _model_cache[key] = SVDBaseline.load(str(path))
        return _model_cache[key]

    for i, (pt, neg_cfg, alpha) in enumerate(grid, 1):
        eid = exp_id(pt, neg_cfg["label"], alpha)
        if eid in done:
            continue

        print(f"  [{i}/{len(grid)}] {eid}")

        arm_a = load_model("arm_a", pos_label(pt))
        arm_b = load_model("arm_b", neg_cfg["label"])
        if arm_a is None or arm_b is None:
            print(f"    SKIP (missing model)")
            continue

        # negatives for sim_to_neg metric  Arm B's threshold definition
        if neg_cfg["type"] == "fixed":
            user_neg_items = get_all_user_negative_items(
                train_df, "fixed", int(neg_cfg["value"]), user_thresholds
            )
        else:
            user_neg_items = get_all_user_negative_items(
                train_df, neg_cfg["type"], None, user_thresholds
            )

        set_global_seed(args.seed)
        metrics = evaluate_arm_c(
            arm_a=arm_a,
            arm_b=arm_b,
            alpha=alpha,
            test_df=test_df,
            train_df=train_df,
            user_neg_items=user_neg_items,
            all_items=all_items,
            k=k,
            n_candidates=n_candidates,
            seed=seed_eval,
            max_users=args.max_users,
        )

        print(f"    nDCG@{k}={metrics[f'ndcg@{k}']:.4f}  "
              f"HR@{k}={metrics[f'hit@{k}']:.4f}  "
              f"MRR={metrics['mrr']:.4f}  "
              f"sim_neg@{k}={metrics[f'sim_to_neg@{k}']:.4f}")

        summary["experiments"].append({
            "exp_id":           eid,
            "variant":          "arm_c",
            "pos_threshold":    pt,
            "neg_threshold_type":  neg_cfg["type"],
            "neg_threshold_value": str(neg_cfg["value"]),
            "neg_label":        neg_cfg["label"],
            "alpha":            alpha,
            "metrics":          metrics,
        })
        done.add(eid)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

    summary["meta"] = {
        "approach":    "arm_c_hybrid",
        "formula":     "final_score = minmax(pos_score) alpha * minmax(neg_score)",
        "references":  [
            "Paudel Bonta Bernstein Kuflik 2016 avoiding known dislikes in top-N",
            "Sinha Sodhani Pineau Hamilton 2019 value of dispreference information",
        ],
        "note":        "alpha=0 degrades to Arm A; higher alpha applies stronger dislike penalty",
        "completed_at": datetime.now().isoformat(),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results in {out_path}")


if __name__ == "__main__":
    main()