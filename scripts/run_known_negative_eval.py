"""
Known-Negative Candidate Injection Evaluation

Standard evaluation (as in main grid) samples 500 UNSEEN items as candidates.
Because negatives (low-rated items) are already seen, they are structurally excluded
from the candidate pool  so negative@10 is always 0 and tells us nothing.

This script implements a different evaluation protocol from the literature:
we deliberately INJECT each user's known-disliked items into the candidate pool.

Theoretical basis

Krichene & Rendle (2020) "On Sampled Metrics for Item Recommendation" (KDD 2020)
https://dl.acm.org/doi/10.1145/3383313.3412259

  They show that the content of the candidate set fundamentally changes what a
  ranking metric measures. Our injection approach extends their framework:
  instead of sampling uniformly from unseen items, we deliberately seed the pool
  with items the user has explicitly disliked, creating a harder, targeted test.

Research question answered here

"If a model knows which items a user dislikes, can it actively keep those items
out of the top 10 when they appear in the candidate pool?"

  - Baseline (pure SVD):        no dislike signal → dislikes land in top 10 by chance
  - Filter variant:             explicitly removes negatives → negative@10 = 0 by design
  - Rerank / Weighted variants: push negatives down → lower negative@10 and lower NDCG cost

This is a STRICTER test than the main grid. Results here are NOT directly
comparable to the main grid numbers  they use a different candidate set.

Speed design

  - SVD trained ONCE, shared across all 36 grid configurations
    (all variants use the same underlying model weights)
  - User evaluation parallelised with ThreadPoolExecutor
    (numpy predict_batch releases the GIL → true thread parallelism)
  - Candidate set: N_NEUTRAL + N_INJECTED + 1 per user, not all items
    (sampled evaluation, Krichene & Rendle 2020)

Output

  outputs/<dataset>/grid_summary_known_neg_eval.json — full experiment summary

Usage

  python scripts/run_known_negative_eval.py --config configs/movielens_1m.yaml
  python scripts/run_known_negative_eval.py --config configs/movielens_10m.yaml --n_workers 12
  python scripts/run_known_negative_eval.py --config configs/movielens_20m.yaml --n_workers 12
  python scripts/run_known_negative_eval.py --config configs/spotify.yaml
"""

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.threshold_utils import (
    get_all_user_negative_items,
    get_all_user_negative_items_with_ratings,
)
from src.eval.metrics import (
    hit_at_k,
    ndcg_at_k,
    negative_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    sim_to_neg_at_k,
)
from src.models.negative_variants import FilterNegatives, RerankPenalty, WeightedPenalty
from src.models.svd_baseline import SVDBaseline
from src.utils.config import ExperimentConfig
from src.utils.io import load_parquet
from src.utils.seed import set_global_seed


# candidate pool per user: N_INJECTED known dislikes + N_NEUTRAL unseen + 1 test item
N_NEUTRAL = 450
N_INJECTED = 50


def build_injected_candidates(
    user_id: int,
    test_item: int,
    seen_items: Set[int],
    negative_items: Set[int],
    all_items: Set[int],
    rng: random.Random,
    n_neutral: int = N_NEUTRAL,
    n_inject: int = N_INJECTED,
) -> List[int]:
    """Build a candidate pool with injected known dislikes (harder evaluation)."""
    neg_list = list(negative_items - {test_item})
    injected = rng.sample(neg_list, min(n_inject, len(neg_list)))

    neutral_pool = list(all_items - seen_items - negative_items - {test_item})
    neutral = rng.sample(neutral_pool, min(n_neutral, len(neutral_pool)))

    candidates = injected + neutral
    if test_item not in candidates:
        candidates.append(test_item)
    return candidates


def evaluate_one_user(
    user_id: int,
    model,
    is_baseline: bool,
    candidates: List[int],
    test_item: int,
    neg_set: Set[int],
    k: int,
    similarity_fn,
) -> Dict:
    if is_baseline:
        ranked = model.rank_items_for_user(user_id, candidates)
    else:
        ranked = model.rank_items_for_user(user_id, candidates, neg_set)

    ranked_items = [item for item, _ in ranked]
    return {
        "user_id": user_id,
        f"precision@{k}":  precision_at_k(ranked_items, {test_item}, k),
        f"recall@{k}":     recall_at_k(ranked_items, {test_item}, k),
        f"ndcg@{k}":       ndcg_at_k(ranked_items, {test_item}, k),
        f"negative@{k}":   negative_at_k(ranked_items, neg_set, k),
        f"hit@{k}":        hit_at_k(ranked_items, test_item, k),
        "mrr":             reciprocal_rank(ranked_items, test_item),
        f"sim_to_neg@{k}": sim_to_neg_at_k(ranked_items, neg_set, similarity_fn, k),
    }


def run_one_config(
    config: ExperimentConfig,
    variant: str,
    baseline: SVDBaseline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_thresholds: pd.DataFrame,
    user_seen: Dict[int, Set[int]],
    all_items: Set[int],
    threshold_type=None,
    fixed_threshold=None,
    alpha=None,
    max_users: Optional[int] = None,
    seed: int = 42,
    n_workers: int = 1,
) -> Dict:
    """Evaluate one grid configuration, reusing the pre-trained baseline SVD."""
    is_baseline = (variant == "baseline")
    if is_baseline:
        model = baseline
        user_neg: Dict = {uid: set() for uid in test_df["userId"].unique()}
    elif variant == "filter":
        model = FilterNegatives(baseline)
        user_neg = get_all_user_negative_items(
            train_df, threshold_type, fixed_threshold, user_thresholds)
    elif variant == "rerank":
        model = RerankPenalty(baseline, alpha=alpha)
        user_neg = get_all_user_negative_items(
            train_df, threshold_type, fixed_threshold, user_thresholds)
    elif variant == "weighted":
        model = WeightedPenalty(baseline, alpha=alpha)
        user_neg = get_all_user_negative_items_with_ratings(
            train_df, threshold_type, fixed_threshold, user_thresholds)
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    k = config.eval.k
    subset = test_df.head(max_users) if max_users else test_df
    sim_fn = baseline.get_similarity

    # Build candidate lists before threading so rng order is deterministic
    user_rows = []
    for _, row in subset.iterrows():
        user_id  = int(row["userId"])
        test_item = int(row["movieId"])
        rng = random.Random(seed ^ user_id)   # deterministic per-user, independent of eval order
        seen    = user_seen.get(user_id, set())
        neg_set = set(user_neg.get(user_id, set()))

        if not neg_set:
            unseen = list(all_items - seen)
            cands  = rng.sample(unseen, min(N_NEUTRAL + N_INJECTED, len(unseen)))
            if test_item not in cands:
                cands.append(test_item)
        else:
            cands = build_injected_candidates(
                user_id, test_item, seen, neg_set, all_items, rng)

        user_rows.append((user_id, test_item, cands, neg_set))

    def _eval_one(args):
        uid, ti, cands, neg = args
        return evaluate_one_user(uid, model, is_baseline, cands, ti, neg, k, sim_fn)

    per_user: List[Dict] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_eval_one, row) for row in user_rows]
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc=f"{variant}", leave=False):
            per_user.append(fut.result())

    df = pd.DataFrame(per_user)
    metric_cols = [c for c in df.columns if c != "user_id"]
    agg = {col: float(df[col].mean()) for col in metric_cols}
    agg["n_users"] = len(df)
    return agg


# same grid as main experiments
GRID = [
    dict(variant="baseline"),
    *[dict(variant="filter", threshold_type="fixed", fixed_threshold=t) for t in [1, 2, 3]],
    dict(variant="filter", threshold_type="median"),
    dict(variant="filter", threshold_type="modus"),
    *[dict(variant="rerank", threshold_type="fixed", fixed_threshold=t, alpha=a)
      for t in [1, 2, 3] for a in [0.1, 0.3, 1.0]],
    *[dict(variant="rerank", threshold_type=tt, alpha=a)
      for tt in ["median", "modus"] for a in [0.1, 0.3, 1.0]],
    *[dict(variant="weighted", threshold_type="fixed", fixed_threshold=t, alpha=a)
      for t in [1, 2, 3] for a in [0.1, 0.3, 1.0]],
    *[dict(variant="weighted", threshold_type=tt, alpha=a)
      for tt in ["median", "modus"] for a in [0.1, 0.3, 1.0]],
]


def exp_id(cfg: dict) -> str:
    parts = [cfg["variant"]]
    tt = cfg.get("threshold_type")
    if tt:
        parts.append(tt)
        ft = cfg.get("fixed_threshold")
        if ft is not None:
            parts.append(str(ft))
    a = cfg.get("alpha")
    if a is not None:
        parts.append(f"a{a}")
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",    required=True)
    parser.add_argument("--max_users", type=int,  default=None)
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument("--n_workers", type=int,  default=None,
                        help="Threads for user eval (default: CPU count). "
                             "ThreadPoolExecutor — numpy releases GIL during matmul.")
    args = parser.parse_args()

    n_workers = args.n_workers or (os.cpu_count() or 4)
    config = ExperimentConfig.from_yaml(args.config)
    set_global_seed(args.seed)

    output_path = Path(config.output_dir).parent / "grid_summary_known_neg_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with open(output_path) as f:
            summary = json.load(f)
        done = {e["exp_id"] for e in summary.get("experiments", [])}
    else:
        summary = {"experiments": [], "meta": {}}
        done = set()

    # Load data once — shared across all 36 grid configs
    print(f"\nKnown-Negative Candidate Injection Evaluation")
    print(f"  config:    {args.config}")
    print(f"  output:    {output_path}")
    print(f"  protocol:  {N_INJECTED} known-disliked items injected per user")
    print(f"             + {N_NEUTRAL} neutral unseen items + 1 true test item")
    print(f"  workers:   {n_workers} threads (ThreadPoolExecutor)")
    print(f"  paper:     Krichene & Rendle 2020 (KDD) — sampled evaluation")
    print()

    proc = config.data.processed_path
    print("Loading data...")
    train_df        = load_parquet(proc + config.splits.train_file)
    test_df         = load_parquet(proc + config.splits.test_file)
    user_thresholds = load_parquet(proc + config.splits.user_thresholds_file)
    user_seen       = train_df.groupby("userId")["movieId"].apply(set).to_dict()
    all_items       = set(train_df["movieId"].unique())

    # Train baseline SVD ONCE — all 36 configs share the same weights
    print("Training baseline SVD (once, shared across all experiments)...")
    baseline = SVDBaseline(
        n_factors=config.model.n_factors,
        n_epochs=config.model.n_epochs,
        lr_all=config.model.lr_all,
        reg_all=config.model.reg_all,
        random_state=args.seed,
    )
    baseline.fit(train_df)
    print(f"  Done. {len(all_items):,} items, {len(user_seen):,} users")
    print()

    total = len(GRID)
    for i, cfg in enumerate(GRID, 1):
        eid = exp_id(cfg)
        if eid in done:
            print(f"  [{i}/{total}] SKIP: {eid}")
            continue

        print(f"  [{i}/{total}] Running: {eid}")
        try:
            metrics = run_one_config(
                config=config,
                variant=cfg["variant"],
                baseline=baseline,
                train_df=train_df,
                test_df=test_df,
                user_thresholds=user_thresholds,
                user_seen=user_seen,
                all_items=all_items,
                threshold_type=cfg.get("threshold_type"),
                fixed_threshold=cfg.get("fixed_threshold"),
                alpha=cfg.get("alpha"),
                max_users=args.max_users,
                seed=args.seed,
                n_workers=n_workers,
            )
        except Exception as exc:
            print(f"    ERROR: {exc}")
            continue

        k = config.eval.k
        print(f"    NDCG@{k}={metrics.get(f'ndcg@{k}', 0):.4f}  "
              f"negative@{k}={metrics.get(f'negative@{k}', 0):.4f}  "
              f"sim_neg={metrics.get(f'sim_to_neg@{k}', 0):.4f}")

        summary["experiments"].append({
            "exp_id":          eid,
            "variant":         cfg["variant"],
            "threshold_type":  cfg.get("threshold_type"),
            "fixed_threshold": cfg.get("fixed_threshold"),
            "alpha":           cfg.get("alpha"),
            "metrics":         metrics,
        })
        done.add(eid)

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    summary["meta"] = {
        "protocol":    "known_negative_injection",
        "n_neutral":   N_NEUTRAL,
        "n_injected":  N_INJECTED,
        "n_workers":   n_workers,
        "paper":       "Krichene & Rendle 2020 KDD  sampled evaluation framework",
        "speed_notes": (
            "SVD trained once and shared across all 36 grid configs. "
            "User eval parallelised with ThreadPoolExecutor (numpy releases GIL)."
        ),
        "completed_at": datetime.now().isoformat(),
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results saved to {output_path}")
    print("Run generate_all_figures.py to include these in the figures.")


if __name__ == "__main__":
    main()