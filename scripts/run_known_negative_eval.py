"""
Known-Negative Candidate Injection Evaluation


Standard evaluation (as in  main grid) samples 500 UNSEEN items as candidates.
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
comparable to the main grid numbers they use a different candidate set.

Output

  outputs/<dataset>/grid_summary_known_neg_eval.json full experiment summary
  (same format as grid_summary.json so generate_all_figures.py can read it)

Usage

  python scripts/run_known_negative_eval.py --config configs/movielens_1m.yaml
  python scripts/run_known_negative_eval.py --config configs/movielens_10m.yaml
  python scripts/run_known_negative_eval.py --config configs/spotify.yaml
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from tqdm import tqdm

# add project root to path so imports work when called as a script
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


# how many neutral (unseen) items to add alongside the injected negatives
# total candidates per user = n_neutral + n_injected_negatives + 1 true test item
N_NEUTRAL = 450
N_INJECTED = 50   # at most 50 of the user's own dislikes are injected


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
    """
    Build a candidate pool that mixes:
      - n_inject items from the user's own disliked items (the hard part)
      - n_neutral items sampled from truly unseen non-negative items
      - the true test item (always included)

    This deliberately makes the task harder: the model must rank the true item
    above both neutral unseen items AND items it should know are bad.
    """
    # inject up to n_inject of the user's disliked items
    neg_list = list(negative_items - {test_item})
    injected = rng.sample(neg_list, min(n_inject, len(neg_list)))

    # fill the rest with neutral unseen items (unseen AND not negative)
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
        neg_for_rank = neg_set if not isinstance(next(iter([neg_set] or [set()]), None), dict) else neg_set
        ranked = model.rank_items_for_user(user_id, candidates, neg_for_rank)

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
    threshold_type=None,
    fixed_threshold=None,
    alpha=None,
    max_users=None,
    seed: int = 42,
) -> Dict:
    set_global_seed(seed)
    rng = random.Random(seed)

    proc = config.data.processed_path
    train_df = load_parquet(proc + config.splits.train_file)
    test_df   = load_parquet(proc + config.splits.test_file)
    user_thresholds = load_parquet(proc + config.splits.user_thresholds_file)

    baseline = SVDBaseline(
        n_factors=config.model.n_factors,
        n_epochs=config.model.n_epochs,
        lr_all=config.model.lr_all,
        reg_all=config.model.reg_all,
        random_state=seed,
    )
    baseline.fit(train_df)

    is_baseline = (variant == "baseline")
    if is_baseline:
        model = baseline
        user_neg = {uid: set() for uid in test_df["userId"].unique()}
    elif variant == "filter":
        model = FilterNegatives(baseline)
        user_neg = get_all_user_negative_items(train_df, threshold_type, fixed_threshold, user_thresholds)
    elif variant == "rerank":
        model = RerankPenalty(baseline, alpha=alpha)
        user_neg = get_all_user_negative_items(train_df, threshold_type, fixed_threshold, user_thresholds)
    elif variant == "weighted":
        model = WeightedPenalty(baseline, alpha=alpha)
        user_neg = get_all_user_negative_items_with_ratings(train_df, threshold_type, fixed_threshold, user_thresholds)
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    all_items = set(train_df["movieId"].unique())
    user_seen = train_df.groupby("userId")["movieId"].apply(set).to_dict()

    k = config.eval.k
    subset = test_df.head(max_users) if max_users else test_df

    per_user = []
    for _, row in tqdm(subset.iterrows(), total=len(subset),
                       desc=f"{variant} (known-neg eval)", leave=False):
        user_id  = int(row["userId"])
        test_item = int(row["movieId"])
        seen      = user_seen.get(user_id, set())
        neg_set   = set(user_neg.get(user_id, set()))

        if len(neg_set) == 0:
            # user has no labeled negatives — fall back to standard unseen sampling
            unseen = list(all_items - seen)
            candidates = rng.sample(unseen, min(N_NEUTRAL + N_INJECTED, len(unseen)))
            if test_item not in candidates:
                candidates.append(test_item)
        else:
            candidates = build_injected_candidates(
                user_id, test_item, seen, neg_set, all_items, rng
            )

        result = evaluate_one_user(
            user_id, model, is_baseline, candidates,
            test_item, neg_set, k, baseline.get_similarity,
        )
        per_user.append(result)

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
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    parser.add_argument("--max_users", type=int, default=None,
                        help="Limit to first N users (quick test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    output_path = Path(config.output_dir).parent / "grid_summary_known_neg_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # load existing results so we can resume if interrupted
    if output_path.exists():
        with open(output_path) as f:
            summary = json.load(f)
        done = {e["exp_id"] for e in summary.get("experiments", [])}
    else:
        summary = {"experiments": [], "meta": {}}
        done = set()

    print(f"\nKnown-Negative Candidate Injection Evaluation")
    print(f"  config: {args.config}")
    print(f"  output: {output_path}")
    print(f"  protocol: {N_INJECTED} known-disliked items injected per user")
    print(f"            + {N_NEUTRAL} neutral unseen items + 1 true test item")
    print(f"  paper basis: Krichene & Rendle 2020 (KDD) — sampled evaluation framework")
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
                max_users=args.max_users,
                seed=args.seed,
                **cfg,
            )
        except Exception as exc:
            print(f"    ERROR: {exc}")
            continue

        k = config.eval.k
        print(f"    NDCG@{k}={metrics.get(f'ndcg@{k}', 0):.4f}  "
              f"negative@{k}={metrics.get(f'negative@{k}', 0):.4f}  "
              f"sim_neg={metrics.get(f'sim_to_neg@{k}', 0):.4f}")

        record = {
            "exp_id":          eid,
            "variant":         cfg["variant"],
            "threshold_type":  cfg.get("threshold_type"),
            "fixed_threshold": cfg.get("fixed_threshold"),
            "alpha":           cfg.get("alpha"),
            "metrics":         metrics,
        }
        summary["experiments"].append(record)
        done.add(eid)

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    summary["meta"] = {
        "protocol":    "known_negative_injection",
        "n_neutral":   N_NEUTRAL,
        "n_injected":  N_INJECTED,
        "paper":       "Krichene & Rendle 2020 KDD — sampled evaluation framework",
        "completed_at": datetime.now().isoformat(),
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results saved to {output_path}")
    print("Run generate_all_figures.py to include these in the figures.")


if __name__ == "__main__":
    main()
