# pipeline step 6 — ranking evaluation pipeline
# for each test user: sample 500 unseen candidates + the true next item
# ask the model to rank all 501, then measure how high the true item ended up
# sampled evaluation is standard in recsys research faster than ranking all items
# reference  krichene rendle 2020, on sampled metrics for item recommendation
# https://dl.acm.org/doi/10.1145/3383313.3412259

# Full ranking evaluation pipeline with negative candidate sampling
# One evaluation pass:
# 1. For each test user, sample n_candidates unseen items + add the true test item
# 2. Ask the model to rank them
# 3 Compute P@K, R@K, nDCG@K, Negative@K, Hit@K, MRR.
# 4. Return both aggregated means and a per-user DataFrame for significance tests

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from src.eval.metrics import (
    hit_at_k,
    ndcg_at_k,
    negative_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    sim_to_neg_at_k,
)


def sample_negative_candidates(
    user_id: int,
    all_items: Set[int],
    seen_items: Set[int],
    n: int = 500, # as a standart mention
    rng: Optional[random.Random] = None,
) -> List[int]:
    # Sample n items the user has not seen
    # set difference all_items seen_items gives only unseen items
    # n=500 is standard in recsys sampled evaluation literature
    # random model would hit true item at rank ~250 on average (50% chance)
    # so any improvement over position 250 is a real signal from the model
    unseen = list(all_items - seen_items)
    n = min(n, len(unseen))
    if rng is not None:
        return rng.sample(unseen, n)
    return random.sample(unseen, n)


def evaluate_user(
    user_id: int,
    ranked_items: List[int],
    test_item: int,
    negative_items: Set[int],
    k: int = 10,
    similarity_fn=None,
) -> Dict[str, Any]:
    # computes all 7 metrics for a single user in one call
    # relevant = {test_item} exactly one correct answer per user
    relevant = {test_item}
    return {
        "user_id": user_id,
        f"precision@{k}": precision_at_k(ranked_items, relevant, k),
        f"recall@{k}": recall_at_k(ranked_items, relevant, k),
        f"ndcg@{k}": ndcg_at_k(ranked_items, relevant, k),
        f"negative@{k}": negative_at_k(ranked_items, negative_items, k),
        f"hit@{k}": hit_at_k(ranked_items, test_item, k),
        "mrr": reciprocal_rank(ranked_items, test_item),
        f"sim_to_neg@{k}": sim_to_neg_at_k(ranked_items, negative_items, similarity_fn, k),
    }


def evaluate_ranking(
    model,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    user_negative_items: Dict[int, Any],  # Set[int] or Dict[int,float]
    all_items: Set[int],
    k: int = 10,
    n_candidates: int = 500,
    seed: int = 42,
    max_users: Optional[int] = None,
    similarity_fn=None,
    n_workers: Optional[int] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    # Evaluate a model on the test set.
    # model: SVDBaseline or a negative variant Detected by whether it has a
    #        baseline attribute variant or not
    # user_negative_items: maps userId  set of negative movieIds
    #                      WeightedPenalty instead needs Dict[int,float]
    # max_users if set, evaluate only on the first max_users test users
    # n_workers: how many threads to use defaults to number of CPU cores
    #            threads work well here because numpy releases the GIL during matmul

    from src.models.svd_baseline import SVDBaseline
    from src.models.negative_variants import FilterNegatives, RerankPenalty, WeightedPenalty

    if n_workers is None:
        n_workers = os.cpu_count() or 4

    # own rng instance  isolated from global seed, reproducible per-user sampling
    # note: we pre-generate all candidate lists before threading so the rng order
    # is deterministic regardless of which thread finishes first
    rng = random.Random(seed)
    user_train_items: Dict[int, Set[int]] = (
        train_df.groupby("userId")["movieId"].apply(set).to_dict()
    )

    # pick the test rows we want to evaluate
    subset = test_df.head(max_users) if max_users is not None else test_df

    # pre-sample candidates for every user in a fixed order so results are reproducible
    user_rows = []
    for _, row in subset.iterrows():
        user_id = int(row["userId"])
        test_item = int(row["movieId"])
        seen = user_train_items.get(user_id, set())
        candidates = sample_negative_candidates(user_id, all_items, seen, n_candidates, rng)
        # guarantee the true test item is always in the candidate pool
        if test_item not in candidates:
            candidates.append(test_item)
        user_rows.append((user_id, test_item, candidates))

    def _eval_one(args):
        # runs inside a thread processes one user and returns the metric dict
        user_id, test_item, candidates = args
        negatives = user_negative_items.get(user_id, set() if not isinstance(
            next(iter(user_negative_items.values()), {}), dict) else {})

        if isinstance(model, SVDBaseline):
            ranked = model.rank_items_for_user(user_id, candidates)
        else:
            ranked = model.rank_items_for_user(user_id, candidates, negatives)

        ranked_items = [item for item, _ in ranked]
        neg_set = set(negatives) if isinstance(negatives, dict) else negatives
        return evaluate_user(user_id, ranked_items, test_item, neg_set, k, similarity_fn)

    # run all users in parallel using a thread pool
    # ThreadPoolExecutor is fine here: numpy matmul releases the GIL so threads actually run
    per_user: List[Dict] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_eval_one, args): args for args in user_rows}
        for future in tqdm(as_completed(futures), desc="Evaluating", total=len(user_rows)):
            per_user.append(future.result())

    per_user_df = pd.DataFrame(per_user)

    # average all metrics over all users this is the final result number
    metric_cols = [c for c in per_user_df.columns if c != "user_id"]
    aggregated = {col: float(per_user_df[col].mean()) for col in metric_cols}
    aggregated["n_users"] = len(per_user_df)

    return aggregated, per_user_df