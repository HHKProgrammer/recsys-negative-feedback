"""
Arm C  Hybrid Recommender Grid

Arm C combines Arm A (positive preference) and Arm B (dislike-risk detector):

  final_score(u, i) = minmax(arm_a_score) - alpha * minmax(arm_b_score)

Min-max normalisation is per-user per-candidate-list so alpha is a true
blending weight independent of raw SVD score scale.
Flat score range -> 0.5 for all items (Arm C degrades to Arm A).

Grid: pos_threshold [3,4,5] x neg_threshold [1,2,3,median,modus] x alpha [0.05..1.0]
      = 90 combinations per dataset.

Parallelism: 15 pair-tasks (one per (pos, neg) model pair), each task scores
arm_a and arm_b ONCE for all users, then applies all 6 alphas cheaply.
This is 6x fewer predict_batch() calls than the naive per-combination approach.

Speedup profile:
  naive:  90 tasks x N_users x 2 predict_batch calls  = 180 * N scoring passes
  paired: 15 tasks x N_users x 2 predict_batch calls  =  30 * N scoring passes  (6x)
  alpha application is pure numpy arithmetic on 501-element arrays (negligible).

References
  Paudel, Bonta, Bernstein & Kuflik (2016)
  "Fewer flops at the top: Accuracy and Diversity in Recommender Systems"

  Sinha, Sodhani, Pineau & Hamilton (2019)
  "Negative interactions and the value of dispreference information"

Output
  outputs/<dataset>/grid_summary_arm_c.json

Usage
  python scripts/run_arm_c_hybrid_grid.py --config configs/movielens_1m.yaml
  python scripts/run_arm_c_hybrid_grid.py --config configs/movielens_1m.yaml --n_parallel 12
  python scripts/run_arm_c_hybrid_grid.py --config configs/movielens_1m.yaml --n_parallel 12 --max_users 5000
"""

import argparse
import concurrent.futures as cf
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

# Per-worker state populated by _init_worker (fork/spawn initializer)
_SHARED: Dict = {}


def pos_label(t: int) -> str:
    return f"pos_ge_{t}"


def exp_id(pt: int, neg_label: str, alpha: float) -> str:
    return f"arm_c_{pos_label(pt)}_{neg_label}_alpha_{alpha:.2f}"


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < EPS:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def _score_and_evaluate_pair(
    arm_a: SVDBaseline,
    arm_b: SVDBaseline,
    alphas_needed: List[float],
    pt: int,
    neg_cfg: dict,
    test_df: pd.DataFrame,
    user_seen: Dict[int, Set[int]],
    user_neg_items: Dict[int, Set[int]],
    all_items: Set[int],
    k: int,
    n_candidates: int,
    seed: int,
    max_users: Optional[int],
) -> List[Dict]:
    """Score arm_a and arm_b ONCE per user, then apply all alphas cheaply.

    For 90 combinations with 15 unique model pairs, this avoids 5/6 of all
    predict_batch() calls compared to evaluating each (pos, neg, alpha) separately.
    """
    rng    = random.Random(seed)
    subset = test_df.head(max_users) if max_users else test_df

    # One pass: build candidates, score both arms, collect per-alpha metrics
    alpha_metrics: Dict[float, List[Dict]] = {a: [] for a in alphas_needed}

    for _, row in subset.iterrows():
        uid       = int(row["userId"])
        test_item = int(row["movieId"])
        seen      = user_seen.get(uid, set())
        pool      = list(all_items - seen)
        cands     = rng.sample(pool, min(n_candidates, len(pool)))
        if test_item not in cands:
            cands.append(test_item)

        # Score both arms once — the expensive matmul
        a_scores = arm_a.predict_batch(uid, cands).astype(np.float64)
        b_scores = arm_b.predict_batch(uid, cands).astype(np.float64)
        a_norm   = minmax_norm(a_scores)
        b_norm   = minmax_norm(b_scores)

        neg_set  = user_neg_items.get(uid, set())
        relevant = {test_item}

        for alpha in alphas_needed:
            final      = a_norm - alpha * b_norm
            ranked_pairs = sorted(zip(cands, final.tolist()), key=lambda x: x[1], reverse=True)
            ranked     = [item for item, _ in ranked_pairs]
            alpha_metrics[alpha].append({
                f"ndcg@{k}":       ndcg_at_k(ranked, relevant, k),
                f"hit@{k}":        hit_at_k(ranked, test_item, k),
                "mrr":             reciprocal_rank(ranked, test_item),
                f"precision@{k}":  precision_at_k(ranked, relevant, k),
                f"recall@{k}":     recall_at_k(ranked, relevant, k),
                f"negative@{k}":   negative_at_k(ranked, neg_set, k),
                f"sim_to_neg@{k}": sim_to_neg_at_k(ranked, neg_set, arm_a.get_similarity, k),
            })

    results = []
    for alpha in alphas_needed:
        df  = pd.DataFrame(alpha_metrics[alpha])
        agg = {col: float(df[col].mean()) for col in df.columns}
        agg["n_users"] = len(df)
        results.append({
            "exp_id":              exp_id(pt, neg_cfg["label"], alpha),
            "variant":             "arm_c",
            "pos_threshold":       pt,
            "neg_threshold_type":  neg_cfg["type"],
            "neg_threshold_value": str(neg_cfg["value"]),
            "neg_label":           neg_cfg["label"],
            "alpha":               alpha,
            "metrics":             agg,
        })
    return results


# multiprozessing
def _init_worker(config_path: str, models_dir_str: str,
                 seed_eval: int, k: int, n_candidates: int) -> None:
    """Load data once per worker process (called by ProcessPoolExecutor initializer)."""
    global _SHARED
    config = ExperimentConfig.from_yaml(config_path)
    proc   = config.data.processed_path
    _SHARED["train_df"]        = load_parquet(proc + config.splits.train_file)
    _SHARED["test_df"]         = load_parquet(proc + config.splits.test_file)
    _SHARED["user_thresholds"] = load_parquet(proc + config.splits.user_thresholds_file)
    _SHARED["all_items"]       = set(_SHARED["train_df"]["movieId"].unique())
    _SHARED["user_seen"]       = (
        _SHARED["train_df"].groupby("userId")["movieId"].apply(set).to_dict()
    )
    _SHARED["models_dir"]      = Path(models_dir_str)
    _SHARED["model_cache"]     = {}
    _SHARED["seed_eval"]       = seed_eval
    _SHARED["k"]               = k
    _SHARED["n_candidates"]    = n_candidates


def _worker_pair(task: tuple) -> Optional[List[Dict]]:
    """Evaluate one (pos, neg) model pair for all missing alphas.

    Scores arm_a and arm_b ONCE per user, then applies all needed alphas.
    Returns a list of result dicts (one per alpha), or None if all done.
    """
    pt, neg_cfg, max_users, done_set = task

    alphas_needed = [a for a in ALPHAS if exp_id(pt, neg_cfg["label"], a) not in done_set]
    if not alphas_needed:
        return None

    s   = _SHARED
    md  = s["models_dir"]
    a_key = pos_label(pt)
    b_key = neg_cfg["label"]

    for key, sub in [(a_key, "arm_a"), (b_key, "arm_b")]:
        if key not in s["model_cache"]:
            p = md / sub / key / "model.pkl"
            if not p.exists():
                return [{"exp_id": exp_id(pt, neg_cfg["label"], a), "_missing": sub}
                        for a in alphas_needed]
            s["model_cache"][key] = SVDBaseline.load(str(p))

    arm_a = s["model_cache"][a_key]
    arm_b = s["model_cache"][b_key]
    train_df        = s["train_df"]
    user_thresholds = s["user_thresholds"]

    if neg_cfg["type"] == "fixed":
        user_neg_items = get_all_user_negative_items(
            train_df, "fixed", int(neg_cfg["value"]), user_thresholds)
    else:
        user_neg_items = get_all_user_negative_items(
            train_df, neg_cfg["type"], None, user_thresholds)

    set_global_seed(42)
    return _score_and_evaluate_pair(
        arm_a=arm_a,
        arm_b=arm_b,
        alphas_needed=alphas_needed,
        pt=pt,
        neg_cfg=neg_cfg,
        test_df=s["test_df"],
        user_seen=s["user_seen"],
        user_neg_items=user_neg_items,
        all_items=s["all_items"],
        k=s["k"],
        n_candidates=s["n_candidates"],
        seed=s["seed_eval"],
        max_users=max_users,
    )


# main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--max_users",  type=int, default=None)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n_parallel", type=int, default=1,
                        help="Worker processes for parallel pair evaluation. "
                             "Recommended: total_cores / parallel_datasets "
                             "(e.g. 12 on a 48-core server running 4 datasets in parallel).")
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

    k            = config.eval.k
    n_candidates = config.eval.n_candidates
    seed_eval    = config.eval.random_seed

    all_combos = [(pt, nc, a)
                  for pt in POS_THRESHOLDS
                  for nc in NEG_CONFIGS
                  for a in ALPHAS]
    n_remaining = sum(1 for pt, nc, a in all_combos
                      if exp_id(pt, nc["label"], a) not in done)

    print(f"\nArm C  Hybrid Recommender Grid")
    print(f"  Dataset:     {config.data.name}")
    print(f"  Pos:         {POS_THRESHOLDS}")
    print(f"  Neg:         {[c['label'] for c in NEG_CONFIGS]}")
    print(f"  Alpha:       {ALPHAS}")
    print(f"  Total:       {len(all_combos)} combinations  ({n_remaining} remaining)")
    print(f"  Workers:     {args.n_parallel}")
    print(f"  Strategy:    15 model pairs, each scored once, 6 alphas applied per pair")
    print(f"  Output:      {out_path}")
    print()

    pair_tasks = [
        (pt, neg_cfg, args.max_users, frozenset(done))
        for pt in POS_THRESHOLDS
        for neg_cfg in NEG_CONFIGS
    ]

    if args.n_parallel > 1:
        #  parallel path 
        with cf.ProcessPoolExecutor(
            max_workers=args.n_parallel,
            initializer=_init_worker,
            initargs=(args.config, str(models), seed_eval, k, n_candidates),
        ) as pool:
            futures = {pool.submit(_worker_pair, t): t for t in pair_tasks}
            with tqdm(total=n_remaining, desc="Arm C grid") as pbar:
                for fut in cf.as_completed(futures):
                    results_list = fut.result()
                    if not results_list:
                        continue
                    for result in results_list:
                        if "_missing" in result:
                            tqdm.write(f"  SKIP {result['exp_id']} "
                                       f"(model missing: {result['_missing']})")
                            pbar.update(1)
                            continue
                        m = result["metrics"]
                        tqdm.write(f"  {result['exp_id']}: "
                                   f"nDCG@{k}={m[f'ndcg@{k}']:.4f}  "
                                   f"HR@{k}={m[f'hit@{k}']:.4f}  "
                                   f"sim_neg@{k}={m[f'sim_to_neg@{k}']:.4f}")
                        summary["experiments"].append(result)
                        done.add(result["exp_id"])
                        pbar.update(1)
                        with open(out_path, "w") as f:
                            json.dump(summary, f, indent=2)

    else:
        #  sequential path 
        proc            = config.data.processed_path
        train_df        = load_parquet(proc + config.splits.train_file)
        test_df         = load_parquet(proc + config.splits.test_file)
        user_thresholds = load_parquet(proc + config.splits.user_thresholds_file)
        all_items       = set(train_df["movieId"].unique())
        user_seen       = train_df.groupby("userId")["movieId"].apply(set).to_dict()
        _model_cache: Dict[str, Optional[SVDBaseline]] = {}

        def _load(arm_sub: str, label: str) -> Optional[SVDBaseline]:
            key = f"{arm_sub}/{label}"
            if key not in _model_cache:
                p = models / arm_sub / label / "model.pkl"
                if not p.exists():
                    print(f"    WARNING: model not found at {p}")
                    _model_cache[key] = None
                else:
                    _model_cache[key] = SVDBaseline.load(str(p))
            return _model_cache[key]

        for pt in POS_THRESHOLDS:
            for neg_cfg in NEG_CONFIGS:
                alphas_needed = [a for a in ALPHAS
                                 if exp_id(pt, neg_cfg["label"], a) not in done]
                if not alphas_needed:
                    print(f"  SKIP (all done): arm_c_{pos_label(pt)}_{neg_cfg['label']}")
                    continue

                print(f"  Pair: arm_c_{pos_label(pt)}_{neg_cfg['label']}  "
                      f"({len(alphas_needed)} alphas needed)")

                arm_a = _load("arm_a", pos_label(pt))
                arm_b = _load("arm_b", neg_cfg["label"])
                if arm_a is None or arm_b is None:
                    print(f"    SKIP (missing model)")
                    continue

                if neg_cfg["type"] == "fixed":
                    user_neg_items = get_all_user_negative_items(
                        train_df, "fixed", int(neg_cfg["value"]), user_thresholds)
                else:
                    user_neg_items = get_all_user_negative_items(
                        train_df, neg_cfg["type"], None, user_thresholds)

                set_global_seed(args.seed)
                results = _score_and_evaluate_pair(
                    arm_a=arm_a,
                    arm_b=arm_b,
                    alphas_needed=alphas_needed,
                    pt=pt,
                    neg_cfg=neg_cfg,
                    test_df=test_df,
                    user_seen=user_seen,
                    user_neg_items=user_neg_items,
                    all_items=all_items,
                    k=k,
                    n_candidates=n_candidates,
                    seed=seed_eval,
                    max_users=args.max_users,
                )
                for result in results:
                    m = result["metrics"]
                    print(f"    alpha={result['alpha']:.2f}  "
                          f"nDCG@{k}={m[f'ndcg@{k}']:.4f}  "
                          f"HR@{k}={m[f'hit@{k}']:.4f}  "
                          f"sim_neg@{k}={m[f'sim_to_neg@{k}']:.4f}")
                    summary["experiments"].append(result)
                    done.add(result["exp_id"])
                with open(out_path, "w") as f:
                    json.dump(summary, f, indent=2)

    summary["meta"] = {
        "approach":    "arm_c_hybrid",
        "formula":     "final_score = minmax(pos_score) - alpha * minmax(neg_score)",
        "speed_notes": (
            "15 model pairs scored once each; 6 alphas applied per pair via numpy arithmetic. "
            "6x fewer predict_batch() calls vs naive per-combination approach."
        ),
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