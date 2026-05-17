"""
Arm C  Hybrid Recommender Grid

Arm C combines Arm A (positive preference) and Arm B (dislike-risk detector):

  final_score(u, i) = minmax(arm_a_score) - alpha * minmax(arm_b_score)

Min-max normalisation is per-user per-candidate-list so alpha is a true
blending weight independent of raw SVD score scale.
Flat score range -> 0.5 for all items (Arm C degrades to Arm A).

Grid: pos_threshold [3,4,5] x neg_threshold [1,2,3,median,modus] x alpha [0.05..1.0]
      = 90 combinations per dataset.

Parallelism: use --n_parallel N to run N combinations simultaneously via
             Python multiprocessing (ProcessPoolExecutor, initializer pattern).
             Each worker loads data once; models are cached inside each worker.
             like: --n_parallel = total_cores / parallel_datasets
             Example 48-core server, 4 datasets in parallel: --n_parallel 12

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


def rank_arm_c(
    arm_a: SVDBaseline,
    arm_b: SVDBaseline,
    user_id: int,
    candidates: List[int],
    alpha: float,
) -> List[Tuple[int, float]]:
    pos_scores = arm_a.predict_batch(user_id, candidates).astype(np.float64)
    neg_scores = arm_b.predict_batch(user_id, candidates).astype(np.float64)
    final = minmax_norm(pos_scores) - alpha * minmax_norm(neg_scores)
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
    show_progress: bool = True,
) -> Dict:
    rng = random.Random(seed)
    user_seen = train_df.groupby("userId")["movieId"].apply(set).to_dict()
    subset = test_df.head(max_users) if max_users else test_df

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
    iterable = tqdm(user_rows, desc=f"alpha={alpha:.2f}", leave=False,
                    disable=not show_progress)
    for uid, test_item, cands in iterable:
        ranked = [item for item, _ in rank_arm_c(arm_a, arm_b, uid, cands, alpha)]
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


#  multiprocessing worker 

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
    _SHARED["models_dir"]      = Path(models_dir_str)
    _SHARED["model_cache"]     = {}
    _SHARED["seed_eval"]       = seed_eval
    _SHARED["k"]               = k
    _SHARED["n_candidates"]    = n_candidates


def _worker(task: tuple) -> Optional[Dict]:
    """Evaluate one combination in a worker process. Returns result dict or None (skip)."""
    pt, neg_cfg, alpha, max_users, done_set = task

    eid = exp_id(pt, neg_cfg["label"], alpha)
    if eid in done_set:
        return None  # already completed in a previous run

    s   = _SHARED
    md  = s["models_dir"]

    a_key = pos_label(pt)
    b_key = neg_cfg["label"]

    if a_key not in s["model_cache"]:
        p = md / "arm_a" / a_key / "model.pkl"
        if not p.exists():
            return {"exp_id": eid, "_missing": "arm_a"}
        s["model_cache"][a_key] = SVDBaseline.load(str(p))

    if b_key not in s["model_cache"]:
        p = md / "arm_b" / b_key / "model.pkl"
        if not p.exists():
            return {"exp_id": eid, "_missing": "arm_b"}
        s["model_cache"][b_key] = SVDBaseline.load(str(p))

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
    metrics = evaluate_arm_c(
        arm_a=arm_a, arm_b=arm_b, alpha=alpha,
        test_df=s["test_df"], train_df=train_df,
        user_neg_items=user_neg_items, all_items=s["all_items"],
        k=s["k"], n_candidates=s["n_candidates"],
        seed=s["seed_eval"], max_users=max_users,
        show_progress=False,   # suppress per-user bars inside worker
    )

    return {
        "exp_id":              eid,
        "variant":             "arm_c",
        "pos_threshold":       pt,
        "neg_threshold_type":  neg_cfg["type"],
        "neg_threshold_value": str(neg_cfg["value"]),
        "neg_label":           neg_cfg["label"],
        "alpha":               alpha,
        "metrics":             metrics,
    }


#  m main 

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--max_users",  type=int, default=None)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n_parallel", type=int, default=1,
                        help="Worker processes for parallel combination evaluation. "
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

    grid = [
        (pt, neg_cfg, alpha)
        for pt in POS_THRESHOLDS
        for neg_cfg in NEG_CONFIGS
        for alpha in ALPHAS
    ]
    n_remaining = sum(1 for pt, nc, a in grid
                      if exp_id(pt, nc["label"], a) not in done)

    print(f"\nArm C  Hybrid Recommender Grid")
    print(f"  Dataset:     {config.data.name}")
    print(f"  Pos:         {POS_THRESHOLDS}")
    print(f"  Neg:         {[c['label'] for c in NEG_CONFIGS]}")
    print(f"  Alpha:       {ALPHAS}")
    print(f"  Total:       {len(grid)} combinations  ({n_remaining} remaining)")
    print(f"  Workers:     {args.n_parallel}")
    print(f"  Output:      {out_path}")
    print()

    if args.n_parallel > 1:
        #  parallel path 
        tasks = [(pt, nc, alpha, args.max_users, frozenset(done))
                 for pt, nc, alpha in grid]

        with cf.ProcessPoolExecutor(
            max_workers=args.n_parallel,
            initializer=_init_worker,
            initargs=(args.config, str(models), seed_eval, k, n_candidates),
        ) as pool:
            futures = {pool.submit(_worker, t): t for t in tasks}
            with tqdm(total=len(grid), desc="Arm C combinations") as pbar:
                for fut in cf.as_completed(futures):
                    pbar.update(1)
                    result = fut.result()
                    if result is None:
                        continue
                    if "_missing" in result:
                        tqdm.write(f"  SKIP {result['exp_id']} (model missing: {result['_missing']})")
                        continue
                    m = result["metrics"]
                    tqdm.write(f"  {result['exp_id']}: "
                               f"nDCG@{k}={m[f'ndcg@{k}']:.4f}  "
                               f"HR@{k}={m[f'hit@{k}']:.4f}  "
                               f"sim_neg@{k}={m[f'sim_to_neg@{k}']:.4f}")
                    summary["experiments"].append(result)
                    done.add(result["exp_id"])
                    with open(out_path, "w") as f:
                        json.dump(summary, f, indent=2)

    else:
        #  sequential path (original behaviour) 
        proc            = config.data.processed_path
        train_df        = load_parquet(proc + config.splits.train_file)
        test_df         = load_parquet(proc + config.splits.test_file)
        user_thresholds = load_parquet(proc + config.splits.user_thresholds_file)
        all_items       = set(train_df["movieId"].unique())
        _model_cache: Dict[str, Optional[SVDBaseline]] = {}

        def _load(arm_sub, label):
            key = f"{arm_sub}/{label}"
            if key not in _model_cache:
                p = models / arm_sub / label / "model.pkl"
                if not p.exists():
                    print(f"    WARNING: model not found at {p}")
                    _model_cache[key] = None
                else:
                    _model_cache[key] = SVDBaseline.load(str(p))
            return _model_cache[key]

        for i, (pt, neg_cfg, alpha) in enumerate(grid, 1):
            eid = exp_id(pt, neg_cfg["label"], alpha)
            if eid in done:
                continue
            print(f"  [{i}/{len(grid)}] {eid}")

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
            metrics = evaluate_arm_c(
                arm_a=arm_a, arm_b=arm_b, alpha=alpha,
                test_df=test_df, train_df=train_df,
                user_neg_items=user_neg_items, all_items=all_items,
                k=k, n_candidates=n_candidates,
                seed=seed_eval, max_users=args.max_users,
            )
            print(f"    nDCG@{k}={metrics[f'ndcg@{k}']:.4f}  "
                  f"HR@{k}={metrics[f'hit@{k}']:.4f}  "
                  f"MRR={metrics['mrr']:.4f}  "
                  f"sim_neg@{k}={metrics[f'sim_to_neg@{k}']:.4f}")

            summary["experiments"].append({
                "exp_id":              eid,
                "variant":             "arm_c",
                "pos_threshold":       pt,
                "neg_threshold_type":  neg_cfg["type"],
                "neg_threshold_value": str(neg_cfg["value"]),
                "neg_label":           neg_cfg["label"],
                "alpha":               alpha,
                "metrics":             metrics,
            })
            done.add(eid)
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)

    summary["meta"] = {
        "approach":    "arm_c_hybrid",
        "formula":     "final_score = minmax(pos_score) - alpha * minmax(neg_score)",
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