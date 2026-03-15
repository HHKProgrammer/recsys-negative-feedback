# pipeline step 8 — single experiment runner
# ties together all pipeline steps: seed -> load data -> train svd -> label negatives -> evaluate
# one call to run_single_experiment() runs one complete experiment and saves all outputs
# called by run_grid.py for each of the 36 grid configurations

# Run a single experiment (one variant + one threshold config
# Returns the aggregated metrics dict and saves everything to the run director

from typing import Dict, Optional

import pandas as pd

from src.data.threshold_utils import (
    get_all_user_negative_items,
    get_all_user_negative_items_with_ratings,
)
from src.eval.ranking_metrics import evaluate_ranking
from src.models.negative_variants import FilterNegatives, RerankPenalty, WeightedPenalty
from src.models.svd_baseline import SVDBaseline
from src.utils.config import ExperimentConfig
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.io import load_parquet
from src.utils.seed import set_global_seed


def run_single_experiment(
    config: ExperimentConfig,
    variant: str,
    threshold_type: Optional[str] = None,
    fixed_threshold: Optional[int] = None,
    alpha: Optional[float] = None,
    max_users: Optional[int] = None,
) -> Dict:
    # variant: baseline, filter, rerank, or weighted
    # threshold_type: fixed median or modus(ignored for baseline)
    # fixed_threshold: value when threshold_type=="fixed"
    # alpha: penalty weight for rerank/weighted
    # max_users: evaluate on a subset (quick mode)
    # returns: aggregated metrics dict

    set_global_seed(config.random_seed)

    # Build a descriptive run name like "weighted_fixed_3_a0.1"
    # this becomes the run folder name and is human-readable in outputs/
    name_parts = [variant]
    if threshold_type:
        name_parts.append(threshold_type)
        if threshold_type == "fixed" and fixed_threshold is not None:
            name_parts.append(str(fixed_threshold))
    if alpha is not None:
        name_parts.append(f"a{alpha}")
    run_name = "_".join(name_parts)

    tracker = ExperimentTracker(config.output_dir, run_name)
    tracker.log(f"Starting experiment: {run_name}")

    # Save config stores exactly which parameters were used for this run
    exp_config_dict = config.to_dict()
    exp_config_dict["run"] = {
        "variant": variant,
        "threshold_type": threshold_type,
        "fixed_threshold": fixed_threshold,
        "alpha": alpha,
        "max_users": max_users,
    }
    tracker.save_config(exp_config_dict)

    # Load data
    tracker.log("Loading data ...")
    proc = config.data.processed_path
    train_df = load_parquet(proc + config.splits.train_file)
    test_df = load_parquet(proc + config.splits.test_file)
    user_thresholds = load_parquet(proc + config.splits.user_thresholds_file)
    tracker.log(f"  train={len(train_df):,} | test={len(test_df):,}")

    # Train baseline
    # all variants share the same svd model zhey differ only in ranking logic
    # training happens once here, not once per variant
    tracker.log("Training SVD ...")
    baseline = SVDBaseline(
        n_factors=config.model.n_factors,
        n_epochs=config.model.n_epochs,
        lr_all=config.model.lr_all,
        reg_all=config.model.reg_all,
        random_state=config.random_seed,
    )
    baseline.fit(train_df)
    tracker.log("  SVD trained.")

    # Build model wrap baseline with the selected variant
    if variant == "baseline":
        # no negative feedback pure svd ranking
        model = baseline
        user_negative_items = {uid: set() for uid in test_df["userId"].unique()}
    elif variant == "filter":
        model = FilterNegatives(baseline)
        user_negative_items = get_all_user_negative_items(
            train_df, threshold_type, fixed_threshold, user_thresholds
        )
    elif variant == "rerank":
        model = RerankPenalty(baseline, alpha=alpha)
        # rerank needs set{movieIds no rating values needed
        user_negative_items = get_all_user_negative_items(
            train_df, threshold_type, fixed_threshold, user_thresholds
        )
    elif variant == "weighted":
        model = WeightedPenalty(baseline, alpha=alpha)
        # weighted needs {movieId: rating} to compute severity weights
        user_negative_items = get_all_user_negative_items_with_ratings(
            train_df, threshold_type, fixed_threshold, user_thresholds
        )
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    n_neg = sum(len(v) for v in user_negative_items.values())
    tracker.log(f"  Negative items total: {n_neg:,}")

    # Evaluate — candidate pool = all training items (unseen items only, sampled per user)
    all_items = set(train_df["movieId"].unique())
    tracker.log(f"Evaluating (k={config.eval.k}, n_candidates={config.eval.n_candidates}) ...")
    aggregated, per_user_df = evaluate_ranking(
        model=model,
        test_df=test_df,
        train_df=train_df,
        user_negative_items=user_negative_items,
        all_items=all_items,
        k=config.eval.k,
        n_candidates=config.eval.n_candidates,
        seed=config.eval.random_seed,
        max_users=max_users,
        similarity_fn=baseline.get_similarity,  # always use baseline similarity same latent space
    )

    tracker.log("Results:")
    for metric, value in aggregated.items():
        if metric != "n_users":
            tracker.log(f"  {metric}: {value:.4f}")

    tracker.save_metrics(aggregated)
    tracker.save_per_user_metrics(per_user_df)  # needed for statistical tests later
    tracker.finish()

    return aggregated