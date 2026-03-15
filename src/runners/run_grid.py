# pipeline step 9 — full experiment grid
# runs all 36 experiments in sequence and saves a summary with best results
# resume-safe: if the process crashes restart and it picks up where it stopped
# the checkpoint file tracks which experiments are already done

# Run the full experiment grid (36 experiments for ML-1M).
# Experiment grid:
#   1   baseline
#   5   filter x {fixed(1 fixed(2, fixed(3, median, modus}
#   15  rerank x 5 thresholds x 3 alphas
#   15  weighted x 5 thresholds x 3 alphas
#   : 1 + 5 + 5x3x3 = 1 + 5 + 30 = 36 Experimente minus 3 Filter-Duplikate bei Alphas = 33 effektive
#
# The grid is resume-safema checkpoint file records which run_ids have
# complete If you kill the process and restart, it skips finished runs
#
# The summary is built from ALL runs saved on disk (not just the current
# session so checkpoint-resumed experiments are always included

import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.runners.run_experiment import run_single_experiment
from src.utils.config import ExperimentConfig
from src.utils.io import ensure_dir

# checkpoint file stores list of completed experiment ids as json
CHECKPOINT_FILE = "outputs/.grid_checkpoint.json"


def _load_checkpoint() -> List[str]:
    # returns list of alreadycompleted experiment ids
    # empty list if no checkpoint exists yet first run
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return []


def _save_checkpoint(completed: List[str]) -> None:
    # saves checkpoint after each experiment so progress is never lost
    ensure_dir(os.path.dirname(CHECKPOINT_FILE))
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(completed, f, indent=2)


def _experiment_id(variant, threshold_type, fixed_threshold, alpha) -> str:
    # builds a unique readable id like "weighted_fixed_3_a0.1"
    # used as the checkpoint key and as part of the run folder name
    parts = [variant or "baseline"]
    if threshold_type:
        parts.append(threshold_type)
        if threshold_type == "fixed" and fixed_threshold is not None:
            parts.append(str(fixed_threshold))
    if alpha is not None:
        parts.append(f"a{alpha}")
    return "_".join(parts)


def build_experiment_list(config: ExperimentConfig, quick: bool = False) -> List[Dict]:
    # Return the ordered list of experiment specs.
    # full grid: 1 baseline + 5 filter + 15 rerank + 15 weighted = 36 total
    # quick mode: reduced set for sanity checks (500 users, 1 threshold, 1 alpha)
    experiments = []

    # Baseline always first
    experiments.append(dict(
        variant="baseline",
        threshold_type=None,
        fixed_threshold=None,
        alpha=None,
    ))

    if quick:
        thresholds = [(config.quick_mode.thresholds[0], "fixed")]
        alphas = config.quick_mode.alphas
        variants = [v for v in config.quick_mode.variants if v != "baseline"]
    else:
        # 5 threshold configs: fixed(1), fixed(2), fixed(3), median, modus
        thresholds = (
            [(t, "fixed") for t in config.negative_feedback.fixed_thresholds]
            + [(None, "median"), (None, "modus")]
        )
        alphas = config.negative_feedback.alphas  # [0.1, 0.3, 1.0]
        variants = ["filter", "rerank", "weighted"]

    for (fixed_t, thresh_type) in thresholds:
        for variant in variants:
            if variant == "filter":
                # filter has no alpha only one experiment per threshold
                experiments.append(dict(
                    variant="filter",
                    threshold_type=thresh_type,
                    fixed_threshold=fixed_t,
                    alpha=None,
                ))
            else:
                # rerank and weighted each run 3 times per threshold once per alpha
                for alpha in alphas:
                    experiments.append(dict(
                        variant=variant,
                        threshold_type=thresh_type,
                        fixed_threshold=fixed_t,
                        alpha=alpha,
                    ))

    return experiments


def run_full_grid(config_path: str, quick: bool = False) -> None:
    config = ExperimentConfig.from_yaml(config_path)
    ensure_dir(config.output_dir)

    experiments = build_experiment_list(config, quick)
    completed = _load_checkpoint()

    print(f"\n{'='*60}")
    print(f"Experiment grid: {len(experiments)} runs ({'quick' if quick else 'full'})")
    print(f"Already completed: {len(completed)}")
    print(f"{'='*60}\n")

    all_results = []

    for i, exp in enumerate(experiments, 1):
        exp_id = _experiment_id(**exp)
        if exp_id in completed:
            print(f"[{i}/{len(experiments)}] SKIP (done): {exp_id}")
            continue

        print(f"\n[{i}/{len(experiments)}] Running: {exp_id}")

        max_users = config.quick_mode.max_users if quick else None

        metrics = run_single_experiment(
            config=config,
            variant=exp["variant"],
            threshold_type=exp["threshold_type"],
            fixed_threshold=exp["fixed_threshold"],
            alpha=exp["alpha"],
            max_users=max_users,
        )

        result = {**exp, "exp_id": exp_id, "metrics": metrics}
        all_results.append(result)

        # save checkpoint immediately after each run  crash safety
        completed.append(exp_id)
        _save_checkpoint(completed)

    _save_grid_summary(config_path, config, all_results)
    print(f"\nGrid complete. See {config.output_dir}")


def _load_all_results_from_disk(runs_dir: str) -> List[Dict]:
    # Rebuild experiment results by reading every run folder on disk.
    # This makes the summary complete even when some runs were resumed from
    # the checkpoint and not executed in the current session.
    # Runs that were executed in quick mode (max_users < full test set) are
    # detected via their config.json and excluded with a warning, since their
    # metrics are not comparable to full-evaluation runs.
    results = []
    for run_dir in sorted(Path(runs_dir).iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue

        with open(config_path) as f:
            saved_cfg = json.load(f)
        with open(metrics_path) as f:
            metrics = json.load(f)

        run_info = saved_cfg.get("run", {})
        max_users = run_info.get("max_users")
        if max_users is not None:
            # skip quick-mode runs their metrics used fewer users and are not comparable
            print(
                f"  WARNING: skipping quick-mode run '{run_dir.name}' "
                f"(max_users={max_users}, metrics not comparable to full runs)"
            )
            continue

        results.append(
            {
                "exp_id": _experiment_id(
                    run_info.get("variant"),
                    run_info.get("threshold_type"),
                    run_info.get("fixed_threshold"),
                    run_info.get("alpha"),
                ),
                "variant": run_info.get("variant"),
                "threshold_type": run_info.get("threshold_type"),
                "fixed_threshold": run_info.get("fixed_threshold"),
                "alpha": run_info.get("alpha"),
                "metrics": {k: v for k, v in metrics.items() if k != "runtime_seconds"},
            }
        )
    return results


def _save_grid_summary(config_path: str, config: ExperimentConfig, _unused_results: List[Dict]) -> None:
    # Save a summary JSON built from ALL run folders on disk
    # _unused_results argument is kept for backward compatibility but is
    # ignored function always reads fresh data from disk so that
    # checkpoint-resumed experiments are never missing from the summary

    results = _load_all_results_from_disk(config.output_dir)

    if not results:
        print("No completed runs found on disk — summary not written.")
        return

    metric_keys = [
        k for k in results[0]["metrics"].keys()
        if k not in ("n_users",)
    ]

    # Metrics where LOWER is better
    # sim_to_neg@10 lower = recommendations are further from hated items = better
    lower_is_better = {"negative@10", "sim_to_neg@10"}

    best_by_metric: Dict[str, Dict] = {}
    for metric in metric_keys:
        reverse = metric not in lower_is_better  # True = sort descending = higher is better
        sorted_r = sorted(
            results,
            key=lambda r: r["metrics"].get(metric, float("inf")),
            reverse=reverse,
        )
        best = sorted_r[0]
        best_by_metric[metric] = {
            "exp_id": best["exp_id"],
            "value": best["metrics"].get(metric),
        }

    summary = {
        "generated_at": datetime.now().isoformat(),
        "config_file": config_path,
        "n_experiments": len(results),
        "experiments": results,
        "best_by_metric": best_by_metric,
    }

    path = os.path.join(config.output_dir, "..", "grid_summary.json")
    path = os.path.normpath(path)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved -> {path}")