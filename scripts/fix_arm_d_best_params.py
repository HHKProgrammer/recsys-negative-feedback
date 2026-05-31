"""
Fix / normalize Arm-D best_params.json files.

Problem:
Some Arm-D tuning folders contain old flat best_params.json files:

{
  "n_factors": 10,
  "n_epochs": 30,
  "lr_all": 0.002,
  "reg_all": 0.05,
  "biased": false
}

Other folders contain the newer nested format:

{
  "best_value": 0.3009,
  "metric": "...",
  "best_params": {...}
}

Some folders have study.db or trials.csv but no best_params.json.

This script normalizes all Arm-D best_params.json files to the new nested format
without deleting existing data. Old files are backed up.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


LABELS = [
    ("pos_ge_4_neg_le_2", "4", "2"),
    ("pos_ge_5_neg_le_1", "5", "1"),
    ("pos_ge_4_neg_le_1", "4", "1"),
    ("pos_ge_3_neg_le_2", "3", "2"),
]

DATASET_INFO = {
    "ml-1m": {
        "base": Path("outputs/movielens/ml-1m/tuning/arm_d"),
        "dataset": "movielens-1m",
    },
    "ml-10m": {
        "base": Path("outputs/movielens/ml-10m/tuning/arm_d"),
        "dataset": "movielens-10m",
    },
    "ml-20m": {
        "base": Path("outputs/movielens/ml-20m/tuning/arm_d"),
        "dataset": "movielens-20m",
    },
    "spotify": {
        "base": Path("outputs/spotify/tuning/arm_d"),
        "dataset": "spotify",
    },
}


def is_param_dict(data: dict) -> bool:
    return {"n_factors", "n_epochs", "lr_all", "reg_all"}.issubset(data.keys())


def normalize_params(params: dict) -> dict:
    return {
        "n_factors": int(params["n_factors"]),
        "n_epochs": int(params["n_epochs"]),
        "lr_all": float(params["lr_all"]),
        "reg_all": float(params["reg_all"]),
        "biased": bool(params.get("biased", False)),
    }


def backup_file(path: Path) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak_{ts}")
    shutil.copy2(path, backup)
    print(f"    Backup -> {backup}")


def load_from_optuna(run_dir: Path) -> Optional[dict]:
    db_path = run_dir / "study.db"
    if not db_path.exists():
        return None

    try:
        import optuna

        storage = f"sqlite:///{db_path}"
        summaries = optuna.study.get_all_study_summaries(storage=storage)
        if not summaries:
            return None

        # Pick the first study in this DB.
        study_name = summaries[0].study_name
        study = optuna.load_study(study_name=study_name, storage=storage)
        best = study.best_trial

        params = normalize_params(best.params)

        return {
            "best_value": float(best.value),
            "n_trials": len(study.trials),
            "best_params": params,
            "source": "study.db",
        }

    except Exception as exc:
        print(f"    Could not read Optuna study.db: {exc}")
        return None


def find_metric_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "value",
        "best_value",
        "ndcg@10",
        "ndcg_10",
        "nDCG@10",
        "best_ndcg_10",
        "score",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_from_trials_csv(run_dir: Path) -> Optional[dict]:
    csv_path = run_dir / "trials.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        metric_col = find_metric_col(df)

        if metric_col is None:
            print(f"    trials.csv columns: {list(df.columns)}")
            return None

        required = {"n_factors", "n_epochs", "lr_all", "reg_all"}
        if not required.issubset(df.columns):
            print(f"    trials.csv missing param columns. Columns: {list(df.columns)}")
            return None

        best = df.loc[df[metric_col].idxmax()]
        params = normalize_params(best.to_dict())

        return {
            "best_value": float(best[metric_col]),
            "n_trials": int(len(df)),
            "best_params": params,
            "source": "trials.csv",
        }

    except Exception as exc:
        print(f"    Could not read trials.csv: {exc}")
        return None


def write_payload(
    out_path: Path,
    dataset_name: str,
    label: str,
    threshold: str,
    neg_threshold: str,
    best_value: Optional[float],
    n_trials: Optional[int],
    params: dict,
    source: str,
) -> None:
    payload = {
        "arm": "d",
        "threshold": threshold,
        "neg_threshold": neg_threshold,
        "threshold_label": label,
        "dataset": dataset_name,
        "best_value": best_value,
        "metric": "nDCG@10 (val, binary-target model)",
        "n_trials": n_trials,
        "best_params": normalize_params(params),
        "source": source,
        "note": (
            "Arm D uses binary targets: rating>=pos_threshold -> 1.0, "
            "rating<=neg_threshold -> 0.0, middle ratings dropped."
        ),
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"    Wrote normalized best_params.json -> {out_path}")


def fix_one(dataset_key: str, label: str, threshold: str, neg_threshold: str, normalize_existing: bool) -> None:
    info = DATASET_INFO[dataset_key]
    base = info["base"]
    dataset_name = info["dataset"]

    run_dir = base / label
    out_path = run_dir / "best_params.json"

    print(f"\n[{dataset_key}] {label}")

    if not run_dir.exists():
        print(f"    Missing folder: {run_dir}")
        return

    if out_path.exists():
        data = json.loads(out_path.read_text())

        # Already new format
        if "best_params" in data:
            print("    Already nested format.")
            if normalize_existing:
                backup_file(out_path)
                write_payload(
                    out_path=out_path,
                    dataset_name=dataset_name,
                    label=label,
                    threshold=threshold,
                    neg_threshold=neg_threshold,
                    best_value=data.get("best_value"),
                    n_trials=data.get("n_trials"),
                    params=data["best_params"],
                    source=data.get("source", "existing_nested_json"),
                )
            return

        # Old flat format
        if is_param_dict(data):
            print("    Flat JSON found. Normalizing.")
            backup_file(out_path)
            write_payload(
                out_path=out_path,
                dataset_name=dataset_name,
                label=label,
                threshold=threshold,
                neg_threshold=neg_threshold,
                best_value=data.get("best_value"),
                n_trials=data.get("n_trials"),
                params=data,
                source="existing_flat_json",
            )
            return

        print(f"    Unsupported existing JSON keys: {list(data.keys())}")
        return

    # Missing best_params.json: try study.db first
    loaded = load_from_optuna(run_dir)
    if loaded is None:
        loaded = load_from_trials_csv(run_dir)

    if loaded is None:
        print("    No best_params.json, and could not extract from study.db/trials.csv.")
        return

    write_payload(
        out_path=out_path,
        dataset_name=dataset_name,
        label=label,
        threshold=threshold,
        neg_threshold=neg_threshold,
        best_value=loaded["best_value"],
        n_trials=loaded["n_trials"],
        params=loaded["best_params"],
        source=loaded["source"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(DATASET_INFO.keys()) + ["all"],
        required=True,
        help="Dataset to fix. Can be repeated. Use --dataset all for all datasets.",
    )
    parser.add_argument(
        "--normalize-existing",
        action="store_true",
        help="Also rewrite already nested JSON files in normalized form.",
    )
    args = parser.parse_args()

    datasets = args.dataset
    if "all" in datasets:
        datasets = list(DATASET_INFO.keys())

    for dataset_key in datasets:
        for label, threshold, neg_threshold in LABELS:
            fix_one(
                dataset_key=dataset_key,
                label=label,
                threshold=threshold,
                neg_threshold=neg_threshold,
                normalize_existing=args.normalize_existing,
            )


if __name__ == "__main__":
    main()
