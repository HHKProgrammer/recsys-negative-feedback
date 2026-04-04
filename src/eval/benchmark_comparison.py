# i compare my svd results to the official surprise benchmarks here
# surprise publishes rmse/mae for all algorithms on ml-1m using 5fold cross-validation
# source: https://surpriselib.com ( table on homepage)
# this is a bit extra but easier for me to keep track
# i use this to validate my implementation is correct
# my rmse should be close to surprise svd default  (0.873 )
# i expect to be slightly better because i tuned the hyperparameters
# this comparison gives my results academic context and credibility

from typing import Dict
import pandas as pd

# official surprise benchmarks taken from surpriselib.com
# 5-fold cross-validation, default parameters, ml-1m dataset
SURPRISE_BENCHMARKS = {
    "ml-1m": {
        "SVD":          {"rmse": 0.873, "mae": 0.686},
        "SVD++":        {"rmse": 0.862, "mae": 0.672},
        "NMF":          {"rmse": 0.916, "mae": 0.723},
        "KNNBaseline":  {"rmse": 0.895, "mae": 0.706},
        "BaselineOnly": {"rmse": 0.909, "mae": 0.719},
        "Random":       {"rmse": 1.504, "mae": 1.206},
    },
    "ml-100k": {
        "SVD":   {"rmse": 0.934, "mae": 0.737},
        "SVD++": {"rmse": 0.919, "mae": 0.721},
    },
}


def compare_to_surprise(my_rmse: float, my_mae: float = None, dataset: str = "ml-1m") -> pd.DataFrame:
    # builds a comparison table of surprise benchmark vs my result
    # prints whether my result is  the expected range for a correctly implemented svd
    benchmarks = SURPRISE_BENCHMARKS[dataset]

    rows = []
    for algo, metrics in benchmarks.items():
        rows.append({
            "Algorithm": algo,
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
            "Source": "Surprise default (5-fold CV)",
        })

    rows.append({
        "Algorithm": "My SVD (tuned)",
        "RMSE": round(my_rmse, 4),
        "MAE": round(my_mae, 4) if my_mae else "N/A",
        "Source": "This thesis",
    })

    df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

    # how much better are my results compared to surprise default svd
    default_rmse = benchmarks["SVD"]["rmse"]
    improvement = (default_rmse - my_rmse) / default_rmse * 100
    print(f"\nSurprise SVD default RMSE : {default_rmse}")
    print(f"My tuned SVD RMSE         : {my_rmse}")
    print(f"Improvement over default  : {improvement:+.1f}%")

    # sanity check that my implementation is correct
    if my_rmse > benchmarks["Random"]["rmse"]:
        print("ERROR rmse worse than random, something is wrong with the implementation")
    elif my_rmse > benchmarks["SVD"]["rmse"] + 0.05:
        print("WARNING rmse significantly worse than surprise default svd, check hyperparameters")
    elif my_rmse < benchmarks["SVD++"]["rmse"] - 0.05:
        print("WARNING rmse suspiciously low, check for data leakage")
    else:
        print("OK rmse is in the expected range for a tuned SVD on this dataset")

    return df


def print_benchmark_table(dataset: str = "ml-1m") -> None:
    # just prints the surprise reference table, useful for the thesis section
    print(f"\nSurprise benchmarks ({dataset}, 5-fold CV, default params):")
    print("-" * 50)
    for algo, metrics in SURPRISE_BENCHMARKS[dataset].items():
        print(f"  {algo:<15} RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}")