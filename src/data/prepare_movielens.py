# pipeline step 2 — data preparation
# reads raw movielens ratings, cleans them, splits by time, computes per-user thresholds
# temporal leave-one-out split: last rating per user goes to test, second-last to val
# this avoids data leakage — the model only ever sees the past
# reference — koren 2010, collaborative filtering with temporal dynamics
# https://dl.acm.org/doi/10.1145/1721654.1721677

# splitting raw data into clean reproducble trainingdata

# Produces 5 files in the output directory:
#   train.parquet          — all training interactions
#   test.parquet           — last interaction per user
#   train_inner.parquet    — train minus second-last interaction per user
#   val.parquet            — second-last interaction per user (for tuning)
#   user_thresholds.parquet — per-user median, modus, mean, count

# Usage:
#   python -m src.data.prepare_movielens --dataset 1m
#   python -m src.data.prepare_movielens --dataset 10m
#   python -m src.data.prepare_movielens --dataset 20m

import argparse
import json
import os
from typing import Tuple

import pandas as pd

from src.utils.io import save_parquet, ensure_dir


# config per dataset version — ml-1m and ml-10m use :: separator, ml-20m uses csv with header
DATASET_CONFIGS = {
    "1m": {
        "filename": "ratings.dat",
        "sep": "::",
        "columns": ["userId", "movieId", "rating", "timestamp"],
        "has_header": False,
        "raw_subdir": "ml-1m",
    },
    "10m": {
        "filename": "ratings.dat",
        "sep": "::",
        "columns": ["userId", "movieId", "rating", "timestamp"],
        "has_header": False,
        "raw_subdir": "ml-10M100K",
    },
    "20m": {
        "filename": "ratings.csv",
        "sep": ",",
        "columns": ["userId", "movieId", "rating", "timestamp"],
        "has_header": True,
        "raw_subdir": "ml-20m",
    },
}


def load_raw_ratings(raw_dir: str, dataset: str) -> pd.DataFrame:
    # reads the raw ratings file and enforces correct column types
    # ml-1m: ~1m ratings, userId::movieId::rating::timestamp format
    cfg = DATASET_CONFIGS[dataset]
    filepath = os.path.join(raw_dir, cfg["raw_subdir"], cfg["filename"])
    print(f"Loading {filepath} ...")

    if cfg["has_header"]:
        df = pd.read_csv(filepath)
        df.columns = cfg["columns"]
    else:
        df = pd.read_csv(
            filepath,
            sep=cfg["sep"],
            names=cfg["columns"],
            engine="python",
        )

    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)
    print(f"  Loaded {len(df):,} ratings | {df['userId'].nunique():,} users | {df['movieId'].nunique():,} items")
    return df


"""filtering out if a user rates to less movies min rating so that we cannot releay on their opinion or also movies /objectsthat have to less ratings
min of 5 was in literature mentioned a lot """
def filter_cold_start(df: pd.DataFrame, min_ratings: int = 5) -> pd.DataFrame:
    # iterative filter — removing users can cause items to drop below threshold, and vice versa
    # loop runs until no more rows are removed, usually converges in 1-2 iterations
    # min_ratings=5 is standard cold-start threshold in recsys literature
    # reference — he et al 2017, neural collaborative filtering
    # https://dl.acm.org/doi/10.1145/3038912.3052569
    print(f"\nFiltering cold-start (min {min_ratings} ratings per user/item) ...")
    prev_len = 0
    iteration = 0
    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1
        user_counts = df["userId"].value_counts()
        df = df[df["userId"].isin(user_counts[user_counts >= min_ratings].index)]
        item_counts = df["movieId"].value_counts()
        df = df[df["movieId"].isin(item_counts[item_counts >= min_ratings].index)]
        print(f"  Iter {iteration}: {len(df):,} ratings remaining")
    print(f"  After filtering: {df['userId'].nunique():,} users | {df['movieId'].nunique():,} items")
    return df


def remap_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
    # Assign dense 0-based integer IDs to users and items since we got so much empty space it make ssense to start id by 0
    # original ml-1m movie ids go 1..3952 with gaps , model matrices need contiguous 0-based indices
    # without remapping surprise would waste memory on empty matrix rows
    print("\nRemapping IDs to dense integers ...")
    user_map = {u: i for i, u in enumerate(sorted(df["userId"].unique()))}
    item_map = {v: i for i, v in enumerate(sorted(df["movieId"].unique()))}
    df = df.copy()
    df["userId"] = df["userId"].map(user_map).astype(int)
    df["movieId"] = df["movieId"].map(item_map).astype(int)
    print(f"  Users: 0–{len(user_map)-1} | Items: 0–{len(item_map)-1}")
    return df, user_map, item_map


def last_item_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Hold out the last-rated item per user as testmfor TRain/TEst slit
    # after last_item_split gets trained eith train_df the second last rating gets used for hyperparameter tuning val.parquet
    df = df.sort_values(["userId", "timestamp"]) #sort chronological
    test_idx = df.groupby("userId").tail(1).index #last reviews per iser
    test_df = df.loc[test_idx].copy()
    train_df = df.drop(test_idx).copy()
    return train_df, test_df


def compute_user_thresholds(train_df: pd.DataFrame) -> pd.DataFrame:
    # Compute per-user median, modus (mode), and mean from training ratings
    # trying diffrent configs like median modus etc helps later to decide what is actually considered negative?
    # computed from train only, no leakage from val or test
    # on ml-1m: average median ~3.84, average mode ~3.99 across all users
    print("\nComputing per-user thresholds ...")

    def safe_mode(s): #all modes iff they are same oftan
        m = s.mode()
        return m.iloc[0] if len(m) > 0 else s.median() # iloc as first  if no modus (maybe empty) median is fallback

    stats = train_df.groupby("userId")["rating"].agg(
        median_rating="median",
        modus_rating=safe_mode,
        mean_rating="mean",
        count="count",
    ).reset_index()

    print(f"  Median mean={stats['median_rating'].mean():.2f} | Modus mean={stats['modus_rating'].mean():.2f}")
    return stats


def prepare_movielens(
    raw_dir: str,
    output_dir: str,
    dataset: str = "1m",
    min_ratings: int = 5,
) -> None:
    # main pipeline function , calls all steps in order and saves outputs
    print(f"\n{'='*60}")
    print(f"Preparing MovieLens {dataset.upper()}")
    print(f"{'='*60}")

    ensure_dir(output_dir)

    df = load_raw_ratings(raw_dir, dataset)
    df = filter_cold_start(df, min_ratings)
    df, user_map, item_map = remap_ids(df)

    # Primary split: last item per user → test
    train_df, test_df = last_item_split(df)
    print(f"\nSplit: train={len(train_df):,} | test={len(test_df):,}")

    # Secondary split: second-last item per user → val
    # gives a held-out set for hyperparameter tuning without ever touching test
    train_inner_df, val_df = last_item_split(train_df)
    print(f"Split: train_inner={len(train_inner_df):,} | val={len(val_df):,}")

    # Thresholds computed on train only (no leakage)
    user_thresholds = compute_user_thresholds(train_df)

    # Save splits
    print("\nSaving files ...")
    save_parquet(train_df, os.path.join(output_dir, "train.parquet"))
    save_parquet(test_df, os.path.join(output_dir, "test.parquet"))
    save_parquet(train_inner_df, os.path.join(output_dir, "train_inner.parquet"))
    save_parquet(val_df, os.path.join(output_dir, "val.parquet"))
    save_parquet(user_thresholds, os.path.join(output_dir, "user_thresholds.parquet"))

    # Save original ID mappings for reference
    mappings = {
        "n_users": len(user_map),
        "n_items": len(item_map),
    }
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(mappings, f, indent=2)

    print(f"\nAll files written to {output_dir}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Prepare MovieLens dataset splits")
    parser.add_argument("--dataset", choices=["1m", "10m", "20m"], default="1m")
    parser.add_argument("--raw_dir", default="data/raw/movielens")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--min_ratings", type=int, default=5)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"data/processed/movielens/ml-{args.dataset}"

    prepare_movielens(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        min_ratings=args.min_ratings,
    )


if __name__ == "__main__":
    main()