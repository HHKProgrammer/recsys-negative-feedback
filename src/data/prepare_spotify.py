# spotify mssd data preparation
# unlike movielens, spotify data IS sequential because it comes from real listening sessions
# reference: Brost et al. 2019 "The Music Streaming Sessions Dataset" (WSDM Cup 2019)
#
# key differences from movielens:
# 1. negative feedback is implicit (skips) not explicit (star ratings)
# 2. not all skips are equal: skip after 2 seconds means more than skip after 50 seconds
# 3. context matters: skipping while searching for a song is normal, not a dislike signal
#
# i treat each listening session as a "user" and each track as an "item"
# this maps the session data to the same userId/movieId/rating format as movielens
# so the same evaluation pipeline runs on both datasets unchanged
#
# rating mapping: inverted negative weight scaled to 1-5
#   not_skipped (weight=0.0) -> rating 5  (fully listened, positive)
#   skip_3      (weight=0.3) -> rating 4  (mostly listened, weak positive)
#   skip_2      (weight=0.7) -> rating 2  (brief listen, negative)
#   skip_1      (weight=1.0) -> rating 1  (immediate skip, strongly negative)
#
# skip taxonomy (4 levels defined by spotify in the mssd dataset):
#   skip_1 = very brief listen (first few seconds)  strongest negative
#   skip_2 = brief listen (several seconds)          moderate negative
#   skip_3 = most of track played, then skipped      weak negative
#   not_skipped = full track completed               positive signal

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.io import save_parquet, ensure_dir


# how strongly each skip level counts as negative feedback
# skip_1 is the strongest negative because the user left within seconds
SKIP_WEIGHTS = {
    "skip_1": 1.0,   # very brief: user hated it immediately
    "skip_2": 0.7,   # brief: user gave it a chance but still left
    "skip_3": 0.3,   # most played: user almost finished it, weak negative
    "not_skipped": 0.0,  # completed: positive signal
}

# context modifier: skips from focused listening contexts mean more than browsing skips
CONTEXT_WEIGHTS = {
    "user_collection": 1.0,        # skip from own library is strong dislike
    "personalized_playlist": 0.9,  # skip from my recommendations is strong
    "editorial_playlist": 0.7,     # skip from curated playlist is moderate
    "radio": 0.5,                  # skip from radio is expected exploration
    "charts": 0.4,                 # skip from charts is browsing
    "catalog": 0.2,                # skip while searching is not really negative
}


def compute_negative_weight(row: pd.Series) -> float:
    # combines skip level and context to get one negative weight per track play
    # weight = skip_level_weight * context_modifier
    # returns 0.0 for completed tracks (positive), up to 1.0 for immediate skips
    if row.get("not_skipped", False):
        skip_w = SKIP_WEIGHTS["not_skipped"]
    elif row.get("skip_1", False):
        skip_w = SKIP_WEIGHTS["skip_1"]
    elif row.get("skip_2", False):
        skip_w = SKIP_WEIGHTS["skip_2"]
    elif row.get("skip_3", False):
        skip_w = SKIP_WEIGHTS["skip_3"]
    else:
        skip_w = 0.5

    context = row.get("context_type", "unknown")
    modifier = CONTEXT_WEIGHTS.get(context, 0.5)
    return round(skip_w * modifier, 4)


def weight_to_rating(w: float) -> int:
    # maps negative weight (0=positive, 1=most negative) to integer rating 1-5
    # inverts the scale so the existing pipeline (higher rating = more positive) works unchanged
    return max(1, round(5 - w * 4))


def is_search_behavior(row: pd.Series) -> bool:
    # detects if this skip happened while the user was searching or browsing
    # skipping while looking for a specific song is not a dislike signal
    indicators = [
        row.get("context_type") == "catalog",
        row.get("reason_start") in ["click_row", "forward_button"],
        row.get("context_switch", 0) > 0,
    ]
    return sum(indicators) >= 2


def load_spotify_sessions(raw_dir: str) -> pd.DataFrame:
    # loads the spotify mssd csv files
    # tries the mini dataset first (for development), falls back to full training set
    raw_path = Path(raw_dir)
    mini = raw_path / "training_set" / "log_mini.csv"
    full = raw_path / "training_set" / "log_0_20180715_000000000000.csv"

    if mini.exists():
        print(f"Loading {mini} ...")
        df = pd.read_csv(mini)
    elif full.exists():
        print(f"Loading {full} ...")
        df = pd.read_csv(full)
    else:
        raise FileNotFoundError(
            f"No spotify data found in {raw_dir}/training_set/\n"
            "Expected: log_mini.csv or log_0_20180715_000000000000.csv"
        )

    print(f"  Loaded {len(df):,} track plays | {df['session_id'].nunique():,} sessions")
    return df


def filter_cold_start(df: pd.DataFrame, min_ratings: int = 5) -> pd.DataFrame:
    # iterative cold-start filter: min interactions per user (session) and item (track)
    # same logic as movielens — loop until stable
    print(f"\nFiltering cold-start (min {min_ratings} plays per session/track) ...")
    prev_len = 0
    iteration = 0
    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1
        user_counts = df["userId"].value_counts()
        df = df[df["userId"].isin(user_counts[user_counts >= min_ratings].index)]
        item_counts = df["movieId"].value_counts()
        df = df[df["movieId"].isin(item_counts[item_counts >= min_ratings].index)]
        print(f"  Iter {iteration}: {len(df):,} plays remaining")
    print(f"  After filtering: {df['userId'].nunique():,} sessions | {df['movieId'].nunique():,} tracks")
    return df


def remap_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
    # assign dense 0-based integer IDs to sessions and tracks
    print("\nRemapping IDs to dense integers ...")
    user_map = {u: i for i, u in enumerate(sorted(df["userId"].unique()))}
    item_map = {v: i for i, v in enumerate(sorted(df["movieId"].unique()))}
    df = df.copy()
    df["userId"] = df["userId"].map(user_map).astype(int)
    df["movieId"] = df["movieId"].map(item_map).astype(int)
    print(f"  Sessions: 0–{len(user_map)-1} | Tracks: 0–{len(item_map)-1}")
    return df, user_map, item_map


def random_leave_one_out_split(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # hold out one random track per session for test/val
    # using random (not temporal) because sessions are already split temporally at the session level
    test_idx = df.groupby("userId").sample(n=1, random_state=seed).index
    test_df = df.loc[test_idx].copy()
    train_df = df.drop(test_idx).copy()
    return train_df, test_df


def compute_user_thresholds(train_df: pd.DataFrame) -> pd.DataFrame:
    # compute per-session median and mode of ratings
    # same logic as movielens so the same threshold strategies work
    rows = []
    for user_id, grp in train_df.groupby("userId"):
        ratings = grp["rating"].values
        median_r = float(np.median(ratings))
        mode_vals = pd.Series(ratings).mode()
        modus_r = float(mode_vals.iloc[0]) if len(mode_vals) > 0 else median_r
        mean_r = float(np.mean(ratings))
        rows.append({"userId": user_id, "median_rating": median_r, "modus_rating": modus_r, "mean_rating": mean_r})
    return pd.DataFrame(rows)


def prepare_spotify(
    raw_dir: str,
    output_dir: str,
    test_ratio: float = 0.2,
    min_ratings: int = 5,
) -> None:
    # full spotify preparation pipeline
    # 1. load sessions
    # 2. compute negative weights (skip level + context) -> integer ratings 1-5
    # 3. map session_id -> userId, track_id_clean -> movieId
    # 4. cold-start filter
    # 5. remap IDs to dense integers
    # 6. random leave-one-out for test (seed=42) and val (seed=43)
    # 7. compute per-session thresholds
    # 8. save all 5 parquet files + dataset_info.json

    print(f"\n{'='*60}")
    print("Preparing Spotify MSSD")
    print(f"{'='*60}")
    print(f"Split: random leave-one-out per session, test_ratio={test_ratio}")
    print("(sessions are treated as users; one track held out per session for test/val)")

    ensure_dir(output_dir)

    raw_df = load_spotify_sessions(raw_dir)

    # compute negative weights and convert to ratings
    print("\nComputing negative weights (skip level x context) ...")
    raw_df["negative_weight"] = raw_df.apply(compute_negative_weight, axis=1)

    skip_rate = (raw_df["negative_weight"] > 0).mean() * 100
    search_rate = raw_df.apply(is_search_behavior, axis=1).mean() * 100
    print(f"  Skip (negative) rate: {skip_rate:.1f}%")
    print(f"  Search behavior rate: {search_rate:.1f}% (informational only)")

    # convert to integer rating scale 1-5 (inverted: higher = more positive)
    raw_df["rating"] = raw_df["negative_weight"].apply(weight_to_rating)

    # rename to standard pipeline columns
    df = raw_df[["session_id", "track_id_clean", "rating"]].rename(
        columns={"session_id": "userId", "track_id_clean": "movieId"}
    ).copy()

    # keep one row per (session, track)aggregate duplicate plays by taking the min rating
    # (most negative interaction wins: if a user skipped a track once, that's a dislike signal)
    df = df.groupby(["userId", "movieId"], as_index=False)["rating"].min()

    # cold-start filter
    df = filter_cold_start(df, min_ratings=min_ratings)

    # remap IDs
    df, user_map, item_map = remap_ids(df)

    # remove sessions with fewer than 3 plays (can't hold out test + val + 1 training item)
    session_counts = df.groupby("userId").size()
    valid_sessions = session_counts[session_counts >= 3].index
    df = df[df["userId"].isin(valid_sessions)].copy()
    print(f"  After min-3 filter: {df['userId'].nunique():,} sessions")

    # random leave-one-out: test (seed=42) then val from remaining train (seed=43)
    train_df, test_df = random_leave_one_out_split(df, seed=42)
    train_inner_df, val_df = random_leave_one_out_split(train_df, seed=43)

    print(f"\nSplit (random leave-one-out):")
    print(f"  train={len(train_df):,} | test={len(test_df):,} ({test_df['userId'].nunique():,} sessions)")
    print(f"  train_inner={len(train_inner_df):,} | val={len(val_df):,}")

    # compute per-session thresholds from training data
    print("\nComputing per-session thresholds ...")
    user_thresholds = compute_user_thresholds(train_df)
    med_mean = user_thresholds["median_rating"].mean()
    mod_mean = user_thresholds["modus_rating"].mean()
    print(f"  Median mean={med_mean:.2f} | Modus mean={mod_mean:.2f}")

    # save all 5 parquet files
    print("\nSaving files ...")
    save_parquet(train_df, os.path.join(output_dir, "train.parquet"))
    save_parquet(train_inner_df, os.path.join(output_dir, "train_inner.parquet"))
    save_parquet(val_df, os.path.join(output_dir, "val.parquet"))
    save_parquet(test_df, os.path.join(output_dir, "test.parquet"))
    save_parquet(user_thresholds, os.path.join(output_dir, "user_thresholds.parquet"))

    dataset_info = {
        "split_type": "random_leave_one_out_per_session",
        "n_sessions": int(df["userId"].nunique()),
        "n_tracks": int(df["movieId"].nunique()),
        "n_plays": len(df),
        "skip_rate_pct": round(skip_rate, 2),
        "rating_scale": "1-5 (5=fully listened, 1=immediate skip)",
        "reference": "Brost et al. 2019 WSDM Cup",
    }
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\nAll files written to {output_dir}")
    print("Done.")