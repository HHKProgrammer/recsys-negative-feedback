# spotify mssd data preparation
# unlike movielens, spotify data IS sequential because it comes from real listening sessions
# so i use a temporal split here: last 20% of sessions become test
# reference: Brost et al. 2019 "The Music Streaming Sessions Dataset" (WSDM Cup 2019)
#
# key differences from movielens:
# 1. temporal split is correct here because sessions have real order
# 2. negative feedback is implicit (skips) not explicit (star ratings)
# 3. not all skips are equal: skip after 2 seconds means more than skip after 50 seconds
# 4. context matters: skipping (skipping in a row) while searching for a song is normal, not a dislike signal
#
# skip taxonomy (4 levels defined by spotify in the mssd dataset):
#   skip_1 = very brief listen (first few seconds)  strongest negative
#   skip_2 = brief listen (several seconds)          moderate negative
#   skip_3 = most of track played, then skipped      weak negative
#   not_skipped = full track completed               positive signal
#
# i also differentiate by context because a skip during catalog browsing
# (searching for a specific song) is not really a dislike signal

import json
import os
from pathlib import Path
from typing import Tuple

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
# if i skip from my own collection that is a strong signal
# if i skip while browsing charts that could just be looking for something specific
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
    # returns 0.0 for completed tracks (positive), up to 1.0 for immediate skips in focused listening

    # determine skip level
    if row.get("not_skipped", False):
        skip_w = SKIP_WEIGHTS["not_skipped"]
    elif row.get("skip_1", False):
        skip_w = SKIP_WEIGHTS["skip_1"]
    elif row.get("skip_2", False):
        skip_w = SKIP_WEIGHTS["skip_2"]
    elif row.get("skip_3", False):
        skip_w = SKIP_WEIGHTS["skip_3"]
    else:
        skip_w = 0.5  # unknown, assume moderate

    # context modifier
    context = row.get("context_type", "unknown")
    modifier = CONTEXT_WEIGHTS.get(context, 0.5)

    return round(skip_w * modifier, 4)


def is_search_behavior(row: pd.Series) -> bool:
    # detects if this skip happened while the user was searching or browsing
    # skipping while looking for a specific song is not a dislike signal
    # i check: catalog context + user-initiated track start (click or forward) + context switch
    indicators = [
        row.get("context_type") == "catalog",
        row.get("reason_start") in ["click_row", "forward_button"],
        row.get("context_switch", 0) > 0,
    ]
    # need at least 2 signals to classify as search behavior
    return sum(indicators) >= 2


def load_spotify_sessions(raw_dir: str) -> pd.DataFrame:
    # loads the spotify mssd csv files
    # the dataset has separate files for training and test logs
    # expected columns: session_id, track_id_clean, skip_1, skip_2, skip_3, not_skipped,
    #                   context_type, reason_start, reason_end, etc.
    raw_path = Path(raw_dir)

    # try the mini dataset first (for development), fall back to full training set
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


def temporal_split_sessions(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # temporal split: last 20% of sessions by time become the test set
    # this is correct for spotify because the data IS sequential
    # i sort sessions by their first track play timestamp and split at 80%
    #
 
    # spotify sessions capture real listening behavior over time
    # i want to predict future sessions from past sessions
    # random split would leak future listening patterns into training

    # get the timestamp of the first track in each session to represent session time
    session_times = df.groupby("session_id")["session_position"].min().reset_index()
    session_times = session_times.sort_values("session_position")

    n = len(session_times)
    split_idx = int(n * (1 - test_ratio))

    train_sessions = set(session_times.iloc[:split_idx]["session_id"])
    test_sessions = set(session_times.iloc[split_idx:]["session_id"])

    train_df = df[df["session_id"].isin(train_sessions)].copy()
    test_df = df[df["session_id"].isin(test_sessions)].copy()

    return train_df, test_df


def prepare_spotify(
    raw_dir: str,
    output_dir: str,
    test_ratio: float = 0.2,
) -> None:
    # full spotify preparation pipeline
    # 1. load sessions
    # 2. compute negative weights (skip level + context)
    # 3. flag search behavior (skips that are not real negatives)
    # 4. temporal split (last 20% of sessions = test)
    # 5. save to parquet + split_info.json

    print(f"\n{'='*60}")
    print("Preparing Spotify MSSD")
    print(f"{'='*60}")
    print(f"Split: temporal, test_ratio={test_ratio}")
    print("(temporal split because spotify data IS sequential)")

    ensure_dir(output_dir)

    df = load_spotify_sessions(raw_dir)

    # add negative weight column combining skip level and context
    print("\nComputing negative weights (skip level x context) ...")
    df["negative_weight"] = df.apply(compute_negative_weight, axis=1)

    # flag search/browse behavior where skips should not count as negative
    df["is_search"] = df.apply(is_search_behavior, axis=1)

    # binary negative flag: skip_2 or stronger AND not search behavior
    # i use skip_2 as the threshold because skip_1 and skip_2 both indicate clear dislike
    df["is_negative"] = (
        (df.get("skip_2", pd.Series(False, index=df.index)) == True) &
        (~df["is_search"])
    )

    skip_rate = df["is_negative"].mean() * 100
    search_rate = df["is_search"].mean() * 100
    print(f"  Skip (negative) rate: {skip_rate:.1f}%")
    print(f"  Search behavior rate: {search_rate:.1f}% (excluded from negatives)")

    # temporal split
    train_df, test_df = temporal_split_sessions(df, test_ratio)
    print(f"\nTemporal split: train={len(train_df):,} plays in {train_df['session_id'].nunique():,} sessions")
    print(f"               test ={len(test_df):,} plays in {test_df['session_id'].nunique():,} sessions")

    # save
    print("\nSaving files ...")
    save_parquet(train_df, os.path.join(output_dir, "train.parquet"))
    save_parquet(test_df, os.path.join(output_dir, "test.parquet"))

    split_info = {
        "split_type": "temporal_sessions",
        "test_ratio": test_ratio,
        "train_sessions": int(train_df["session_id"].nunique()),
        "test_sessions": int(test_df["session_id"].nunique()),
        "negative_rate_pct": round(skip_rate, 2),
        "search_excluded_pct": round(search_rate, 2),
        "reference": "Brost et al. 2019 WSDM Cup, temporal split because data is sequential",
    }
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nAll files written to {output_dir}")
    print("Done.")