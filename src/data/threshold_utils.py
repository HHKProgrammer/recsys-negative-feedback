# pipeline step 3 — negative item labeling
# decides for each user which of their training ratings count as "negative feedback"
# three strategies: fixed threshold, per-user median, per-user mode (modus)
# this is the core of my research ,defining what negative actually means

# Utilities for labelling user ratings as negative.
# Threshold semantics:
#   fixed:  rating <= threshold  (e.g. threshold=2 -> {1, 2} are negative)
#   median: rating <  user_median   strict less-than so median itself is not negative
#   modus:  rating <  user_mode     same idea

from typing import Dict, Optional, Set

import pandas as pd


def _get_adaptive_threshold(
    user_id: int, # for each user
    user_thresholds: pd.DataFrame,
    col: str,
) -> Optional[float]:
    # looks up the precomputed threshold for a specific user from user_thresholds.parquet
    # returns None if user not found  caller handles that as empty negative set
    row = user_thresholds[user_thresholds["userId"] == user_id]
    if len(row) == 0:
        return None #if user doesnt exist
    return float(row[col].iloc[0]) # gets threshold for each user


def get_user_negative_items( #main
    user_id: int,
    train_df: pd.DataFrame,
    threshold_type: str,
    fixed_threshold: int = 2, #rated 2 or lower try 4 later etc this was testing 1m
    user_thresholds: pd.DataFrame = None,
) -> Set[int]:
    # Return the set of movieIds that user_id rated negatively.
    # threshold_type: "fixed", "median", or "modus"
    # fixed_threshold: rating threshold for "fixed" type (rating <= threshold -> negative)
    # user_thresholds: DataFrame with columns [userId, median_rating, modus_rating]

    # key difference between fixed and adaptive:
    # fixed uses <= (inclusive) so rating=threshold counts as negative
    # median/modus use strict < souser's own average rating is not considered negative
    # a generous user with median=4 would have 3-stars as NOT negative with < but negative with <=

    user_ratings = train_df[train_df["userId"] == user_id]

    if threshold_type == "fixed": # lower than fixed threshold
        mask = user_ratings["rating"] <= fixed_threshold
    elif threshold_type == "median": #rated lower tha avarage
        t = _get_adaptive_threshold(user_id, user_thresholds, "median_rating")
        if t is None:
            return set()
        mask = user_ratings["rating"] < t  # strict less-thanmedian itself is not negative
    elif threshold_type == "modus": #rated lower tha avarage
        t = _get_adaptive_threshold(user_id, user_thresholds, "modus_rating")
        if t is None:
            return set()
        mask = user_ratings["rating"] < t  # strict less-than  mode itself is not negative
    else:
        raise ValueError(f"Unknown threshold_type: {threshold_type!r}")

    # returns a set for O(1) membership checks later  "if item in negative_items
    return set(user_ratings.loc[mask, "movieId"].astype(int).values)


def get_user_negative_items_with_ratings(
    user_id: int,
    train_df: pd.DataFrame,
    threshold_type: str,
    fixed_threshold: int = 2,
    user_thresholds: pd.DataFrame = None,
) -> Dict[int, float]:
    # Same as get_user_negative_items, but returns {movieId: rating} for all user at the same time
    # Used by WeightedPenalty to scale penalty by how negative the rating was.
    # the rating value matters here  a 1star gets a higher penalty weight than a 2star
    user_ratings = train_df[train_df["userId"] == user_id]

    if threshold_type == "fixed":
        neg = user_ratings[user_ratings["rating"] <= fixed_threshold]
    elif threshold_type == "median":
        t = _get_adaptive_threshold(user_id, user_thresholds, "median_rating")
        if t is None:
            return {}
        neg = user_ratings[user_ratings["rating"] < t]
    elif threshold_type == "modus":
        t = _get_adaptive_threshold(user_id, user_thresholds, "modus_rating")
        if t is None:
            return {}
        neg = user_ratings[user_ratings["rating"] < t]
    else:
        raise ValueError(f"Unknown threshold_type: {threshold_type!r}")

    # dict(zip(...)) builds {movieId: rating} in one line
    return dict(zip(neg["movieId"].astype(int), neg["rating"].astype(float)))


def get_all_user_negative_items(
    train_df: pd.DataFrame,
    threshold_type: str,
    fixed_threshold: int = 2,
    user_thresholds: pd.DataFrame = None,
) -> Dict[int, Set[int]]:
    # computes negative item sets for every user in train_df at once
    # result is {userId: set(negative_movieIds)computed once, looked up per user during eval
    return {
        user_id: get_user_negative_items(
            user_id, train_df, threshold_type, fixed_threshold, user_thresholds
        )
        for user_id in train_df["userId"].unique()
    }


def get_all_user_negative_items_with_ratings(
    train_df: pd.DataFrame,
    threshold_type: str,
    fixed_threshold: int = 2,
    user_thresholds: pd.DataFrame = None,
) -> Dict[int, Dict[int, float]]:
    # same as above but includes rating valuesused for weighted penalty variant
    return {
        user_id: get_user_negative_items_with_ratings(
            user_id, train_df, threshold_type, fixed_threshold, user_thresholds
        )
        for user_id in train_df["userId"].unique()
    }


def describe_threshold_stats(
    train_df: pd.DataFrame,
    user_thresholds: pd.DataFrame,
) -> Dict:
    # computes summary statistics for each threshold type
    # useful for thesis documentation of how many ratings each method flags
    # shows: total negative count, percentage, mean per user
    stats = {}
    n_total = len(train_df)

    for thresh in [1, 2, 3]:
        mask = train_df["rating"] <= thresh
        neg_counts = train_df[mask].groupby("userId").size()
        stats[f"fixed_{thresh}"] = {
            "total_negative": int(mask.sum()),
            "pct_negative": round(mask.mean() * 100, 2),
            "mean_per_user": round(neg_counts.mean(), 2),
            "median_per_user": round(neg_counts.median(), 2),
        }

    for col, label in [("median_rating", "median"), ("modus_rating", "modus")]:
        merged = train_df.merge(user_thresholds[["userId", col]], on="userId")
        mask = merged["rating"] < merged[col]
        stats[label] = {
            "total_negative": int(mask.sum()),
            "pct_negative": round(mask.mean() * 100, 2),
        }

    return stats