"""Unit tests for src/data/threshold_utils.py"""
import pandas as pd
import pytest
from src.data.threshold_utils import (
    get_user_negative_items,
    get_user_negative_items_with_ratings,
)


def make_train():
    return pd.DataFrame({
        "userId": [1, 1, 1, 1, 1],
        "movieId": [10, 20, 30, 40, 50],
        "rating": [1.0, 2.0, 3.0, 4.0, 5.0],
        "timestamp": [1, 2, 3, 4, 5],
    })


def make_thresholds(median=3.0, modus=4.0):
    return pd.DataFrame({
        "userId": [1],
        "median_rating": [median],
        "modus_rating": [modus],
        "mean_rating": [3.0],
        "count": [5],
    })


# ── Fixed threshold (<=) ─────────────────────────────────────────────────────

def test_fixed_threshold_1():
    neg = get_user_negative_items(1, make_train(), "fixed", fixed_threshold=1)
    assert neg == {10}  # only rating=1


def test_fixed_threshold_2():
    neg = get_user_negative_items(1, make_train(), "fixed", fixed_threshold=2)
    assert neg == {10, 20}  # ratings 1 and 2


def test_fixed_threshold_3():
    neg = get_user_negative_items(1, make_train(), "fixed", fixed_threshold=3)
    assert neg == {10, 20, 30}  # ratings 1, 2, 3


def test_fixed_threshold_rating3_is_negative_at_threshold3():
    # rating=3, threshold=3 → 3 <= 3 → IS negative
    assert 30 in get_user_negative_items(1, make_train(), "fixed", fixed_threshold=3)


def test_fixed_threshold_rating3_not_negative_at_threshold2():
    # rating=3, threshold=2 → 3 <= 2 → NOT negative
    assert 30 not in get_user_negative_items(1, make_train(), "fixed", fixed_threshold=2)


# ── Adaptive thresholds (<) ──────────────────────────────────────────────────

def test_median_threshold():
    # median=3.0 → rating < 3 → negatives are ratings 1 and 2
    neg = get_user_negative_items(
        1, make_train(), "median", user_thresholds=make_thresholds(median=3.0)
    )
    assert neg == {10, 20}


def test_median_threshold_exact_not_negative():
    # rating=3, median=3.0 → 3 < 3 is False → NOT negative
    neg = get_user_negative_items(
        1, make_train(), "median", user_thresholds=make_thresholds(median=3.0)
    )
    assert 30 not in neg


def test_modus_threshold():
    # modus=4.0 → rating < 4 → negatives are 1,2,3
    neg = get_user_negative_items(
        1, make_train(), "modus", user_thresholds=make_thresholds(modus=4.0)
    )
    assert neg == {10, 20, 30}


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_user_with_all_high_ratings_has_no_negatives():
    df = pd.DataFrame({
        "userId": [2, 2, 2],
        "movieId": [1, 2, 3],
        "rating": [5.0, 5.0, 5.0],
        "timestamp": [1, 2, 3],
    })
    neg = get_user_negative_items(2, df, "fixed", fixed_threshold=2)
    assert neg == set()


def test_unknown_user_returns_empty_set():
    neg = get_user_negative_items(
        999, make_train(), "median", user_thresholds=make_thresholds()
    )
    assert neg == set()


# ── Ratings dict variant ──────────────────────────────────────────────────────

def test_negative_items_with_ratings():
    d = get_user_negative_items_with_ratings(
        1, make_train(), "fixed", fixed_threshold=2
    )
    assert d == {10: 1.0, 20: 2.0}
