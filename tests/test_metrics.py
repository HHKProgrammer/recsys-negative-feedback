"""Unit tests for src/eval/metrics.py"""
import pytest
from src.eval.metrics import (
    hit_at_k,
    ndcg_at_k,
    negative_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k_hit():
    assert precision_at_k([1, 2, 3, 4, 5], {3}, k=5) == 1 / 5


def test_precision_at_k_miss():
    assert precision_at_k([1, 2, 3], {9}, k=3) == 0.0


def test_recall_at_k_hit():
    assert recall_at_k([1, 2, 3], {1, 2}, k=3) == 1.0


def test_recall_at_k_miss():
    assert recall_at_k([1, 2, 3], {9}, k=3) == 0.0


def test_recall_at_k_empty_relevant():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


def test_ndcg_at_k_perfect():
    # Relevant item at rank 1 → max nDCG
    assert ndcg_at_k([1, 2, 3], {1}, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_rank2():
    import math
    score = ndcg_at_k([2, 1, 3], {1}, k=3)  # relevant at rank 2
    assert score == pytest.approx(1 / math.log2(3))


def test_negative_at_k():
    assert negative_at_k([1, 2, 3, 4, 5], {2, 4}, k=5) == 2


def test_negative_at_k_none():
    assert negative_at_k([1, 2, 3], {9, 8}, k=3) == 0


def test_hit_at_k_true():
    assert hit_at_k([1, 2, 3], 2, k=3) == 1


def test_hit_at_k_false():
    assert hit_at_k([1, 2, 3], 9, k=3) == 0


def test_reciprocal_rank_found():
    assert reciprocal_rank([5, 3, 1], 3) == pytest.approx(0.5)


def test_reciprocal_rank_not_found():
    assert reciprocal_rank([5, 3, 1], 9) == 0.0
