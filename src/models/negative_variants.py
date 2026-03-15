# pipeline step 5 — negative feedback variants
# three different strategies for using disliked items to improve ranking
# all three wrap SVDBaseline — they dont retrain, only change the ranking at query time
# the key idea: use cosine similarity in latent space to measure how close a candidate
# is to items the user already hated, then penalize accordingly

# Negative feedback handling variants.
# All three wrap SVDBaseline and accept a set/dict of items the user disliked
#
# Variant A FilterNegatives
#   Remove known-disliked items from the candidate pool before ranking
#
# Variant B RerankPenalty
#   score = predicted_rating - alpha * max_similarity_to_any_negative_item
#
# Variant C  WeightedPenalty
#   Like B but weights each negative item's contribution by how negative the
#   rating was: weight = (max_rating - rating) / max_rating
#   A rating of 1 penalises more than a rating of 2

from typing import Dict, List, Set, Tuple

import numpy as np

from src.models.svd_baseline import SVDBaseline


class FilterNegatives:
    # Variant A: delete disliked items from candidates, then rank normally.
    # note: in this experiment setup this has no effect because candidates are
    # already unseen items and negatives are rated items they cant overlap
    # would be useful in production where a pre-built index includes seen items

    def __init__(self, baseline: SVDBaseline):
        self.baseline = baseline

    def rank_items_for_user(
        self,
        user_id: int,
        candidate_items: List[int],
        negative_items: Set[int],
    ) -> List[Tuple[int, float]]:
        # remove any candidate that appears in the user's dislike list
        filtered = [i for i in candidate_items if i not in negative_items]
        return self.baseline.rank_items_for_user(user_id, filtered)


class RerankPenalty:
    # Variant B: penalise items similar to the user's disliked items
    # score(u, i) = r_hat(u,i) - alpha * max_{n in N_u} cos(qi, qn)
    # r_hat(u,i) predicition, alpha stronger weaker(weight), cos(qi, qn) cosine similarty
    # score = predicted_rating - alpha * max(sim(item, neg) for neg in negatives)

    # uses max() most conservative, worst-case similarity drives the penalty
    # this can be aggressive: one similar hated item penalizes heavily
    # alpha=1.0 often hurts performance because it overwrites the svd ranking entirely

    def __init__(self, baseline: SVDBaseline, alpha: float = 0.3):
        self.baseline = baseline
        self.alpha = alpha  # penalty strength — 0.1 gentle, 1.0 aggressive

    def _penalty(self, item_id: int, negative_items: Set[int]) -> float:
        # max cosine similarity between this candidate and any of the user's negative items
        if not negative_items:
            return 0.0
        return max(self.baseline.get_similarity(item_id, neg) for neg in negative_items)

    def rank_items_for_user(
        self,
        user_id: int,
        candidate_items: List[int],
        negative_items: Set[int],
    ) -> List[Tuple[int, float]]:
        scores = []
        for item in candidate_items:
            pred = self.baseline.predict(user_id, item)
            penalty = self._penalty(item, negative_items)
            # adjusted score items near hated items get pushed down the list
            scores.append((item, pred - self.alpha * penalty))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class WeightedPenalty:
    # Variant C: like RerankPenalty but weight each negative by how bad its rating wa
    # weight(neg) = (max_rating - rating) / max_rating
    # penalty = mean over all negatives of: sim(item, neg) * weight(neg)
    # score = predicted_rating - alpha * penalty
    # weight(n) = (R_max - r_n) / R_max        R_max = 5.0
    #
    # penalty(u,i) = (1/|N_u|) * sum_{n in N_u} [cos(qi, qn) * weight(n)]
    # score(u,i) = r_hat(u,i) - alpha * penalty(u,i)

    # uses mean() instead of max()  smoother, less extreme than rerank
    # a single very similar hated item doesnt dominate  averaged with all others
    # rating severity matters: 1-star (weight=0.8) contributes more than 3-star (weight=0.4
    # this is why weighted_fixed_3_a0.1 is the best result  many negatives, gentle penalty

    def __init__(
        self,
        baseline: SVDBaseline,
        alpha: float = 0.3,
        max_rating: float = 5.0,
    ):
        self.baseline = baseline
        self.alpha = alpha
        self.max_rating = max_rating  # 5.0 for ml-1m rating scale

    def _penalty(
        self, item_id: int, negative_items_with_ratings: Dict[int, float]
    ) -> float:
        # weighted mean of (similarity * severity_weight) across all negative items
        # weight examples: rating=1 -> (5-1)/5=0.8, rating=2 -> 0.6, rating=3 -> 0.4
        if not negative_items_with_ratings:
            return 0.0
        contributions = []
        for neg_item, rating in negative_items_with_ratings.items():
            sim = self.baseline.get_similarity(item_id, neg_item)
            weight = (self.max_rating - rating) / self.max_rating
            contributions.append(sim * weight)
        return float(np.mean(contributions))

    def rank_items_for_user(
        self,
        user_id: int,
        candidate_items: List[int],
        negative_items_with_ratings: Dict[int, float],  # {movieId: rating}
    ) -> List[Tuple[int, float]]:
        scores = []
        for item in candidate_items:
            pred = self.baseline.predict(user_id, item)
            penalty = self._penalty(item, negative_items_with_ratings)
            scores.append((item, pred - self.alpha * penalty))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores