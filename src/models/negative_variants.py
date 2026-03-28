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

    def rank_items_for_user(
        self,
        user_id: int,
        candidate_items: List[int],
        negative_items: Set[int],
    ) -> List[Tuple[int, float]]:
        # batch predict all candidates at once one matmul instead of 500 calls
        preds = self.baseline.predict_batch(user_id, candidate_items)

        if not negative_items:
            scores = list(zip(candidate_items, preds.tolist()))
        else:
            # sim_matrix shape: (n_candidates, n_negatives)
            # each row = one candidate, each column = one hated item
            neg_list = list(negative_items)
            sim_matrix = self.baseline.similarity_matrix_batch(candidate_items, neg_list)
            # penalty per candidate = max similarity to any hated item
            penalties = sim_matrix.max(axis=1)  # shape (n_candidates,)
            adjusted = preds - self.alpha * penalties
            scores = list(zip(candidate_items, adjusted.tolist()))

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

    def rank_items_for_user(
        self,
        user_id: int,
        candidate_items: List[int],
        negative_items_with_ratings: Dict[int, float],  # {movieId: rating}
    ) -> List[Tuple[int, float]]:
        # batch predict all candidates at once
        preds = self.baseline.predict_batch(user_id, candidate_items)

        if not negative_items_with_ratings:
            scores = list(zip(candidate_items, preds.tolist()))
        else:
            neg_items = list(negative_items_with_ratings.keys())
            # weight per negative: 1-star -> 0.8, 2-star -> 0.6, 3-star -> 0.4
            weights = np.array(
                [(self.max_rating - negative_items_with_ratings[n]) / self.max_rating
                 for n in neg_items],
                dtype=np.float32,
            )

            # sim_matrix shape: (n_candidates, n_negatives)
            sim_matrix = self.baseline.similarity_matrix_batch(candidate_items, neg_items)

            # penalty per candidate = mean(sim * weight) across all negatives
            # (n_candidates, n_negatives) @ (n_negatives,) = (n_candidates,)
            penalties = (sim_matrix @ weights) / len(neg_items)

            adjusted = preds - self.alpha * penalties
            scores = list(zip(candidate_items, adjusted.tolist()))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores