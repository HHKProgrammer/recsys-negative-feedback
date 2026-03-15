# pipeline step 6 — evaluation metrics
# measures how well the model predicts the next item a user will choose
# all metrics computed per user, then averaged over all 6040 test users
# reference jarvelin kekalainen 2002, cumulated gain-based evaluation of ir techniques
# https://dl.acm.org/doi/10.1145/582415.582418

import numpy as np

# P@10, R@10, nDCG@10

#Goal For each user in test, could the model have guessed the held-out item?

#  quantitative result

#We will later compare this baseline vs a model that understands negative feedback.

#how many of the top 10 are relevant
def precision_at_k(recommended_items, relevant_items, k=10):
    # recommended_items: list of itemIds in rank order (top-first)
    # relevant_items: set of "true" items for that user (usually 1 item in your split)
    # with 1 relevant item per user: p@10 = 0.1 if hit, 0.0 if miss
    # so p@10 = hit_rate / 10 in this setup
    hit_count = sum([1 for i in recommended_items[:k] if i in relevant_items])
    return hit_count / k

#did we find the thing you actually chose next Did I miss something important that I should have recommended?
def recall_at_k(recommended_items, relevant_items, k=10):
    # with only 1 relevant item per user: recall = 1.0 if found, 0.0 if not
    # identical to hit@k in this single-item evaluation setup
    if len(relevant_items) == 0:
        return 0.0
    hit_count = sum([1 for i in recommended_items[:k] if i in relevant_items])
    return hit_count / len(relevant_items)

#did we rank it high or bury it If the relevant item is rank 1, you get a high reward.
def ndcg_at_k(recommended_items, relevant_items, k=10):
    # position-sensitive metric — rewards finding the item at high rank
    # rank 1 -> ndcg ~1.0, rank 5 -> ~0.565, rank 10 -> ~0.442, not found -> 0.0
    # formula: dcg = sum(1/log2(rank+1)) for each relevant item in top-k
    # normalized by idcg (best possible dcg) so result is 0 to 1
    # reference  jarvelin kekalainen 2002
    # https://dl.acm.org/doi/10.1145/582415.582418

    # DCG: reward hits more if they are ranked high (1/log2(rank+2))
    # rank starts at 0 in enumerate, so +2 gives 1-based formula equivalent
    dcg = 0.0
    for rank, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            dcg += 1 / np.log2(rank + 2)

    # IDCG: best possible DCG for this user
    # (if all relevant items were ranked at the top)
    idcg = 0.0
    for ideal_rank in range(min(len(relevant_items), k)):
        idcg += 1 / np.log2(ideal_rank + 2)

    return dcg / idcg if idcg > 0 else 0.0


def negative_at_k(recommended_items, negative_items, k=10):
    # Count how many known-disliked items appear in the top-K list.
    # always 0 in this experiment because candidates = unseen items and
    # negatives = already-rated items they can never overlap structurally
    # kept for completeness and for setups where this would matter
    return sum(1 for item in recommended_items[:k] if item in negative_items)


def sim_to_neg_at_k(recommended_items, negative_items, similarity_fn, k=10):
    # average maximum cosine similarity between each top-K item and the users
    # negative items in latent space

    # measures whether recommendations are "close to" disliked content even when
    # the exact negative items cannot appear in the candidate pool (because
    # candidates are unseen items, and negatives are already-rated items

    # lower is better: variants that successfully push recommendations away from
    # the negative neighbourhood should produce lower values than the baseline

    # this is my key metric for showing the variants actually do something different
    # even when negative@k is always zero

    # Returns 0.0 if there are no negative items or no similarity function.
    if not negative_items or similarity_fn is None:
        return 0.0
    negative_list = list(negative_items)
    scores = []
    for item in recommended_items[:k]:
        max_sim = max(
            (similarity_fn(item, neg) for neg in negative_list),
            default=0.0,
        )
        scores.append(max_sim)
    return float(np.mean(scores)) if scores else 0.0


def hit_at_k(recommended_items, target_item, k=10):
    # 1 if target_item is in the top-K list, else 0
    # binary  did the model find the right item at all?
    # random baseline would hit ~2% of the time (10/501 candidates)
    # my best result 8.69% -> ~4x better than random
    return 1 if target_item in recommended_items[:k] else 0


def reciprocal_rank(recommended_items, target_item):
    # Reciprocal of the rank position of target_item (0.0 if not found)
    # rank 1 -> 1.0, rank 2 -> 0.5, rank 10 -> 0.1, not in list -> 0.0
    # mean reciprocal rank (mrr) across users tells how high the true item ranked on average
    try:
        rank = recommended_items.index(target_item) + 1  # 1-based rank
        return 1.0 / rank
    except ValueError:
        return 0.0