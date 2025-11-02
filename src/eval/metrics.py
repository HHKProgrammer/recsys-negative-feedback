import numpy as np
# P@10, R@10, nDCG@10

#Goal For each user in test, could the model have guessed the held-out item?

#  quantitative result

#We will later compare this baseline vs a model that understands negative feedback.

#how many of the top 10 are relevant
def precision_at_k(recommended_items, relevant_items, k=10):
    # recommended_items: list of itemIds in rank order (top-first)
    # relevant_items: set of "true" items for that user (usually 1 item in your split)
    hit_count = sum([1 for i in recommended_items[:k] if i in relevant_items])
    return hit_count / k

#did we find the thing you actually chose next Did I miss something important that I should have recommended?
def recall_at_k(recommended_items, relevant_items, k=10):
    if len(relevant_items) == 0:
        return 0.0
    hit_count = sum([1 for i in recommended_items[:k] if i in relevant_items])
    return hit_count / len(relevant_items)

#did we rank it high or bury it If the relevant item is rank 1, you get a high reward.
def ndcg_at_k(recommended_items, relevant_items, k=10):
    # DCG: reward hits more if they are ranked high (1/log2(rank+2))
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
