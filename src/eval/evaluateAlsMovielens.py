from pathlib import Path
import numpy as np
from src.data.load_movielens import load_movielens
from src.models.baseline_cf import train_surprise_svd
from src.eval.metrics import precision_at_k, recall_at_k, ndcg_at_k

def recommend_top_n_for_user(model, userId, all_item_ids, known_items, n=10):
    # model: Surprise SVD model
    # we score every candidate item for this user, except ones they've already interacted with in train
    scores = []
    for itemId in all_item_ids:
        if itemId in known_items:
            continue #Skip items the user already interacted with
        est = model.predict(uid=userId, iid=itemId).est
        scores.append((itemId, est))
    # sort by score high → low
    scores.sort(key=lambda x: x[1], reverse=True)
    return [item for (item, score) in scores[:n]]


def build_user_history(df):#set of items they’ve interacted with in train
    # df: trainDf
    # return dict: userId -> set of items they've seen (train interactions)
    hist = {}
    for row in df.itertuples(index=False):
        u = int(row.userId)
        i = int(row.itemId)
        hist.setdefault(u, set()).add(i)
    return hist

def build_user_ground_truth(df):
    # df: testDf
    # return dict: userId -> set of held-out "true next items"
    truth = {}
    for row in df.itertuples(index=False):
        u = int(row.userId)
        i = int(row.itemId)
        truth.setdefault(u, set()).add(i)
    return truth

def main():
    trainDf, testDf = load_movielens()
    model = train_surprise_svd(trainDf)

    user_history = build_user_history(trainDf)
    user_truth   = build_user_ground_truth(testDf)

    all_items = np.sort(trainDf["itemId"].unique())

    p_list, r_list, ndcg_list = [], [], []

    for userId in user_truth.keys():
        recs = recommend_top_n_for_user(
            model,
            userId,
            all_items,
            known_items=user_history.get(userId, set()),
            n=10
        )

        relevant = user_truth[userId]

        p_list.append(precision_at_k(recs, relevant, k=10))
        r_list.append(recall_at_k(recs, relevant, k=10))
        ndcg_list.append(ndcg_at_k(recs, relevant, k=10))

    print("Baseline CF model (positive-only):")
    print(" P@10  =", sum(p_list)/len(p_list))
    print(" R@10  =", sum(r_list)/len(r_list))
    print(" nDCG@10 =", sum(ndcg_list)/len(ndcg_list))

if __name__ == "__main__":
    main()
