# conda activate recsys-conda
#pip install matplotlib seaborn
#conda install -c conda-forge matplotlib -y
#conda install -c conda-forge matplotlib scikit-learn -y

from surprise import SVD, Dataset, Reader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def train_surprise_svd(trainDf):
    """
    Train a standard baseline CF model (no negative feedback).

    Input:
        trainDf: DataFrame with columns [userId, itemId, rating, timestamp]
    Output:
        Trained Surprise SVD model
    """

    # Reader converts pandas DataFrame to Surprise dataset (ratings 1–5)
    reader = Reader(rating_scale=(1, 5))

    # Only use userId, itemId, rating for model training
    train_data = Dataset.load_from_df(
        trainDf[["userId", "itemId", "rating"]],
        reader
    )

    # Build internal index mappings (users/items → numeric ids)
    trainset = train_data.build_full_trainset()

    # Tuned hyperparameters in my notbook 03:hyperparameters
    algo = SVD(
        n_factors=150,  # latent dimensions 
        n_epochs=40,    # full passes through data
        lr_all=0.01,    # learning rate for gradient descent
        reg_all=0.1     # regularization (prevents overfitting)
    )

    # Fit model on training data
    algo.fit(trainset)
    return algo
