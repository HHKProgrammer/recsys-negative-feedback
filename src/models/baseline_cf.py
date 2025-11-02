from surprise import SVD, Dataset, Reader
import pandas as pd
#vanilla recommender that only models ratings
def train_surprise_svd(trainDf):
    # trainDf columns: userId, itemId, rating, timestamp
    reader = Reader(rating_scale=(1, 5))#ratings 1-5
    train_data = Dataset.load_from_df(#we dont use timestamp 
        trainDf[["userId", "itemId", "rating"]],
        reader
    )
    trainset = train_data.build_full_trainset()#own index mapping sparese matrix
    
    algo = SVD(#Singular Value Decomposition (SVD) algorithm
        n_factors=50,#This means we're learning 50 latent dimensions to represent user preferences and item characteristics. I chose 50 based on literature i will mention in thesis and my notebook. Too few (e.g., 10) underfits; too many (e.g., 200) overfits on this dataset size.
        n_epochs=20,#I monitored training and validation error and found that performance stabilized around 15–20 epochs. More epochs didn’t improve accuracy.in my notebook
        #one full pass through training data
        lr_all=0.005,#“This controls how fast the model updates its parameters. A rate of 0.005 balances speed and stability. Higher values risk overshooting; lower values slow convergence
        #stepsize
        reg_all=0.02 #This is the λ in the loss function. It penalizes large weights to keep the model generalizable. I tuned this using cross-validation.
    )
    algo.fit(trainset)
    return algo
