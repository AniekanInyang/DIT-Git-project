# https://deeplearningcourses.com/c/cluster-analysis-unsupervised-machine-learning-python
# https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python

# data is from https://www.kaggle.com/c/digit-recognizer
# each image is a D = 28x28 = 784 dimensional vector
# there are N = 42000 samples
# you can plot an image by reshaping to (28,28) and using plt.imshow()

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util_3 import plot_k_means, get_simple_data
from datetime import datetime

def get_data(limit=None):
    print("Reading in and transforming data...")
    df = pd.read_csv("Mall_Customers.csv").dropna()
    df.drop(columns =["CustomerID", "Gender", "Age"], inplace=True)
    new_df = df[["Annual Income (k$)", "Spending Score (1-100)"]]
#     new_male_df = df["Gender"] == "Male"
#     new_female_df = df["Gender"] == "Female"
#     df.loc[new_male_df, "Gender"] = 1
#     df.loc[new_female_df, "Gender"] = 0
    data = new_df.values
#     scaler = MinMaxScaler()
#     data_scaled = scaler.fit_transform(data)
    np.random.shuffle(data)
    X = data[:, 0:] / 99
    Y = data[:, 1]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y


# hard labels
def purity2(Y, R):
    # maximum purity is 1, higher is better
    C = np.argmax(R, axis=1) # cluster assignments

    N = len(Y) # number of data pts
    K = len(set(Y)) # number of labels

    total = 0.0
    for k in range(K):
        max_intersection = 0
        for j in range(K):
            intersection = ((C == k) & (Y == j)).sum()
            if intersection > max_intersection:
                max_intersection = intersection
        total += max_intersection
    return total / N

#this requires the cluster assignments which are the responsibility and Y which is the label 
def purity(Y, R):
    # maximum purity is 1, higher is better
    N, K = R.shape
    p = 0
    for k in range(K):    #we are looping K through all the clusters and J through all the target labels both are from 0 to k
        best_target = -1 # we don't strictly need to store this
        max_intersection = 0    #we get the intersection by looking at the R matrix, remember that the shape of R is N by K 
        for j in range(K):
            intersection = R[Y==j, k].sum()   #we want only rows which correspond to the target labels which is J, that is what the first index is
                                              ##the second index is which cluster K we are currently looking at
                                               #finally we take the sum of all these responsibilities since that is how much that data points belongs to this cluster
                    #Note that in the case of hard k-means where R only contains the value of 0 or 1, this equation is still valid
            if intersection > max_intersection: 
                max_intersection = intersection 
                best_target = j   #we find the best j corresponding to the best intersection and add to the final purity
        p += max_intersection
    return p / N   # the last step is to divide by N so that it's independent of the number of data poits

#we need all the data points X and the means M and the responsiblities R
# hard labels
def DBI2(X, R):
    N, D = X.shape
#     print("DBI2", N, D)
    _, K = R.shape
#     print("DBI2", _, K)

    # get sigmas, means first
    sigma = np.zeros(K)
#     print(sigma)
    M = np.zeros((K, D))
    
    assignments = np.argmax(R, axis=1)
#     print("DBI 2", assignments)
    for k in range(K):    #the first loop is to calculate the sigmas, this is the average distance between all
                          #the datapoints in the cluster K from the center, but since every point could be part
                        #of this cluster we need to use all of that, we can then weigh it by our K later
        Xk = X[assignments == k]
        
        M[k] = Xk.mean(axis=0)
        
        # assert(Xk.mean(axis=0).shape == (D,))
        n = len(Xk)
        diffs = Xk - M[k]
        sq_diffs = diffs * diffs
        sigma[k] = np.sqrt( sq_diffs.sum() / n )

    #the second loop actually calculate the Davies-Bouldin Index using the sigmas we calculated previously
    # calculate Davies-Bouldin Index
    dbi = 0
    for k in range(K):
        max_ratio = 0
        for j in range(K):
            if k != j:
                numerator = sigma[k] + sigma[j]
                denominator = np.linalg.norm(M[k] - M[j])  #distance between cluster center K and cluster center J
                ratio = numerator / denominator  #WE DIVIDE, notice that even though this equation involves K remeber that lower is better
                                                    # meaning that if K is very high we will get a better score, but ths doesn't save us from 
                                                #from reaching the trivial case of where K equals N every data point is its own cluster
                if ratio > max_ratio:
                    max_ratio = ratio
        dbi += max_ratio
    return dbi / K



def DBI(X, M, R):
    # ratio between sum of std deviations between 2 clusters / distance between cluster means
    # lower is better
    N, D = X.shape
    K, D = M.shape
#     print("DBI", K, D)

    # get sigmas first
    sigma = np.zeros(K)
#     print(sigma)
    for k in range(K):
        diffs = X - M[k] # should be NxD
        
        squared_distances = (diffs * diffs).sum(axis=1) # now just N
        
        weighted_squared_distances = R[:,k]*squared_distances
        sigma[k] = np.sqrt( weighted_squared_distances.sum() / R[:,k].sum() )
        
    # calculate Davies-Bouldin Index
    dbi = 0
    for k in range(K):
        max_ratio = 0
        for j in range(K):
            if k != j:
                numerator = sigma[k] + sigma[j]
                
                denominator = np.linalg.norm(M[k] - M[j])
#                 print(denominator)
                if denominator == 0.0:
                    denominator = 1e-5
                else:
                    ratio = numerator / denominator
#                 print(ratio)
                if ratio > max_ratio:
                    max_ratio = ratio
        dbi += max_ratio
#         print(dbi)
    return dbi / K


def main():
    # mnist data
    X, Y = get_data(100)   #we limit our data points to 1000 as k means takes a long time
                            #we pass K equals 10 since there are 10 digits

    print("Number of data points:", len(Y))
    M, R = plot_k_means(X, len(set(Y)))
    # Exercise: Try different values of K and compare the evaluation metrics
    print("Purity:", purity(Y, R))
    print("Purity 2 (hard clusters):", purity2(Y, R))
    print("DBI:", DBI(X, M, R))
    print("DBI 2 (hard clusters):", DBI2(X, R))




if __name__ == "__main__":
    main()
