# https://deeplearningcourses.com/c/cluster-analysis-unsupervised-machine-learning-python
# https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


#calculating squared distance
def d(u, v):
    diff = u - v
    return diff.dot(diff)


def cost(X, R, M):
    cost = 0
    for k in range(len(M)):
        # method 1
        # for n in range(len(X)):
        #     cost += R[n,k]*d(M[k], X[n])

        # method 2
        diff = X - M[k]
        sq_distances = (diff * diff).sum(axis=1)
        cost += (R[:,k] * sq_distances).sum()
    return cost


def plot_k_means(X, K, max_iter=20, beta=1.0, show_plots=False):
    N, D = X.shape          #we get the shape of X
    M = np.zeros((K, D))     #mean will be initialize to zero
    # R = np.zeros((N, K))
    exponents = np.empty((N, K))  # responsibility matrix 

    # initialize M to random point X
    for k in range(K):
        M[k] = X[np.random.choice(N)]

    costs = []
    k = 0
    for i in range(max_iter):
        k += 1
        # step 1: determine assignments / resposibilities
        # is this inefficient?
        for k in range(K):
            for n in range(N):
                exponents[n,k] = np.exp(-beta*d(M[k], X[n]))
        R = exponents / exponents.sum(axis=1, keepdims=True)


        # step 2: recalculate means
        # decent vectorization
        # for k in range(K):
        #     M[k] = R[:,k].dot(X) / R[:,k].sum()
        # oldM = M

        # full vectorization
        M = R.T.dot(X) / R.sum(axis=0, keepdims=True).T
        # print("diff M:", np.abs(M - oldM).sum())

        c = cost(X, R, M)
        costs.append(c)
        if i > 0:
            if np.abs(costs[-1] - costs[-2]) < 1e-5:  # checking if the cost has not change that much between the two iterations
                break

        if len(costs) > 1:
            if costs[-1] > costs[-2]:
                pass
                # print("cost increased!")
                # print("M:", M)
                # print("R.min:", R.min(), "R.max:", R.max())

    if show_plots:
        plt.plot(costs)
        plt.title("Costs")
        plt.show()

        random_colors = np.random.random((K, 3))
        colors = R.dot(random_colors)
        plt.scatter(X[:,0], X[:,1], c=colors)
        plt.show()

    print("Final cost", costs[-1])
    return M, R


def get_simple_data():
    df = pd.read_csv("Mall_Customers.csv").dropna()
    df.drop(columns =["CustomerID", "Gender"], inplace=True)
    new_data = df[["Spending Score (1-100)", "Age", "Annual Income (k$)"]]
#     data = new_data.values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(new_data)
    np.random.shuffle(data_scaled)
    # assume 3 means
    #creating 3 guassian class
    D = 2 # so we can visualize it more easily
    s = 4 # separation so we can control how far apart the means are
    mu1 = np.array([0, 0]) #this is going to be the origin
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 200 # number of samples
    X = data_scaled[:, 1:]
    X[:68, :] = np.random.randn(68, D) + mu1
    X[68:134, :] = np.random.randn(66, D) + mu2
    X[134:, :] = np.random.randn(66, D) + mu3
    return X


def main():
    X = get_simple_data()

    # what does it look like without clustering?
    plt.scatter(X[:,0], X[:,1])
    plt.show()

    K = 3 
    plot_k_means(X, K, beta=1.0, show_plots=True)

    K = 3 
    plot_k_means(X, K, beta=3.0, show_plots=True)

    K = 3 
    plot_k_means(X, K, beta=10.0, show_plots=True)

    K = 5 # what happens if we choose a "bad" K?
    plot_k_means(X, K, max_iter=30, show_plots=True)

    K = 5 # what happens if we change beta?
    plot_k_means(X, K, max_iter=30, beta=0.3, show_plots=True)

    


if __name__ == '__main__':
    main()
