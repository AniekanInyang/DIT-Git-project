import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#function to get the squared distance
def d(u, v):
    diff = u - v
    return diff.dot(diff)



def plot_k_means(X, K, max_iter=20, beta=3.0):
    N, D = X.shape   #get the samples and feature length of X
    M = np.zeros((K, D)) #   mean to be K samples and D dimentions
    R = np.zeros((N, K))

    for k in range(K):
        M[k] = X[np.random.choice(N)] #initialize mean to random points in X

    #costs = np.zeros(max_iter)
    for i in range(max_iter):
        for k in range(K):
            for n in range(N):
                R[n, k] = np.exp(-beta * d(M[k], X[n])) / np.sum(np.exp(-beta * d(M[j], X[n])) for j in range(K))

        for k in range(K): #recalculate the means
            M[k] = R[:, k].dot(X) / R[:, k].sum()

        

    random_colours = np.random.random((K, 3))
    colors = R.dot(random_colours)
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.title('Age and Spending Score')
    plt.show()

def get_data():
    df = pd.read_csv('Mall_Customers.csv')
    data = df.values
    np.random.shuffle(data)

    X = data[:, 2::2]



    #normalizing columns  using mean and Standard deviation
    for i in (0, 1):
        m = X[:, i].mean()
        sd = X[:, i].std()
        X[:, i] = (X[:, i] - m) / sd

    return X



def main():
    X=get_data()
    plt.scatter(X[:, 0], X[:, 1])
    plt.title('Age and Spending score without cluster')
    plt.show()


    K=2
    plot_k_means(X,K)

if __name__ == '__main__':
    main()
