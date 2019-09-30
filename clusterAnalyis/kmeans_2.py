import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def d(u, v):
    diff = u - v
    return diff.dot(diff)


def plot_k_means(X2, K, max_iter=20, beta=3.0):
    N, D = X2.shape
    M = np.zeros((K, D))
    R = np.zeros((N, K))

    for k in range(K):
        M[k] = X2[np.random.choice(N)]

    for i in range(max_iter):
        for k in range(K):
            for n in range(N):
                R[n, k] = np.exp(-beta * d(M[k], X2[n])) / np.sum(np.exp(-beta * d(M[j], X2[n])) for j in range(K))

        for k in range(K):
            M[k] = R[:, k].dot(X2) / R[:, k].sum()

    random_colours = np.random.random((K, 3))
    colors = R.dot(random_colours)
    plt.scatter(X2[:, 0], X2[:, 1], c=colors)
    plt.title('Income and Spending Score clusters')
    plt.show()


def get_data():
    df = pd.read_csv('Mall_Customers.csv')
    data = df.values
    np.random.shuffle(data)

    X2 = data[:, 3:]

    # normalizing columns
    for i in (0, 1):
        m = X2[:, i].mean()
        sd = X2[:, i].std()
        X2[:, i] = (X2[:, i] - m) / sd

    return X2


def main():
    X2 = get_data()
    plt.scatter(X2[:, 0], X2[:, 1])
    plt.title('Income and Spending score without cluster')
    plt.show()

    K = 3
    plot_k_means(X2, K)


if __name__ == '__main__':
    main()
