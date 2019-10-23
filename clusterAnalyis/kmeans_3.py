import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def d(u, v):
    diff = u - v
    return diff.dot(diff)


def plot_k_means(X3, K, max_iter=20, beta=3.0):
    N, D = X3.shape
    M = np.zeros((K, D))
    R = np.zeros((N, K))

    for k in range(K):
        M[k] = X3[np.random.choice(N)]

    for i in range(max_iter):
        for k in range(K):
            for n in range(N):
                R[n, k] = np.exp(-beta * d(M[k], X3[n])) / np.sum(np.exp(-beta * d(M[j], X3[n])) for j in range(K))

        for k in range(K):
            M[k] = R[:, k].dot(X3) / R[:, k].sum()

    random_colours = np.random.random((K, 3))
    colors = R.dot(random_colours)
    plt.scatter(X3[:, 0], X3[:, 1], c=colors)
    plt.title('Age, Income and Spending Score clusters')
    plt.show()


def get_data():
    df = pd.read_csv('Mall_Customers.csv')
    data = df.values
    np.random.shuffle(data)

    X3 = data[:, 2:]

    # normalizing columns
    for i in (0, 1,2):
        m = X3[:, i].mean()
        sd = X3[:, i].std()
        X3[:, i] = (X3[:, i] - m) / sd

    return X3


def main():
    X3 = get_data()
    plt.scatter(X3[:, 0], X3[:, 1])
    plt.title('Income and Spending score without cluster')
    plt.show()

    K = 2
    plot_k_means(X3, K)


if __name__ == '__main__':
    main()
