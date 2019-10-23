import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage


df = pd.read_csv('Mall_Customers.csv')


data = df.values
np.random.shuffle(data)
X = data[:, 2::2]

    #normalizing columns
for i in (0, 1):
    m = X[:, i].mean()
    sd = X[:, i].std()
    X[:, i] = (X[:, i] - m) / sd


# what does it look like without clustering?
plt.scatter(X[:, 0], X[:, 1])
plt.title("Scatter plot based on Age and Spending score without clustering")
plt.show()

#using hierachical clusters to get the number of clusters which will serve as our value for k
Z = linkage(X, 'ward')
plt.title("Cluster Analysis based on Age and Spending score using Ward's criterion")
dendrogram(Z)
plt.show()