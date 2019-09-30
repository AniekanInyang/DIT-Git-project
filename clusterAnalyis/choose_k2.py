import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage


df = pd.read_csv('Mall_Customers.csv')


data = df.values
np.random.shuffle(data)
X2 = data[:, 3:]

    #normalizing columns
for i in (0, 1):
    m = X2[:, i].mean()
    sd = X2[:, i].std()
    X2[:, i] = (X2[:, i] - m) / sd


# what does it look like without clustering?
plt.scatter(X2[:, 0], X2[:, 1])
plt.title("Scatter plot based on income  and Spending score without clustering")
plt.show()

#using hierachical clusters to get the number of clusters which will serve as our value for k
Z = linkage(X2, 'ward')
plt.title("Cluster Analysis based on income and Spending score using Ward's criterion")
dendrogram(Z)
plt.show()