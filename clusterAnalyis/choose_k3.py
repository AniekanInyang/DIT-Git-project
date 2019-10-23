import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage


df = pd.read_csv('Mall_Customers.csv')
data = df.values
np.random.shuffle(data)
X3 = data[:, 2:]

    #normalizing columns
for i in (0, 1,2):
    m = X3[:, i].mean()
    sd = X3[:, i].std()
    X3[:, i] = (X3[:, i] - m) / sd


# what does it look like without clustering?
plt.scatter(X3[:, 0], X3[:, 1])
plt.title("Scatter plot based on Age Income  and Spending score without clustering")
plt.show()

#using hierachical clusters to get the number of clusters which will serve as our value for k
Z = linkage(X3, 'ward')
plt.title("Cluster Analysis based on Age, income and Spending score using Ward's criterion")
dendrogram(Z)
plt.show()