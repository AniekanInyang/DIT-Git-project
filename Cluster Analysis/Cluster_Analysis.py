import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage


"""
QUESTION 1

The dataset contains customers spending habits in a mall. The major features of the customers are their age, income 
and spending score. The test is to do a cluster analysis on the dataset based on these features
1.	Age and Spending score

ANSWER
======
There are only 2 clusters for the dataset based on Age and Spending score from the dendogram
"""

data = pd.read_csv('Mall_Customers.csv', usecols=["Age", "Spending Score (1-100)"])
X = data.values
np.random.shuffle(X)

# normalizing the data
mm_scaler = preprocessing.MinMaxScaler()
X = mm_scaler.fit_transform(X)

# what does it look like without clustering?
plt.scatter(X[:, 0], X[:, 1])
plt.title("Scatter plot based on Age and Spending score without clustering")
plt.show()

Z = linkage(X, 'ward')
plt.title("Cluster Analysis based on Age and Spending score using Ward's criterion")
dendrogram(Z)
plt.show()

"""
QUESTION 2

The dataset contains customers spending habits in a mall. The major features of the customers are their age, income 
and spending score. The test is to do a cluster analysis on the dataset based on these features
2.	Income and Spending score.

ANSWER
======
There are only 3 clusters for the dataset based on Income and Spending score from the dendogram
"""

data2 = pd.read_csv('Mall_Customers.csv', usecols=["Annual Income (k$)", "Spending Score (1-100)"])
X2 = data2.values
np.random.shuffle(X2)

# normalizing the data
mm_scaler = preprocessing.MinMaxScaler()
X2 = mm_scaler.fit_transform(X2)

# what does it look like without clustering?
plt.scatter(X2[:, 0], X2[:, 1])
plt.title("Scatter plot based on Income and Spending score without clustering")
plt.show()

Z2 = linkage(X2, 'ward')
plt.title("Cluster Analysis based on Income and Spending Score using Ward's criterion")
dendrogram(Z2)
plt.show()


"""
QUESTION 3

The dataset contains customers spending habits in a mall. The major features of the customers are their age, income 
and spending score. The test is to do a cluster analysis on the dataset based on these features
3.	Age, Income and Spending Score

ANSWER
======
There are only 2 clusters for the dataset based on Age, Income and Spending Score from the dendogram
"""

data3 = pd.read_csv('Mall_Customers.csv', usecols=["Age", "Annual Income (k$)", "Spending Score (1-100)"])
X3 = data3.values
np.random.shuffle(X3)

# normalizing the data
mm_scaler = preprocessing.MinMaxScaler()
X3 = mm_scaler.fit_transform(X3)

Z3 = linkage(X3, 'ward')
plt.title("Cluster Analysis based on Age, Income and Spending Score using Ward's criterion")
dendrogram(Z3)
plt.show()
