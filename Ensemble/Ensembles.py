# https://deeplearningcourses.com/c/machine-learning-in-python-random-forest-adaboost
# https://www.udemy.com/machine-learning-in-python-random-forest-adaboost
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle


# create the data 
df = pd.read_csv("data.csv")
data = df.dropna()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
np.random.shuffle(data_scaled)
   
# X = data_scaled[:, :3]
# Y = data_scaled[:, 3]

X = data[['# num_pages', 'ratings_count', 'text_reviews_count']]
Y = data['average_rating']

# get the training data
N = len(data)
T =int(0.7*N)
train_idx = np.random.choice(N, size=T, replace=False)
test_idx = [i for i in range(N) if i not in train_idx]
dfX_train = X.loc[train_idx]
dfY_train = Y.loc[train_idx]
dfX_test = X.loc[test_idx]
dfY_test = Y.loc[test_idx]

Xtrain = scaler.fit_transform(dfX_train)
Ytrain = dfY_train.values.astype(float)
Xtest = scaler.transform(dfX_test)
Ytest = dfY_test.values.astype(float)

model = DecisionTreeRegressor()
model.fit(Xtrain, Ytrain)
prediction = model.predict(Xtest)
print("score for 1 tree:", model.score(Xtest, Ytest))

# now try bagging
class BaggedTreeRegressor:
  def __init__(self, B):
    self.B = B

  def fit(self, X, Y):
    N = len(X)
    self.models = []
    for b in range(self.B):
      idx = np.random.choice(N, size=N, replace=True)
      Xb = X[idx]
      Yb = Y[idx]

      model = DecisionTreeRegressor()
      model.fit(Xb, Yb)
      self.models.append(model)

  def predict(self, X):
    predictions = np.zeros(len(X))     #preiction start from zero
    for model in self.models:
      predictions += model.predict(X)    #we accumulate the prediction
    return predictions / self.B          #we calculate the mean 

  def score(self, X, Y):    #score function since we need to calculte the rsquared
    d1 = Y - self.predict(X)   #we get the difference between the target and the prediction
    d2 = Y - Y.mean()           #the difference between the target and the mean of the target
    return 1 - d1.dot(d1) / d2.dot(d2)


model = BaggedTreeRegressor(100)     #the common values for B is somewher between 200 and 500
model.fit(Xtrain, Ytrain)
print("R-squared for bagged tree:", model.score(Xtest, Ytest))
prediction = model.predict(Xtest)
     
# plot the bagged regressor's predictions
plt.plot(prediction, label="pred")
plt.plot(Y, label="target")
plt.legend()
plt.show()


# Question 3
# The best and only way to improve our model is by adding more columns which invariably means
# increasing the number of features we are training on.
