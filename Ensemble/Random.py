
from __future__ import print_function, division
from future.utils import iteritems  
from builtins import range, input

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


NUMERICAL_COLS = [
  '# num_pages', # numerical
  'ratings_count', # numerical
  'text_reviews_count', # numerical

]


class DataTransformer:
  def fit(self, df):
    self.scalers = {}
    for col in NUMERICAL_COLS:
      scaler = StandardScaler()
      scaler.fit(df[col].values.reshape(-1, 1))
      self.scalers[col] = scaler


  def transform(self, df):
    N, _ = df.shape
    D = len(NUMERICAL_COLS)
    X = np.zeros((N, D))
    i = 0
    for col, scaler in iteritems(self.scalers):
      X[:,i] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
      i += 1
    return X

  def fit_transform(self, df):
    self.fit(df)
    return self.transform(df)


def get_data():
  
  df = pd.read_csv('data.csv')


  transformer = DataTransformer()

  # shuffle the data
  N = len(df)

  train_idx = np.random.choice(N, size=int(0.7*N), replace=False)  #we split the data into 70% train and 30% test
  test_idx = [i for i in range(N) if i not in train_idx]
  df_train = df.loc[train_idx]
  df_test = df.loc[test_idx]
  
  Xtrain = transformer.fit_transform(df_train)
  Ytrain = df_train['average_rating'].values
  Xtest = transformer.transform(df_test)
  Ytest = df_test['average_rating'].values
#   print(Ytrain)
  return Xtrain, Ytrain, Xtest, Ytest
  

if __name__ == '__main__':
  Xtrain, Ytrain, Xtest, Ytest = get_data()

  #we create an instance of randomforest regressor, here we are using 100 trees
  model = RandomForestRegressor(n_estimators=100) 
  model.fit(Xtrain, Ytrain)
  predictions = model.predict(Xtest)

  
  
  # plot predictions vs targets
  plt.scatter(Ytest, predictions)
  plt.xlabel("target")
  plt.ylabel("prediction")
  ymin = np.round( min( min(Ytest), min(predictions) ) )
#   print(min(predictions))
  ymax = np.ceil( max( max(Ytest), max(predictions) ) )
  print("ymin:", ymin, "ymax:", ymax)
  r = range(int(ymin), int(ymax) + 1)
  plt.plot(r, r)
  plt.show()

  plt.plot(Ytest, label='targets')
  plt.plot(predictions, label='predictions')
  plt.legend()
  plt.show()

    #we do cross validation on all of our models
  # do a quick baseline test
  baseline = LinearRegression()
  single_tree = DecisionTreeRegressor()
  print("CV single tree:", cross_val_score(single_tree, Xtrain, Ytrain).mean())
  print("CV baseline:", cross_val_score(baseline, Xtrain, Ytrain).mean())
  print("CV forest:", cross_val_score(model, Xtrain, Ytrain).mean())

    
  # test score
  single_tree.fit(Xtrain, Ytrain)
  baseline.fit(Xtrain, Ytrain)
  print("r-squared single tree:", single_tree.score(Xtest, Ytest))
  print("r-squared baseline:", baseline.score(Xtest, Ytest))
  print("r-squared forest:", model.score(Xtest, Ytest))
  

# Question 3
# The best and only way to improve our model is by adding more columns which invariably means
# increasing the number of features we are training on.