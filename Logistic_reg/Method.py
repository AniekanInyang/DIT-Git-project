import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os



def get_data():
  df = pd.read_csv("weight-height.csv", header=None, skiprows=1)
  new_male_df = df[0] == "Male"
  new_female_df = df[0] == "Female"
  df.loc[new_male_df, 0] = 0
  df.loc[new_female_df, 0] = 1
  data = df[[ 1, 2, 0]].astype(float).values
    
# shuffle it
  np.random.shuffle(data)
 
  # split features and labels into X and Y
  X = data[:, :-1]
  Y = data[:,-1]

  # normalize numerical columns for both X1 AND X2
  X[:,1] = (X[:,1] - X[:,1].mean()) / X[:, 1].std()
  X[2] = (X[2] - X[2].mean()) / X[2].std()
    
  # split train and test
  Xtrain = X[:-500]
  Ytrain = Y[:-500]
  Xtest = X[-500:]
  Ytest = Y[-500:]

  # normalize columns 1 and 2
  for i in (1, 2):
    m = Xtrain[i].mean()
    s = Xtrain[i].std()
    Xtrain[i] = (Xtrain[i] - m) / s
    Xtest[i] = (Xtest[i] - m) / s

  return Xtrain, Ytrain, Xtest, Ytest


def get_binary_data():
  # return only the data from the first 2 classes in binary data as we are not returning dataset
  Xtrain, Ytrain, Xtest, Ytest = get_data() 
  X2train = Xtrain[Ytrain <= 1]
  Y2train = Ytrain[Ytrain <= 1]
  X2test = Xtest[Ytest <= 1]
  Y2test = Ytest[Ytest <= 1]
  return X2train, Y2train, X2test, Y2test
