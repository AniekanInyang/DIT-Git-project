import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_data(limit=None):
  print("Reading in and transforming data...")
  df = pd.read_csv("loan_data.csv", header=None, skiprows=1)
  new_male_df = df[1] == "Male"
  new_female_df = df[1] == "Female"
  df.loc[new_male_df, 1] = 1
  df.loc[new_female_df, 1] = 0

  new_grad = df[4] == "Graduate"
  new_not = df[4] == "Not Graduate"
  df.loc[new_grad, 4] = 1
  df.loc[new_not, 4] = 0
  
  new_male_df = df[5] == "Yes"
  new_female_df = df[5] == "No"
  df.loc[new_male_df, 5] = 1
  df.loc[new_female_df, 5] = 0

  new_no = df[2] == "No"
  new_yes = df[2] == "Yes"
  df.loc[new_no, 2] = 1
  df.loc[new_yes, 2] = 0
  
  #loan_status
  new_Y = df[12] == "Y"
  new_N = df[12] == "N"
  df.loc[new_Y, 12] = 1
  df.loc[new_N, 12] = 0

  new_Rural_df = df[11] == "Rural"
  new_Semiurban_df = df[11] == "Semiurban"
  new_Urban_df = df[11] == "Urban"
  df.loc[new_Rural_df, 11] = 0
  df.loc[new_Semiurban_df, 11] = 0.5
  df.loc[new_Urban_df, 11] = 1
  data = df.dropna()
  new_data = data.drop(columns = [0, 3, 9])
  scaler = MinMaxScaler()
  data_scaled = scaler.fit_transform(new_data)
  np.random.shuffle(data_scaled)
  
 
  X = data_scaled[:, :8]
  Y = data_scaled[:, 9]
  
  #final loan_status after Knn
  #to change the loan value to -1 and 1
  N = len(Y)
  Y2 = np.array([1] * N)
  for n in range(N):
      if Y[n] == 0:
          Y2[n] = -1
      if len(Y[Y == 0]) == len(Y2[Y2 == -1]):
          Y = Y2
          
  
  if limit is not None:   #we set a limit so that our algorithm will not take too long
      X, Y = X[:limit], Y[:limit]
  return X, Y
  