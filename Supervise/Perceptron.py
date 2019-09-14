from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime


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
    df.loc[new_N, 12] = -1
    new_Rural_df = df[11] == "Rural"
    new_Semiurban_df = df[11] == "Semiurban"
    new_Urban_df = df[11] == "Urban"
    df.loc[new_Rural_df, 11] = 0
    df.loc[new_Semiurban_df, 11] = 0.5
    df.loc[new_Urban_df, 11] = 1
    df.drop(columns = [0, 3, 9], inplace = True)
    data = df.dropna()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    np.random.shuffle(data_scaled)
    
   
    X = data_scaled[:, :8]
    Y = data_scaled[:, 9]
    
    #to change the loan value to -1 and 1
    N = len(Y)
    Y2 = np.array([1] * N)
    for n in range(N):
        if Y[n] == 'N':
            Y2[n] = -1
        if len(Y[Y == 'N']) == len(Y2[Y2 == -1]):
            Y = Y2
            
    
    if limit is not None:   #we set a liit so that our algorithm will not take too long
        X, Y = X[:limit], Y[:limit]
    return X, Y
    

class Perceptron: 
    def fit(self, X, Y, learning_rate=0.1, epochs=252):
        

        # initialize random weights
        D = X.shape[1]   #we get the dimensionality which is the shape of X and we create W which is size of D 
        self.w = np.random.randn(D)
        self.b = 0  # b is just the scalar 

        N = len(Y)    #length of items in Y variable
        costs = []   #we create the cost array which is empty
        for epoch in range(epochs):   #we loop through all the epochs
            # determine which samples are misclassified, if any
            Yhat = self.predict(X)  #  get a prediction so that we know what is misclassified
            incorrect = np.nonzero(Y != Yhat)[0]   # using non-zero to get any of the samples where y is not equal to yhat
            if len(incorrect) == 0:  #  if there are no incorrect samples we have to break out of the loop
                # we are done!
                break

            # choose a random incorrect sample
            i = np.random.choice(incorrect) # next we choose the random sample from the incorrect sample
            self.w += learning_rate*Y[i]*X[i]
            self.b += learning_rate*Y[i]

            # cost is incorrect rate
            c = len(incorrect) / float(N)
            costs.append(c)
        print("final w:", self.w, "final b:", self.b, "epochs:", (epoch+1), "/", epochs) #when done with the loops we print out
        plt.plot(costs)  #we plot the cost 
        plt.show()

    def predict(self, X):  # predict function
        return np.sign(X.dot(self.w) + self.b)

    def score(self, X, Y):   # the score function
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == '__main__':
    X, Y = get_data()
    Ntrain = len(Y) // 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = Perceptron()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))
    
    
# 9. Why did you choose your learning rate and epochs?
# Ans: the learning rate helpsme in achieving a perfect classification rate of 1.0 
#      and i choose epochs of 252 as we have to split our data

# 10. What can you do to increase train and test accuracy?
#Ans: by making the best use of our hyperparameters like like the learning rate, bias term and
    # also ensuring all null values are properly taken care ofand finally ensuring
    # all input features in different columns does not outweights others which can
    # result to overfitting which will affect our output.