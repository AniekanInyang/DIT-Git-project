from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from util_per import get_data
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime



class Perceptron: 
    def fit(self, X, Y, learning_rate=0.1, epochs=240):
        

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
            i = np.random.choice(incorrect) # next we choose the random sample from the incorrect sample to update w and b
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

# Ans: the learning rate helps in achieving a perfect classification rate of 1.0, but in which i got 0.8 
#      and i choose epochs of 240 as we have to split our data

# 10. What can you do to increase train and test accuracy?
#Ans: by making the best use of our hyperparameters like like the learning rate, bias term and
    # also ensuring all null values are properly taken care of and finally ensuring
    # all input features in different columns does not outweights others which can
    # result to overfitting and eventually affect our output.