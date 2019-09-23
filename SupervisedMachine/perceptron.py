import numpy as np
from per_util import get_data
import matplotlib.pyplot as plt
from datetime import datetime



class Perceptron:
    def fit(self, X, Y, learning_rate=0.01, epochs=1000):
        D = X.shape[1] #dimension of len X

        self.w = np.random.randn(D)  # creating random weights
        
        self.b = 0  # bias term which is scaler

        N = len(Y)
        costs = []

        for epoch in range(epochs): #loop through the epochs 
            yhat = self.predict(X)  #   get a prediction so as to know what may be misclassified
            incorrect = np.nonzero(Y != yhat)[0] # gets any of the samples where Y does not equal yhat
            if len(incorrect) == 0:  #break look if there are no incorrect samples
                break

            i = np.random.choice(incorrect)   #randomly select one sample from the incorect samples
            self.w = self.w + learning_rate * Y[i] * X[i]       #update w using a random Y and a random X
            self.b = self.b + learning_rate * Y[i]              # update the bias

            c = len(incorrect) / float(N)    #get the incorrect rate
            costs.append(c)                  #append the incorrect rate to the cost empty list
        # print(costs)

        print('final w: ', self.w, "final b: ", self.b, "epochs: ", (epoch + 1), '/', epochs)  #print the final w and b and the total number of epochswe went through
        plt.plot(costs)
        plt.show()

    def predict(self, X):        #write the predict function for X
        return np.sign(X.dot(self.w) + self.b)    #returns a sign value for the matrix dot product between X and w plus the bias

    def score(self, X, Y):    #write the score function for X and Y
        P = self.predict(X)   #run it through the predict function
        return np.mean(P == Y)


if __name__ == '__main__':
    X, Y = get_data()
    # plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    # plt.show()

    Ntrain = len(Y)//2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = Perceptron()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print('Training time: ', (datetime.now() - t0))

    t0 = datetime.now()
    print('Train accuracy: ', model.score(Xtrain, Ytrain))

    t0 = datetime.now()
    print('Test accuracy: ', model.score(Xtest, Ytest))






#Question 9

#  i chose a learning rate of 0.01 as this improves the accuracy
#  i chose an epoch of

#Question 10
# test and train accuracy can be improved by using more samples of X
#by using a good learning rate
#by removing null values
