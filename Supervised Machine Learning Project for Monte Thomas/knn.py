from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
import matplotlib.pyplot as plt
from sortedcontainers import SortedList
from util import pre_process_data
from datetime import datetime

# 3. Code the KNN model class with comments explaining every step 

#Code for the KNN class

class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))
        for i,x in enumerate(X): # test points
            sl = SortedList() # stores (distance, class) tuples
            for j,xt in enumerate(self.X): # training points
                diff = x - xt
                d = diff.dot(diff)
                if len(sl) < self.k:
                    # don't need to check, just add
                    sl.add( (d, self.y[j]) )
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add( (d, self.y[j]) )
            # print "input:", x
            # print "sl:", sl

            # Empty vote dictionary
            votes = {}
            for _, v in sl:
                # print "v:", v
                votes[v] = votes.get(v,0) + 1
            # print "votes:", votes, "true:", Ytest[i]
            max_votes = 0
            max_votes_class = -1
            for v,count in iteritems(votes): #counting votes
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class
        #predictions    
        return y
    #scoring the predicted values 
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == '__main__':
    X, Y = pre_process_data(model='knn')
    Ntrain = len(X)//2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    train_scores = []
    test_scores = []

# 4. For K = 1 to 8, give a list of train accuracy scores and test accuracy stores     
    ks = (1,2,3,4,5,6,7,8)
    for k in ks:
        print("\nk =", k)
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print("Training time:", (datetime.now() - t0))

        t0 = datetime.now()
        train_score = knn.score(Xtrain, Ytrain)
        train_scores.append(train_score)
        print("Train accuracy:", train_score)
        print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

        t0 = datetime.now()
        test_score = knn.score(Xtest, Ytest)
        print("Test accuracy:", test_score)
        test_scores.append(test_score)
        print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))

# 5. Plot a graph of the test scores against k and train scores against k correctly labeled 

    plt.plot(ks, train_scores, label='train scores')
    plt.plot(ks, test_scores, label='test scores')
    plt.legend()
    plt.show()



# 6. What do you notice about k from the graph? Which do you think is the suitable k? and why? 
    #The initial values of k= 1,2 gave train accuracy of 100% which does not generalize very well.
    #Highervalues of k gave an averegely increasing train & test accuracy. The best combination was produced by k value of 8




# 7. Code the Perceptron model with comments explaining every step (Recall you have to add a condition in your preprocessing function that changed the labels (N= -1, Y = 1) for the Perceptron model 

class Perceptron:
    def fit(self, X, Y, learning_rate=1.0, epochs=1000):
        # randomly initializing the weights and setting b to zero
        D = X.shape[1]
        self.w = np.random.randn(D)
        self.b = 0
        print("first w: ", self.w)

        N = len(Y)
        costs = []
        for epoch in range(epochs):
            # making predictions
            Yhat = self.predict(X)
            # sieving out misclassified samples
            incorrect = np.nonzero(Y != Yhat)[0]
            if len(incorrect) == 0:
                # breaking out if all classified correctly
                break

            i = np.random.choice(incorrect)
            # updating values of w and b with randomly selected Y and/or X with learning rate
            self.w =self.w + learning_rate * Y[i] * X[i]
            self.b =self.b + learning_rate * Y[i]

            # cost is incorrect rate
            c = len(incorrect) / float(N)
            costs.append(c)
        print("final w:", self.w, "final b:", self.b, "epochs:", (epoch + 1), "/", epochs)
        plt.plot(costs)
        plt.title("Graph of Costs for Perceptron model")
        plt.show()

    # Predicting Y labels
    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)

    # Scoring the prediction
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
        

# 8. Plot a graph of the costs 

if __name__ == '__main__':
    X, Y = pre_process_data(model='perceptron')

    Ntrain = len(Y) // 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = Perceptron()
learning_rate = [0.001,0.01,0.1,1]
for rates in learning_rate:
    model.fit(Xtrain, Ytrain, learning_rate= rates, epochs=30000)
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Train size:", len(Ytrain))
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Test size:", len(Ytest))


# 9. Why did you choose your learning rate and epochs?

    # I created a list of possible learning rates between the usual lowest and highest learning rates of 0.001 and 1
    # I went further to iterate throught list and printing the train and test accuracy in each scenerio with a gradual increase in the number of epochs
    # I observed that I got the best Train & Test accuracy in each successive iteration with the when the learning rate is 0.01
    # This made me decided to work with the learning rate of 0.01 and the epoch value of 30000


# 10. What can you do to increase train and test accuracy? 
    # Tweaking the size of the train-test split can possibly help increase accuracy.
    # Adjusting the learning rates/epochs can affect the accuracy on the data output
    # Randomly assign your values in the model to avoid batch selection that can derail results gotten

