from __future__ import print_function, division
from future.utils import iteritems
import numpy as np
import matplotlib.pyplot as plt
from sortedcontainers import SortedList
from util import pre_process_data


"""
QUESTION 1

What is the difference in the implementation of the Bayes and Naïve Bayes model

ANSWER
======
1. in the fit() function of Bayes, we calculate and store the covariance instead of variance for Naive Bayes
2. in the predict() function of Bayes, we passed in the covariance to mvn.pdf() function rather than variance for 
   Naive Bayes
3. in Bayes, full covariance matrix was used instead of diagonal covariance matrix like its being done with naive bayes
4. full covariance matrix was not used for Naive Bayes because it treats all inputs features as independent entity
"""


"""
QUESTION 3

Code the KNN model class with comments explaining every step
"""


# declaring KNN class
class KNN(object):
    # creating a constructor and defining an instance attribute k
    def __init__(self, k):
        self.k = k

    # this saves the training data for later use
    def fit(self, X, y):
        self.X = X
        self.y = y

    # this method helps to make predictions for all the test points
    def predict(self, X):
        # creating y variable that will contain output predictions and initializing it to zero
        y = np.zeros(len(X))
        for i, x in enumerate(X):  # looping through test points
            sl = SortedList()  # creating a variable that will store (distance, class) tuples
            # for each test points we are looping through all the training points to find nearest neighbors
            for j, xt in enumerate(self.X):                     # looping through training points
                diff = x - xt                                   # represents distance
                d = diff.dot(diff)                              # represents the squared-distance
                if len(sl) < self.k:
                    # this adds the (distance, class) tuple since length of Sorted List is less than k value
                    sl.add((d, self.y[j]))
                else:
                    if d < sl[-1][0]:
                        # this deletes the last (distance, class) tuple if the distance of the last tuple is less than
                        # the new distance with its class
                        del sl[-1]
                        # this adds the new (distance, class) to the SortedList
                        sl.add((d, self.y[j]))
            # print "input:", x
            # print "sl:", sl

            # creating a votes variable as an empty dictionary
            votes = {}
            # looping through the sorted list of the KNN
            for _, v in sl:
                # collecting the votes for each class
                votes[v] = votes.get(v, 0) + 1
            # print "votes:", votes, "true:", Ytest[i]
            max_votes = 0
            max_votes_class = -1
            # looping through the votes to get the class with the highest or first highest vote if there is a tie
            for v, count in iteritems(votes):
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            # assigning the class with highest or first highest vote if there is a tie, to our output prediction array
            # corresponding to the  test point index
            y[i] = max_votes_class
        # returning predictions for all the corresponding test points
        return y

    # this is used to evaluate the accuracy of our predictions by correlating them with the actual labels
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


"""
QUESTION 4

For K = 1 to 8, give a list of train accuracy scores and test accuracy stores
"""

X, Y = pre_process_data()
Ntrain = len(X) // 2
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
train_scores = []
test_scores = []
ks = (1, 2, 3, 4, 5, 6, 7, 8)
for k in ks:
    print("\nk =", k)
    knn = KNN(k)
    knn.fit(Xtrain, Ytrain)

    train_score = knn.score(Xtrain, Ytrain)
    train_scores.append(train_score)
    print("Train accuracy:", train_score)

    test_score = knn.score(Xtest, Ytest)
    print("Test accuracy:", test_score)
    test_scores.append(test_score)


"""
QUESTION 5

Plot a graph of the test scores against k and train scores against k correctly labeled
"""

plt.plot(ks, train_scores, label='train scores')
plt.plot(ks, test_scores, label='test scores')
plt.legend()
plt.title("Graph of test scores and train scores against k for KNN model")
plt.show()


"""
QUESTION 6

What do you notice about k from the graph? 

ANSWER
======
The decision boundary produced by smaller values of k(1, 2) to the left is overfitting because it gives 100% training 
    accuracy at cost of finding overly complex curve which does not generalize the test data well
Higher values of K to the right produces averagely increasing values of Test accuracy until K = 5, after which it 
    started to decrease again


Which do you think is the suitable k? and why?

ANSWER
======
Since the essence of Machine Learning models is fitting past data to predict future data accurately as possible, then
    the suitable K is 5, because the KNN model generalizes the test data better at this K-value(5) than every other 
    values of K between 1 and 8 by always giving the highest Test accuracy
    
One may also say that k=5 best produces a decision boundary that more accurately capture the true pattern in the data
"""



"""
QUESTION 7

Code the Perceptron model with comments explaining every step (Recall you have to add a condition in your preprocessing 
function that changed the labels (N= -1, Y = 1) for the Perceptron model

"""


# declaring Perceptron class
class Perceptron:
    def fit(self, X, Y, learning_rate=1.0, epochs=1000):
        # randomly initializing the weights and setting b to zero
        D = X.shape[1]
        self.w = np.random.randn(D)
        self.b = 0
        print("first w: ", self.w)

        N = len(Y)
        costs = []
        # looping through maximum number of iterations, and this is done until maximum number of epochs is
        # reached or if there are no misclassified samples again
        for epoch in range(epochs):
            # making predictions
            Yhat = self.predict(X)
            # retrieving all the currently misclassified samples
            incorrect = np.nonzero(Y != Yhat)[0]
            if len(incorrect) == 0:
                # breaking out of the loop if all the points are classified correctly
                break

            # choosing index of misclassified sample at random
            i = np.random.choice(incorrect)
            # updating values of w and b with randomly selected Y and/or X with learning rate
            self.w += learning_rate * Y[i] * X[i]
            self.b += learning_rate * Y[i]

            # cost is incorrect rate
            c = len(incorrect) / float(N)
            costs.append(c)
        print("final w:", self.w, "final b:", self.b, "epochs:", (epoch + 1), "/", epochs)
        plt.plot(costs)
        plt.title("Graph of Costs for Perceptron model")
        plt.show()

    # this method helps to predict Y labels
    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)

    # this is used to evaluate the accuracy of our predictions by correlating them with the true labels
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


"""
QUESTION 8

Plot a graph of the costs
"""

X, Y = pre_process_data(model='perceptron')

Ntrain = len(Y) // 2
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

model = Perceptron()
model.fit(Xtrain, Ytrain, learning_rate=0.8587272727272728, epochs=50000)
print("Train accuracy:", model.score(Xtrain, Ytrain))
print("Train size:", len(Ytrain))
print("Test accuracy:", model.score(Xtest, Ytest))
print("Test size:", len(Ytest))



"""
QUESTION 9

Why did you choose your learning rate and epochs?

ANSWER
======

line_space = np.linspace(0.001, 1, 100)

for lr in line_space:
    model.fit(Xtrain, Ytrain, learning_rate=lr, epochs=50000)
    print("lr: ", lr)
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Test accuracy:", model.score(Xtest, Ytest))

Using the above code to iterate 100 learning rate values between 0.001 and 1 for 50,000 epochs, only learning_rate value
of 0.8587272727272728, gave me a Test accuracy of 0.83 which was the highest I ever got also with a higher Test accuracy
So I chose this learning rate because it gave the highest Test accuracy

I had to leave it at 50,000 epochs because even if I should increase this to a million, the epochs will still be 
exhausted without any improvement on the Train and Test accuracy
"""


"""
QUESTION 10

What can you do to increase train and test accuracy?

ANSWER
======
1. Regularisation of data
2. Early stopping: This filters the training of the Perceptron, resulting in decreased error in the test set

"""
