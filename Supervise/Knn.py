import numpy as np
import matplotlib.pyplot as plt
from sortedcontainers import SortedList
from datetime import datetime
from util_knn import get_data


class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y): # this takes in X and y which are the inputs and the target 
        self.X = X       # all this does is saves X and it saves y
        self.y = y

    def predict(self, X): #next we write the predict function,it only takes in X which is our inputs 
        y = np.zeros(len(X))  # set Y to be np of zeros of thesame sizes X since we need a prediction for every input
        for i,x in enumerate(X): # test points : we loop through every X
            sl = SortedList() # stores (distance, class) tuples : we create sorted list, it has an input parameter called load, so that we can tell it how big we want the sorted list to be,we already know it has to be size k
            for j,xt in enumerate(self.X): # training points: for each input testpoint we have to loop through all the training points(self.X) to find the k nearest neighbors and jis the index X of T is the training point
                diff = x - xt  
                d = diff.dot(diff)  # calculating the squared distance here 
                if len(sl) < self.k:  # if our sorted list is less thank size k,we add a current point without checking anything
                    # don't need to check, just add
                    sl.add( (d, self.y[j]) )
                else:  #so if our current distance is less than that then we should delete the last value
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add( (d, self.y[j]) ) #and then add our cuurent or new value
            
            # collect all the vote 
            votes = {}          # by creating a dictionary call vote
            for _, v in sl:     # we loop through the sorted list of k nearest neighbour
               
                votes[v] = votes.get(v,0) + 1  #we don't care about the first element, we only care about the second elements since that is the class
            
            max_votes = 0
            max_votes_class = -1      # and then count as value
            for v, count in votes.items(): # we loop through all the votes
                if count > max_votes: # if vote is greater than our current max vote,we make our current max ite equal to the current ote
                    max_votes = count
                    max_votes_class = v 
            y[i] = max_votes_class   #we say y is the corresponding class
        return y

    #score function :thesame as psy-kit learn does,
    def score(self, X, Y):  #takes an X of Y
        P = self.predict(X)    #makes prediction on X
        return np.mean(P == Y)   # and then return the accuracy,so that returns an array of true and false which is thesame as zeros and ones.A nd by the taking the mean is the same as taking the sum and diving by N

# Newxt we come to the main area
if __name__ == '__main__':
    X, Y = get_data(503)   #first line is to get the data, which is by number of rows
    Ntrain = 252   #we say the first 252 is our training data and the last 251 as our test data
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    train_scores = []
    test_scores = []
    ks = (1,2,3,4,5,6,7,8)  #we want to test for different values of k
    for k in ks: 
        print("\nk =", k)
        knn = KNN(k)    # we create a model with it's neighbour
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain) 
        print("Training time:", (datetime.now() - t0)) # set a timer to see how long it takes for k to fit to the training
        
        # the training accuracy
        t0 = datetime.now()
        train_score = knn.score(Xtrain, Ytrain)
        train_scores.append(train_score)
        print("Train accuracy:", train_score)
        print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))
        
        # printing the test accuracy, generally test accuracy is going to be worse than your training accuracy
        t0 = datetime.now()
        test_score = knn.score(Xtest, Ytest)
        print("Test accuracy:", test_score)
        test_scores.append(test_score)
        print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))

    plt.plot(ks, train_scores, label='train scores')
    plt.plot(ks, test_scores, label='test scores')
    plt.legend()
    plt.show()
    
# 6. What do you notice about k from the graph? Which do you think is the suitable k? and why?

# ANS: from the graph, the lower the value of K the lower the training accuracy and vice versa

#since smaller values of K are more expressive and larger values of K are less expressive
#In this case, with respect to training accuracy, the best value of K is 1 and 2
#because they both gives us an accurate training accuracy of 1
#but if decision boundary is the focus point then our best values of K might varies