import numpy as np
from sortedcontainers import SortedList
from util import prepare_data
from datetime import datetime
import matplotlib.pyplot as plt


class KNN(object):
    def __init__(self, k):
        self.k=k

    def fit(self, X, y):  #function to take in input X and y
        self.X=X
        self.y=y

    def predict(self, X): #function that takes input X
        y=np.zeros(len(X))  #creates an array of zeros for y equivalent to X
        for i,x in enumerate(X): #loop through test data X, getting the index and the individual points
            sl=SortedList() #create a sorted list
            for j, xt in enumerate(self.X): # loop through all the training points for each input test point
                diff=x-xt # difference between  the test points and the training points to get the distance
                d=diff.dot(diff) # the distance is squared
                if len(sl)<self.k: #if length sorted list is lesser than k, then add the current point
                    sl.add((d, self.y[j]))
                else: #else if the distance is lesser than the sorted list, then we delete the last value
                    if d<sl[-1][0]:
                        del sl[-1]
                        sl.add((d, self.y[j])) #then add the current value
            votes={}  # create an empty dictionary to collect all the votes
            for _,v in sl: #loop through the sorted list containing our k nearest neighbours
                votes[v]=votes.get(v,0)+1   #get the class of k
            max_votes=0
            max_votes_class=-1
            for v, count in votes.items(): #loop through the votes
                if count> max_votes:   #if votes is greater than the maximum votes
                    max_votes=count    #then the maximum votes will be equal to the current votes
                    max_votes_class=v
            y[i]=max_votes_class       #y becomes the corresponding class
        return y

    def score(self, X,Y):       #A    function that takes X and Y
        S=self.predict(X)    #Then predicts X
        return np.mean(S==Y)  # returns the mean score/accuracy



if __name__=='__main__':
    X,Y=prepare_data()
    Ntrain=300 #train 300 data points

    Xtrain, Ytrain=X[:Ntrain], Y[:Ntrain]#
    Xtest, Ytest= X[Ntrain:], Y[Ntrain:]


    test_score=[]
    train_score=[]
    for k in (1,2,3,4,5,6,7,8): # looping through K values from 1 to 8
        knn=KNN(k)
        t0=datetime.now()  #get the current datetime
        knn.fit(Xtrain, Ytrain)
        print("Training time: ",(datetime.now()- t0))

        t0 = datetime.now()
        print("Train accuracy: ", knn.score(Xtrain, Ytrain))
        train_score.append(knn.score(Xtrain, Ytrain))
        #print(train_score)

        t0 = datetime.now()
        print("Test accuracy: ", knn.score(Xtest, Ytest))
        test_score.append(knn.score(Xtest, Ytest))
        #print(test_score)

    print("Test_scores: ",test_score)
    print("Train_scores: ",train_score)

    k=[1,2,3,4,5,6,7,8]

    plt.plot(k, test_score, label="Test_Score")
    plt.plot(k, train_score, label="Train_Score")
    plt.legend()
    plt.show()


#Answer 6

#  From the graph K1 and K2 had a higher training accuracy of 1
#  K1 and K2 should be suitable values for k
#  lower values of k are more expressive. 
            
        
        