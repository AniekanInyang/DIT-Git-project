import numpy as np
from Method import get_binary_data

X, Y, _, _ = get_binary_data()

#QUESTION 3

# randomly initialize weights
D = X.shape[1]  
W = np.random.randn(D) 
b = 0 

# this is our function to make the predictions 
def sigmoid(a):
    return 1 / (1 + np.exp(-a))
    
def forward(X, W, b): 
    return sigmoid(X.dot(W) + b)

P_Y_given_X = forward(X, W, b)
predictions = np.round(P_Y_given_X)

# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P) #this looks like is returning a boolean but is returnung 1 and 0
    #it will divide the number of correct by the total number
print("Score:", classification_rate(Y, predictions))
#we can a lower score percentage of 44%
#next we will look at how to train this data to get a higher percentage of accuracy