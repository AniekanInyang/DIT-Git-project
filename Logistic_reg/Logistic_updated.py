import numpy as np
import pandas as pd
# from sklearn.utils import shuffle
import os
import matplotlib.pyplot as plt



df = pd.read_csv("weight-height.csv", header=None, skiprows=1)
new_male_df = df[0] == "Male"
new_female_df = df[0] == "Female"
df.loc[new_male_df, 0] = 0
df.loc[new_female_df, 0] = 1
data = df.values
    
# shuffle it
np.random.shuffle(data)
 
  # split features and labels into X and Y
X = data[:, :-1]
Y = data[:,0]

  #QUESTION 1
  # normalize numerical columns for both X1 AND X2
X[:,1] = (X[:,1] - X[:,1].mean()) / X[:, 1].std()
X[2] = (X[2] - X[2].mean()) / X[2].std()

    
  # QUESTION 2
  # split train and test
Xtrain = X[:-500]
Ytrain = Y[:-500]
Xtest = X[-500:]
Ytest = Y[-500:]

# randomly initialize weights
D = X.shape[1]  #seeting the dimension of the data set
W = np.random.randn(D) #we then use that to initialize the weight of our logistic regression model
b = 0 # bias term so that's a scalar 

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)

P_Y_given_X = forward(X, W, b)
predictions = np.round(P_Y_given_X)

# cross entropy
def cross_entropy(T, pY):  #pY means p of y given X, which will return below 
    return -np.mean(T*np.log(pY) + (1 - T)*np.log(1 - pY))

# train loop
train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000): #i.e 10000 box
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)

    ctrain = cross_entropy(Ytrain, pYtrain) #traincost
    ctest = cross_entropy(Ytest, pYtest) #testcost
    train_costs.append(ctrain)
    test_costs.append(ctest)

    # gradient descent   #the vectorize formula we derived
    W -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate*(pYtrain - Ytrain).sum()

    if i % 1000 == 0: #so that we print evey 1000 steps
        print(i, ctrain, ctest)

print("Score:", classification_rate(Y, predictions))
print("Final train classification_rate:", classification_rate(Ytrain, np.round(pYtrain)))
print("Final test classification_rate:", classification_rate(Ytest, np.round(pYtest)))
print("Final w:", W)
print("Final b:", b)
print("Initial model output: ", P_Y_given_X)
print("Final model output: ", pYtrain)
legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()
