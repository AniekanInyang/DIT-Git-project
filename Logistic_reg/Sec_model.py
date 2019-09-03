import numpy as np
from Method import get_binary_data
# from sklearn.utils import shuffle

Xtrain, Ytrain, Xtest, Ytest = get_binary_data()
X, Y, _, _ = get_binary_data()

# we randomly our initialize weights
D = Xtrain.shape[1]
W = np.random.randn(D)
b = 0 # bias term



# make predictions
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

P_Y_given_X = forward(X, W, b)
predictions = np.round(P_Y_given_X)
# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)


# cross entropy
epsilon = 1e-5
def cross_entropy(T, pY): 
    return -np.mean(T*np.log(pY + epsilon) + (1 - T)*np.log(1 - pY + epsilon))

# train loop
train_costs = []
test_costs = []
learning_rate = 0.001 
#Question 4
#This is the best learning rate  but we hae others like 0.1, 0.01 and 0.03 so as to avoid the model fom shooting out of range and therefore returns a divion error
#also helps to control how much to change the model in respect to the loss function error
for i in range(10000): 
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)

    ctrain = cross_entropy(Ytrain, pYtrain) #traincost
    ctest = cross_entropy(Ytest, pYtest) #testcost
    train_costs.append(ctrain)
    test_costs.append(ctest)

    # gradient descent   #the vectorize formula we derived
    W -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate*(pYtrain - Ytrain).sum()
    Y = sigmoid(X.dot(W) + b)
    if i % 1000 == 0: #so that we print evey 1000 steps
        # recalculate Y
        
        print(i, ctrain, ctest)

print("Score after Gradient descent:", classification_rate(Y, predictions))
print("Final train classification_rate:", classification_rate(Ytrain, np.round(pYtrain)))
print("Final test classification_rate:", classification_rate(Ytest, np.round(pYtest)))
print("Final w:", W)