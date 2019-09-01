import numpy as np
import pandas as pd

# load the data
df = pd.read_csv('weight-height.csv')
data = df.values

# shuffle it
np.random.shuffle(data)

X = data[:, 1:].astype(np.float)
Y = data[:, 0]

"""
QUESTION 1

Normalize the X variables and convert the Y variable to numeric value. 0 for Male and 1 for Female.
"""

# normalizing the X variables
for i in (0, 1):
    m = X[:, i].mean()
    s = X[:, i].std()
    X[:, i] = (X[:, i] - m) / s

# converting the Y variable to numeric value, 0 for Male and 1 for Female
N = len(Y)
Y2 = np.array([0] * N)

for n in range(N):
    if Y[n] == 'Female':
        Y2[n] = 1

if len(Y[Y == 'Female']) == len(Y2[Y2 == 1]):
    Y = Y2


"""
QUESTION 2

Divide into test and train data. 500 train data and 500 test data.
"""

Xtrain = X[:500]
Ytrain = Y[:500]


Xtest = X[-500:]
Ytest = Y[-500:]


"""
QUESTION 3

Initialize random weights. Calculate the model output.
"""

# Initialize random weights
N, D = Xtrain.shape
w = np.random.randn(D+1)
b = 0  # bias term

ones = np.ones((N, 1))
Xb = np.concatenate((ones, Xtrain), axis=1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X, w, b):
    return sigmoid(X.dot(w) + b)


def classification_rate(Y, P):
    return np.mean(Y == P)


# Calculate the model output
P_Y_given_X = forward(Xb, w, b)

predictions = np.round(P_Y_given_X)
print("Score for Question 3:", classification_rate(Ytrain, predictions))


"""
QUESTION 4

Using the loss function, Choose a learning rate and use gradient descent to attempt to minimize this loss. 

Why did you choose these parameters?

ANSWER
======
I chose the parameters values used below because they best control how much to change the model in response to the 
estimated error each time the model weights are updated

I also chose these values because they give me higher classification rate(score)
"""


# loss function
def cross_entropy(T, P):
    return -np.mean(T * np.log(P) + (1 - T) * np.log(1 - P))


train_loss = []

# Choose a learning rate
learning_rate = 0.009

for i in range(5000):
    pYtrain = forward(Xb, w, b)

    l_train = cross_entropy(Ytrain, pYtrain)

    train_loss.append(l_train)

    # use gradient descent to attempt to minimize this loss
    w -= learning_rate * Xb.T.dot(pYtrain - Ytrain)
    b -= learning_rate * (pYtrain - Ytrain).sum()

    # Print the loss for some steps
    if i % 500 == 0:
        print(i, l_train)

print("Final train classification_rate: ", classification_rate(Ytrain, np.round(pYtrain)))


"""
QUESTION 5

Print the loss for some steps. Print the final weight, the initial model output 
and the model output after gradient descent
"""
# Print the loss for some steps is above in question 4

# Print the final weight
print("Final weight: ", w)

# Print the initial model output
print("Initial model output: ", P_Y_given_X)

# Print the model output after gradient descent
print("Final model output: ", pYtrain)
