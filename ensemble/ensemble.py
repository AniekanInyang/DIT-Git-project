import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
import pandas as pd


df = pd.read_csv('data.csv')
data = df.values
shuffle(data)
df.head(3)
Y = data[:, -1]
X = data[:, 0:3]

for i in (0,1,2):
    m = X[:, i].mean()
    sd = X[:, i].std()
    X[:, i] = (X[:, i] - m) / sd

T = len(X)
N=12000

train_idx = np.random.choice(T, size=N, replace=False)
Xtrain = X[train_idx]
Ytrain = Y[train_idx]

model = DecisionTreeRegressor()
model.fit(Xtrain, Ytrain)
prediction = model.predict(X)
print('score is: ', model.score(Xtrain, Ytrain))

plt.plot(X, prediction)
plt.plot(X, Y)
plt.show()


class BaggedTreeRegressor:
    def __init__(self, B):
        self.B=B

    def fit(self, X,Y):
        S=len(X)
        self.models=[]
        for b in range(self.B):
            idx=np.random.choice(S, size=S, replace=True)
            Xb=X[idx]
            Yb=Y[idx]

            model=DecisionTreeRegressor()
            model.fit(Xb, Yb)
            self.models.append(model)

    def predict(self, X):
        predictions=np.zeros(len(X))
        for model in self.models:
            predictions+=model.predict(X)
        return predictions/self.B



    #Question 2
    #calculating R_square
    def score(self, X,Y):
        d1= Y- self.predict(X)
        d2=Y- Y.mean()
        return 1- d1.dot(d1)/ d2.dot(d2)

model=BaggedTreeRegressor(200)
model.fit(Xtrain, Ytrain)
print('score for bagged tree: ', model.score(Xtrain,Ytrain))
prediction=model.predict(X)

plt.plot(X, prediction)
plt.plot(X,Y)
plt.show()


#Question 3

'''  R_squared can be improved by adding  more features to train on
'''