import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def preparing_data():

    df=pd.read_csv('weight-height-1.csv')
    df_matrix=df.values
    shuffle(df_matrix)


    X=df_matrix[:,1:].astype(np.float)
    Y=df_matrix[:,0]

    #question 1:
    #normalizing X variables
    #X[:,0]=X[:,0]-X[:,0].mean()/X[:,0].std()
    #X[:,1]=X[:,1]-X[:,1].mean()/X[:,1].std()
    #X[:, 1:] = X[:, 1:] - X[:, 1:].mean() / X[:, 1:].std()
    #X[:, 1:]
    for i in (0, 1):
        m = X[:, i].mean()
        s = X[:, i].std()
        X[:, i] = (X[:, i] - m) / s

    #one_hot encoding for Y
    N=len(Y)
    Y2=np.array([0]*N)

    for n in range(N):
        g = Y[n]
        Y2[n]

        if g == 'Male':
            Y2[n] = 0
        else:
            Y2[n] = 1

    Y=Y2

    return X, Y
#print(preparing_data())

#Question 2

X, Y=preparing_data()


Xtrain=X[:500]
Ytrain=Y[:500]

Xtest=X[-500:]
Ytest=Y[-500:]



#Question 3

N,D=Xtrain.shape
w= np.random.randn(D+1)#/np.sqrt((D+1))
b=0


ones=np.ones((N,1))
xb=np.concatenate((ones, Xtrain), axis=1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X,w,b):
    return sigmoid(X.dot(w)+b)

def classification_rate(Y, P):
    return np.mean(Y==P)

P_Y_given_X = forward(xb,w,b)

predictions= np.round(P_Y_given_X)
print("Score is: ", classification_rate(Ytrain, predictions))

# Question 4
def cross_entropy(T, P):
    return -np.mean(T*np.log(P)+(1-T)*np.log(1-P))

train_loss=[]

test_cost=[]
learning_rate = 0.009

for i in range(10000):
    pYtrain=forward(xb, w, b)
   # pYtest=forward(Xtest,w, b)

    ctrain= cross_entropy(Ytrain, pYtrain)
   # ctest=cross_entropy(Xtest,pYtest)

    train_loss.append(ctrain)
    #test_cost.append(ctest)

    w-= learning_rate*xb.T.dot(pYtrain-Ytrain)
    b-=learning_rate*(pYtrain-Ytrain).sum()
  #  n-=learning_rate*(pYtrain-Ytrain).sum()

    if i%1000==0:
        print(i,ctrain)

print('final train classification rate: ', classification_rate(Ytrain, np.round(pYtrain)))
#print('final train classification rate: ', classification_rate(Ytest, np.round(pYtest)))

''' I chose a learning rate of 0.001 because it is small. Choosing a larger rate can overshoot the minimum and make the loss worse. 
'''





#Question 5

print('Final Weight: ', w)

print('Initial model output: ', P_Y_given_X)

print('Final model output after gradient descent : ', pYtrain)