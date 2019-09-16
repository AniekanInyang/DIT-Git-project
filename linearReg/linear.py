import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Question 1
df=pd.read_csv('weight-height.csv')

X=df['Weight']
Y=df['Height']

X=np.array(X)
Y=np.array(Y)


plt.scatter(X,Y)
plt.show()

denominator= X.dot(X)- X.mean()*X.sum()
a=(X.dot(Y)-Y.mean()*X.sum())/denominator
b=(Y.mean()*X.dot(X)-X.mean()*X.dot(Y))/denominator

yhat=a*X +b

plt.scatter(X,Y)
plt.plot(X, yhat)
plt.show()

#Question 2

df['ones']=1
X=df[['Weight', 'ones']]
Y=df['Height']

X=np.array(X)
Y=np.array(Y)

w=np.linalg.solve(X.T.dot(X), X.T.dot(Y))

yhat_w= X.dot(w)

#Question 3
plt.scatter(X[:,0], Y)
plt.plot(X[:,0], yhat_w, label="max like")
plt.legend()
plt.show()

#Question 4
# r_square for question 1

d1= Y- yhat
d2=Y- Y.mean()
r2_question1= 1 - d1.dot(d1)/d2.dot(d2)
print(r2_question1)

# r_square for question 2
d1= Y- yhat_w
d2=Y- Y.mean()
r2_question2=1 - d1.dot(d1)/d2.dot(d2)
print(r2_question2)

#Question 5

print('r_squared can be increased by adding more variables.')