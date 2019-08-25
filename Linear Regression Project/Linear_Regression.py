import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the data
df = pd.read_csv('weight-height.csv')
data = df.values
X = data[:, 2].astype(np.float)
Y = data[:, 1].astype(np.float)

"""
QUESTION 1

Using y=mx+b, and the coefficients (a and b), solve for predicted Y and plot the line of best fit.
"""


denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

Yhat = a * X + b

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Maximum Likelihood Solution Using coefficients a and b')
plt.show()


"""
QUESTION 2

Calculate the maximum likelihood solution using weights to predict Y
"""

N = len(X)
X = np.vstack([np.ones(N), X]).T
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

Yhat_ml = X.dot(w_ml)


plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Yhat_ml)
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Maximum Likelihood Solution Using Weights')
plt.show()


"""
QUESTION 3

Plot the weights of the maximum likelihood solution
"""

plt.plot(w_ml, label='w_ml')
plt.title('Weights of the maximum likelihood solution')
plt.legend()
plt.show()
print("w_ml: {}".format(w_ml))


"""
QUESTION 4

Calculate r_squared of number 1 and number 2
"""

d1 = Y - Yhat
d2 = Y - Y.mean()
r2_one = 1 - d1.dot(d1) / d2.dot(d2)
print("r-squared of number 1 is:", r2_one)


d3 = Y - Yhat_ml
r2_two = 1 - d3.dot(d3) / d2.dot(d2)
print("r-squared of number 2 is:", r2_two)


"""
QUESTION 5

How can you improve your model? I mean, how can you increase the value of you r_squared?



The value of "r_squared" can be improved by adding another column to X be it a NOISE or a column that has a direct 
relationship
"""


