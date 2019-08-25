import numpy as np
import csv
import matplotlib.pyplot as plt


X = []
Y = []

with open('weight-height.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        x = row[2]
        y = row[1]
        X.append(float(x))
        Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean() *X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator

# predictied y
Yhat = a*X + b

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# calculating r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("The r-squared is {}".format(r2))
