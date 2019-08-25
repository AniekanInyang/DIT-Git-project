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
        x = float(x)
        X.append([1,x])
        Y.append(float(y))


X = np.array(X)
Y = np.array(Y)

# maximum likelihood
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

Yhat_ml = X.dot(w_ml) #predicted y

plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Yhat_ml, label = "maximum likelihood")
plt.show()

# predicted y
print (Yhat_ml)


# calculating r-squared
d1 = Y - Yhat_ml
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("The r-squared is {}".format(r2))


# The model can be improved by adding more features, , adding more data and regularization
