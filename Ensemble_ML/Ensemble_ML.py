import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from util import get_data, BaggedTreeRegressor


"""
QUESTION 1

X = # num_pages, ratings_count, text_reviews_count
Y = average_rating

1. Use two model ensembling methods to predict Y
"""

Xtrain, Ytrain, Xtest, Ytest = get_data()

num_trees = 200

rf = RandomForestRegressor(n_estimators=num_trees)
rf.fit(Xtrain, Ytrain)
rf_predictions = rf.predict(Xtest)


bg = BaggedTreeRegressor(n_estimators=num_trees)
bg.fit(Xtrain, Ytrain)
bg_predictions = rf.predict(Xtest)

plt.plot(Ytest, label='targets')
plt.plot(rf_predictions, label='rf_predictions')
plt.plot(bg_predictions, label='bg_predictions')
plt.legend()
plt.show()


"""
QUESTION 2

1. Calculate r_squared of your predicted value and the Y value
"""

rf_score = rf.score(Xtest, Ytest)
bg_score = bg.score(Xtest, Ytest)
print("r_squared for RandomForest Ensemble model: ", rf_score)
print("r_squared for BaggedTree Ensemble model: ", bg_score)

"""
QUESTION 3

1. How can you improve your model? I mean, how can you increase the value of you r_squared

ANSWER
======
1. Adding more attributes/features to increase the number of subsets of features that the tree trains on will help
   improve the model. Since the inventors of random forest recommend that the subsets of features that the tree trains 
   on could be as low as 5, whereas our model is only allowed to train on floor((D = 3) / 3) = 1 which is very low thus
   affecting the accuracy of the model
   
"""