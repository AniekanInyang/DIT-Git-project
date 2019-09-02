#!/usr/bin/env python
# coding: utf-8

# # Week 1 - Linear Regression Assignment
# 

# # 1. Using y=mx+b, and the coefficients (a and b), solve for predicted Y and plot the line of best fit.
# 
# where X - Weight
#       Y - Height

# In[67]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[68]:


data = pd.read_csv("weight-height.csv")
data.head()


# In[69]:


X = data["Weight"]
X.head()


# In[70]:


Y = data["Height"]
Y.head()


# In[ ]:





# #Covert the X and Y to numpy arrays

# In[71]:


X = np.array(X)
Y = np.array(Y)


# In[72]:


#Plot the original data to see how it looks
plt.scatter(X, Y)


# In[73]:


#Calculating the common denominator for a and b from the given standard form of one dimensional linear regression equation


# In[74]:


denominator = X.dot(X) - X.mean()*X.sum()


# In[75]:


a = (X.dot(Y) - Y.mean()*X.sum())/denominator


# In[76]:


b = (Y.mean()*X.dot(X) - X.mean()*X.dot(Y))/denominator


# In[77]:


#Soling for the predicted y i.e yhat


# In[78]:


Yhat = a*X + b
Yhat


# In[79]:


#Plot showing the line of best fit


# In[80]:


plt.scatter(X, Y)
plt.plot(X,Yhat)


# In[ ]:





# # 2.  Calculate the maximum likelihood solution using weights to predict Y 

# In[81]:


N = len(X)

X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)


# In[82]:


X = np.vstack([np.ones(N), X]).T

w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)
Yhat_ml


# In[ ]:





# # 3. Plot the weights of the maximum likelihood solution

# In[83]:


plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Yhat_ml)
plt.show()


# # 4.  Calculate r_squared of number 1 and number 2 

# In[84]:


#r_squared for number 1

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)

print("a:", a, "b:", b)

print("The r-squared value for number one is : ", r2)


# In[ ]:





# In[85]:


#r_squared for number 2

d1 = Y - Yhat_ml
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)

print("a:", a, "b:", b)

print("The r-squared value for number two is : ", r2)


# In[ ]:





# In[ ]:





# # 5. How can you improve your model? I mean, how can you increase the value of you r_squared 

# In[ ]:


#Adding more input feature (i.e indepedent variables) to the raw data initially presented
#It’s also important to check and treat the extreme values or outliers in the variables to avoid it skewing off totally. 


# In[ ]:





# In[ ]:





# In[ ]:




