import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing


# 1. What is the difference in the implementation of the Bayes and Naïve Bayes model

#The Naive Bayes assumes all input features are independent/not correlated while Bayes classifier works with the fact that there's a correlation
#between all input features.


# 2. Create a python file called util.py which hosts a pre_process_data() function that takes in the data and preprocesses it


#a. Read in the data 
def pre_process_data(limit=None, model='knn'):
    print("Reading Data from source and pre-processing it ...")
    df1 = pd.read_csv('C:/Users/USER/Downloads/loan_data.csv', header='infer').dropna()
    df = df1.drop(["Loan_ID","Loan_Amount_Term","Dependents"],inplace = True, axis = 1)
    data = df.values

    
    
#b. Change the labels (N = 0, Y = 1) if it's knn and (N = -1, Y = 1) if it's perceptron
    
    if model == 'knn':
        df.Loan_Status[df.Loan_Status == 'N'] = 0
        df.Loan_Status[df.Loan_Status == 'Y'] = 1

    elif model == 'perceptron':
        df.Loan_Status[df.Loan_Status == 'N'] = -1
        df.Loan_Status[df.Loan_Status == 'Y'] =  1
        
    df.Married[df.Married == 'No'] = 0
    df.Married[df.Married == 'Yes'] = 1
    
    df.Self_Employed[df.Self_Employed == 'No'] = 0
    df.Self_Employed[df.Self_Employed == 'Yes'] = 1
    
    
    df.Property_Area[df.Property_Area == 'Rural'] = 0
    df.Property_Area[df.Property_Area == 'Semiurban'] = 0.5
    df.Property_Area[df.Property_Area == 'Urban'] = 1
    
    df.Gender[df.Gender == 'Male'] = 0
    df.Gender[df.Gender == 'Female'] = 1
    
    df.Education[df.Education == 'Graduate'] = 1
    df.Education[df.Education == 'Not Graduate'] = 0
    
    
    
    shuffle(data)
    X = data[:, :-1].astype(np.float)
    Y = data[:, -1].astype(np.int)
       
    
    
    min_max_scaler = preprocessing.MinMaxScaler()
    data[:, [4, 5, 6]] = min_max_scaler.fit_transform(data[:, [4, 5, 6]])
    
    
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y
    

