import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing

"""
QUESTION 2

Create a python file called util.py which hosts a pre_process_data() function that takes in the data and 
pre-processes it 
a. Read in the data
b. Change the labels (N = 0, Y = 1)
c. Drop Dependents, LoanID and LoanAmountTerm columns
d. Normalize the LoanAmount, ApplicantIncome and CoapplicantIncome columns using the MinMax Scaler so that it’s 
    between 0 and 1
e. Change Property column (Rural = 0, Semiurban = 0.5, Urban = 1)
f. Drop all rows with missing data
"""


def pre_process_data(limit=None, model='knn'):
    print("Reading in and transforming data...")

    # Read in the data
    # Drop all rows with missing data
    df = pd.read_csv('loan_data.csv').dropna(how="any")

    # Change the labels (N = 0, Y = 1) if model = 'knn' or labels (N = -1, Y = 1) if if model = 'perceptron'
    data = df.values
    N = len(data[:, -1])

    if model == 'knn':
        for i in range(N):
            if data[i, -1] == 'N':
                data[i, -1] = 0
            else:
                data[i, -1] = 1
    elif model == 'perceptron':
        for i in range(N):
            if data[i, -1] == 'N':
                data[i, -1] = -1
            else:
                data[i, -1] = 1

    # Drop Dependents, LoanID and LoanAmountTerm columns
    R = len(df.columns)
    idy = []
    [idy.append(i) for i in range(R) if i not in (0, 3, 9)]
    data = data[:, idy]

    # Normalize the LoanAmount, ApplicantIncome and CoapplicantIncome columns using the MinMax Scaler so that it’s
    # between 0 and 1
    mm_scaler = preprocessing.MinMaxScaler()
    data[:, [4, 5, 6]] = mm_scaler.fit_transform(data[:, [4, 5, 6]])

    # Change Property column (Rural = 0, Semiurban = 0.5, Urban = 1)
    N = len(data[:, -2])

    for i in range(N):
        if data[i, -2] == 'Rural':
            data[i, -2] = 0
        elif data[i, -2] == 'Semiurban':
            data[i, -2] = 0.5
        else:
            data[i, -2] = 1

    # Drop all rows with missing data has been done while reading in the data

    # Doing one-hot encoding for Gender, Married, Education, Self_Employed,
    # Gender
    N = len(data[:, 0])
    for i in range(N):
        if data[i, 0] == 'Female':
            data[i, 0] = 1
        else:
            data[i, 0] = 0

    # Married
    N = len(data[:, 1])
    for i in range(N):
        if data[i, 1] == 'Yes':
            data[i, 1] = 1
        else:
            data[i, 1] = 0

    # Education
    N = len(data[:, 2])
    for i in range(N):
        if data[i, 2] == 'Graduate':
            data[i, 2] = 1
        else:
            data[i, 2] = 0

    # Self_Employed
    N = len(data[:, 3])
    for i in range(N):
        if data[i, 3] == 'Yes':
            data[i, 3] = 1
        else:
            data[i, 3] = 0

    # shuffle data and return X and Y
    shuffle(data)
    X = data[:, :-1].astype(np.float)
    Y = data[:, -1].astype(np.int)
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    # if model == 'perceptron':
    #     X = (X * 2) - 1
    return X, Y

