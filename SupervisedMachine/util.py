import numpy as np
import pandas as pd


def prepare_data():
    print("we good right")
    '''read in the data'''
    df = pd.read_csv('loan_data.csv')
    df = df.dropna()
    data = df.values

    '''Change the labels (N=0, Y=1)  Loan_Status'''
    Y = data[:, -1]
    N = len(Y)
    Y2 = np.array([0] * N)

    for n in range(N):
        l = Y[n]
        Y2[n]
        if l == 'N':
            Y2[n] = 0
        else:
            Y2[n] = 1

    Y = Y2
    df['Loan_Status'] = Y

    '''Drop Dependents, LoanID and LoanAmountTerm columns '''
    df.drop(['Dependents', 'Loan_ID', 'Loan_Amount_Term'], axis=1, inplace=True)

    # print(df)
    '''Normalize the LoanAmount, ApplicantIncome and CoapplicantIncome columns 
    using the MinMax Scaler so that it’s between 0 and 1 '''

    X = data[:, 6:9]

    for i in (0, 1,2):
        min = X[:, i].min()
        max = X[:, i].max()
        X[:, i] = ((X[:, i] - min) / (max - min))

    print(X)
    '''Change property column (Rural=0, Semiurban=0.5, Urban=1)'''
    P = data[:, -2]
    N = len(P)
    P2 = np.array([0] * N).astype(np.float)

    for n in range(N):
        p = P[n]
        P2[n]
        if p == 'Rural':
            P2[n] = 0
        elif p == 'Semiurban':
            P2[n] = 0.5
        else:
            P2[n] = 1

    P = P2
    df['Property_Area'] = P

    '''Drop all rows with missing data'''
    df= df.dropna()



    return X,Y
X, Y=prepare_data()
