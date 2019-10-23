import numpy as np
import pandas as pd



def get_data():
    print("we good right")

    '''read in the data'''
    df = pd.read_csv('loan_data.csv')
    df = df.dropna()
    data = df.values

    '''Drop Dependents, LoanID and LoanAmountTerm columns '''
    df.drop(['Dependents', 'Loan_ID', 'Loan_Amount_Term'], axis=1, inplace=True)

    # print(df)
    '''Change the labels (N=0, Y=1)  Loan_Status'''
    Y = data[:, -1]
    N = len(Y)
    Y2 = np.array([0] * N)

    for n in range(N):
        l = Y[n]
        Y2[n]
        if l == 'N':
            Y2[n] = -1
        else:
            Y2[n] = 1

    Y = Y2
    df['Loan_Status'] = Y

    '''one hot encoding for Gender'''
    G = data[:, 1]
    N = len(G)
    G2 = np.array([0] * N)

    for n in range(N):
        g = G[n]
        G2[n]
        if g == 'Male':
            G2[n] = 0
        else:
            G2[n] = 1
    G = G2
    df['Gender'] = G

    '''one hot encoding for Married'''
    M = data[:, 2]
    N = len(M)
    M2 = np.array([0] * N)

    for n in range(N):
        m = M[n]
        M2[n]
        if m == 'No':
            M2[n] = 0
        else:
            M2[n] = 1
    M = M2
    df['Married'] = M

    '''one hot encoding for Education'''
    E = data[:, 4]
    N = len(E)
    E2 = np.array([0] * N)

    for n in range(N):
        e = E[n]
        E2[n]
        if e == 'Graduate':
            E2[n] = 1
        else:
            E2[n] = 0
    E = E2
    df['Education'] = E

    '''one hot encoding for Self_Employed'''
    SE = data[:, 5]
    N = len(SE)
    SE2 = np.array([0] * N)

    for n in range(N):
        se = SE[n]
        SE2[n]
        if se == 'Yes':
            SE2[n] = 1
        else:
            SE2[n] = 0
    SE = SE2
    df['Self_Employed'] = SE

    '''one hot encoding for Credit_History'''
    CH = data[:, 10]
    N = len(CH)
    CH2 = np.array([0] * N)

    for n in range(N):
        ch = CH[n]
        CH2[n]
        if ch == 1.0:
            CH2[n] = 1
        else:
            CH2[n] = 0
    CH = CH2
    df['Credit_History'] = CH

    '''Normalize the LoanAmount, ApplicantIncome and CoapplicantIncome columns 
        using the MinMax Scaler so that it’s between 0 and 1 '''
    norm6 = data[:, 6]
    for i in (0,):
        min = norm6[:].min()
        max = norm6[:].max()
        norm6[:] = ((norm6[:] - min) / (max - min))
        df['ApplicantIncome'] = norm6[:]

    norm7 = data[:, 7]
    for i in (0,):
        min = norm7[:].min()
        max = norm7[:].max()
        norm7[:] = ((norm7[:] - min) / (max - min))
        df['CoapplicantIncome'] = norm7[:]

    norm8 = data[:, 8]
    for i in (0,):
        min = norm8[:].min()
        max = norm8[:].max()
        norm8[:] = ((norm8[:] - min) / (max - min)) 
        df['LoanAmount'] = norm8[:]


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

    data2 = df.values
    X = data2[:, 0:8]
    #print(X)

    return X, Y


get_data()