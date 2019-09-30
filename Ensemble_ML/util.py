import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.tree import DecisionTreeRegressor
from future.utils import iteritems

NUMERICAL_COLS = [
    '# num_pages',
    'ratings_count',
    'text_reviews_count'
]

NO_TRANSFORM = []


class DataTransformer:
    def fit(self, df):
        self.scalers = {}
        for col in NUMERICAL_COLS:
            scaler = StandardScaler()
            scaler.fit(df[col].values.reshape(-1, 1))
            self.scalers[col] = scaler

    def transform(self, df):
        N, _ = df.shape
        D = len(NUMERICAL_COLS) + len(NO_TRANSFORM)
        X = np.zeros((N, D))
        i = 0
        for col, scaler in iteritems(self.scalers):
            X[:, i] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
            i += 1
        for col in NO_TRANSFORM:
            X[:, i] = df[col]
            i += 1
        return X

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


class BaggedTreeRegressor:
    def __init__(self, n_estimators, max_depth=None):
        self.B = n_estimators
        self.max_depth = max_depth

    def fit(self, X, Y):
        N = len(X)
        self.models = []
        for b in range(self.B):
            idx = np.random.choice(N, size=N, replace=True)
            Xb = X[idx]
            Yb = Y[idx]

            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(Xb, Yb)
            self.models.append(model)

    def predict(self, X):
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X)
        return predictions / self.B

    def score(self, X, Y):
        d1 = Y - self.predict(X)
        d2 = Y - Y.mean()
        return 1 - d1.dot(d1) / d2.dot(d2)


def get_data():
    df = pd.read_csv('data.csv')

    # df.drop(df.loc[df["# num_pages"] == "eng"].index, inplace=True)

    transformer = DataTransformer()

    # shuffle the data
    N = len(df)
    train_idx = np.random.choice(N, size=int(0.7 * N), replace=False)
    test_idx = [i for i in range(N) if i not in train_idx]
    df_train = df.loc[train_idx]
    df_test = df.loc[test_idx]

    Xtrain = transformer.fit_transform(df_train)
    Ytrain = df_train['average_rating'].values.astype(float)
    Ytrain = Ytrain / Ytrain.max()
    # Ytrain = np.log(df_train['average_rating'].values.astype(np.float))
    # train_nan_index = np.where(np.isnan(Xtrain))[0]
    # Xtrain = np.delete(Xtrain, train_nan_index, axis=0)
    # Ytrain = np.delete(Ytrain, train_nan_index, axis=0)

    Xtest = transformer.transform(df_test)
    Ytest = df_test['average_rating'].values.astype(float)
    Ytest = Ytest / Ytest.max()
    # test_nan_index = np.where(np.isnan(Xtest))[0]
    # Xtest = np.delete(Xtest, test_nan_index, axis=0)
    # Ytest = np.delete(Ytest, test_nan_index, axis=0)

    return Xtrain, Ytrain, Xtest, Ytest
