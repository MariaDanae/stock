import logging
from turtle import pd

import pandas
from pandas import np
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor


def create_features(df_stock, nlags=10):
    # df_resampled = df_stock.copy()
    # lags_col_names = []
    # for i in range(nlags + 1):
    #     df_resampled['lags_' + str(i)] = df_resampled['close'].shift(i)
    #     lags_col_names.append('lags_' + str(i))
    # df = df_resampled[lags_col_names]
    # print(df)
    # df = df.dropna(axis=0)

    # sort data by date
    df = df_stock.rename_axis('date').reset_index()
    df = df.sort_values(by=['date'])

    # create buy/sell column to be used in classification
    df['buysell'] = 1
    for index, row in df.iterrows():
        if index == 0:
            continue
        if df['close'][index - 1] < df['close'][index]:
            df['buysell'][index - 1] = 1  # buy
        else:
            df['buysell'][index - 1] = 0  # sell
    # remove last row since we don't know for tomrw
    df = df.head(-1)
    df['year'] = pandas.DatetimeIndex(df['date']).year
    df['month'] = pandas.DatetimeIndex(df['date']).month
    df['day'] = pandas.DatetimeIndex(df['date']).day
    df['weekday_name'] = df.date.dt.day_name()
    df = df.set_index('date')
    df = df.drop('ticker', 1)

    label_encoder = preprocessing.LabelEncoder()
    df['weekday_name'] = label_encoder.fit_transform(df['weekday_name'])

    return df


def create_X_Y(df_lags, y_col):
    X = df_lags.drop(y_col, axis=1)
    Y = df_lags[[y_col]]
    return X, Y


class Stock_model(BaseEstimator, TransformerMixin):

    def __init__(self, data_fetcher):
        self.log = logging.getLogger()
        self.lr = LinearRegression()
        self.rfc = RandomForestClassifier(n_estimators=100)
        self._data_fetcher = data_fetcher
        self.log.warning('here')

    def fit(self, X, Y=None):
        data = self._data_fetcher(X)
        df_features = create_features(data)
        df_features, Y = create_X_Y(df_features, 'buysell')
        self.rfc.fit(df_features, Y)
        return self

    def predict(self, X, Y=None):
        results = {}
        print(f'Get data for {X}')
        data = self._data_fetcher(X, last=True)
        print(f'Data:\n {data}')

        # sort data by date
        df = data.rename_axis('date').reset_index()
        df = df.sort_values(by=['date'])

        # create buy/sell column to be used in classification
        df['buysell'] = 1
        for index, row in df.iterrows():
            if index == 0:
                continue
            if df['close'][index-1] < df['close'][index]:
                df['buysell'][index-1] = 1 #buy
            else:
                df['buysell'][index-1] = 0 #sell
        # remove last row since we don't know for tomrw
        df = df.head(-1)
        df['year'] = pandas.DatetimeIndex(df['date']).year
        df['month'] = pandas.DatetimeIndex(df['date']).month
        df['day'] = pandas.DatetimeIndex(df['date']).day
        df['weekday_name'] = df.date.dt.day_name()
        df = df.set_index('date')
        df = df.drop('ticker', 1)
        label_encoder = preprocessing.LabelEncoder()
        df['weekday_name'] = label_encoder.fit_transform(df['weekday_name'])

        X, Y = create_X_Y(df, 'buysell')
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        clf = RandomForestClassifier(max_depth=2) #, max_features=2)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        results["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
        # results["important_features"] = pd.Series(clf.feature_importances_, index=iris.feature_names).sort_values(ascending=False)

        prediction = clf.predict(X_train)
        prediction = np.where(prediction == 1, "Buy", "Sell")
        results["prediction"] = prediction.flatten()[-1]

        return results
