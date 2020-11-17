import logging
from turtle import pd

import pandas
from pandas import np
from datetime import datetime
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
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
    df = df[(df.date <= pandas.to_datetime('now')) &
            (df.date >= pandas.to_datetime('now') - pandas.DateOffset(months=24))]
    df = df.sort_values(by=['date'])
    df = df.reset_index(drop=True)

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
    df['month'] = pandas.DatetimeIndex(df['date']).month
    df['weekday_name'] = df.date.dt.day_name()
    label_encoder = preprocessing.LabelEncoder()
    df['weekday_name'] = label_encoder.fit_transform(df['weekday_name'])
    df = df.set_index('date')
    df = df.drop('ticker', 1)
    df.dropna()

    return df


def create_X_Y(df_lags, y_col):
    X = df_lags.drop(y_col, axis=1)
    Y = df_lags[[y_col]]
    return X, Y


class Stock_model(BaseEstimator, TransformerMixin):

    def __init__(self, data_fetcher):
        self.log = logging.getLogger()
        self.lr = LinearRegression()
        self.lgr = LogisticRegression()
        self.rfc = RandomForestClassifier(n_estimators=100)
        self._data_fetcher = data_fetcher
        self.log.warning('here')

    def fit(self, X, Y=None):
        data = self._data_fetcher(X)
        df_features = create_features(data)
        df_features, Y = create_X_Y(df_features, 'buysell')
        self.lgr.fit(df_features, Y)
        return self

    # def predict(self, X, Y=None):
    #     results = {}
    #     print(f'Get data for {X}')
    #     data = self._data_fetcher(X, last=True)
    #     print(f'Data:\n {data}')
    #
    #     # create features
    #     df = create_features(data)
    #
    #     # Split train test data
    #     X, Y = create_X_Y(df, 'buysell')
    #     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
    #
    #     # Model
    #     logmodel = LogisticRegression()
    #     logmodel.fit(X_train, y_train)
    #
    #     logmodel.fit(X_train, y_train)
    #     y_pred = logmodel.predict(X_test)
    #
    #     results["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
    #
    #     prediction = logmodel.predict(X_train)
    #     prediction = np.where(prediction == 1, "BUY", "SELL")
    #     results["prediction"] = prediction.flatten()[-1]
    #
    #     return results

    def predict(self, X, Y=None):
        results = {}
        print(f'Get data for {X}')
        data = self._data_fetcher(X, last=True)
        print(f'Data:\n {data}')

        # create features
        df = create_features(data)

        # Split test train data
        X, Y = create_X_Y(df, 'buysell')
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)

        # Base model to tune
        rf = RandomForestClassifier()
        param_grid = {
            'max_depth': [5, 10, 20, 40],
            'max_features': ['sqrt'],
            'min_samples_leaf': [2],
            'min_samples_split': [6],
            'n_estimators': [50, 100, 200]
        }
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_param = grid_search.best_params_
        clf = RandomForestClassifier(n_estimators=best_param['n_estimators'],
                                     min_samples_split=best_param['min_samples_split'],
                                     min_samples_leaf=best_param['min_samples_leaf'],
                                     max_features=best_param['max_features'],
                                     max_depth=best_param['max_depth'])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        results["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
        results["important_features"] = {"data": clf.feature_importances_, "index": X.columns}

        prediction = clf.predict(X_train)
        prediction = np.where(prediction == 1, "BUY", "SELL")
        results["prediction"] = prediction.flatten()[-1]

        return results
