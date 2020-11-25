from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import numpy as np


class GridLossLinearModel:
    __linear_model = None
    __train_x = None
    __train_y = None

    def __init__(self, linear_model_pipeline, train_x=None, train_y=None):
        self.__linear_model = linear_model_pipeline
        self.__train_x = train_x
        self.__train_y = train_y

    @staticmethod
    def __include_last_weeks_data(X, y, current_index):
        if current_index < 24 * 7 + 12:
            return None, None

        end = current_index - 24 * 7 + 12

        return X.iloc[:end], y[:end]

    @staticmethod
    def get_pipeline():
        return Pipeline(
            [
                ("imputer", SimpleImputer()),
                (
                    "polys",
                    PolynomialFeatures(),
                ),
                ("linear", LinearRegression(n_jobs=6)),
            ]
        )

    @staticmethod
    def get_untrained_model(params=None):
        if params is None:
            return GridLossLinearModel(GridLossLinearModel.get_pipeline())

        return GridLossLinearModel(
            GridLossLinearModel.get_pipeline().set_params(**params)
        )

    def fit(self, X, y):
        return GridLossLinearModel(self.__linear_model.fit(X, y), X, y)

    def predict(self, X):
        return self.__linear_model.predict(X)

    def online_prediction(self, X, y):
        predictions = list()

        for hour in range(12, len(X) - 24, 24):
            predictions.extend(self.__get_day_ahead_forecast(X, y, hour))
            self.__update_model(X, y, hour)

        return np.array(predictions)

    def __get_day_ahead_forecast(self, X, y, index):
        return self.__linear_model.predict(X.iloc[index + 12 : index + 36])

    def __update_model(self, X, y, current_index):
        new_X, new_y = GridLossLinearModel.__include_last_weeks_data(
            X, y, current_index
        )

        if new_X is None or new_y is None:
            return

        self.__linear_model.fit(
            self.__train_x.append(new_X), self.__train_y.append(new_y)
        )
