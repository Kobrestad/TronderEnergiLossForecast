import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    make_scorer,
)
from sklearn.model_selection import KFold, GridSearchCV
from DataLoading import get_datasets
from Baseline import Baseline
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from Evaluation import Evaluation


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
                ("knn_imputer", KNNImputer()),
                (
                    "polys",
                    PolynomialFeatures(),
                ),
                (
                    "selection",
                    SelectFromModel(RandomForestRegressor(n_estimators=10, n_jobs=6)),
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

    @staticmethod
    def get_trained_model():
        (x_train, y_train), _ = get_data()

        params = GridLossLinearModel.__grid_search_params(
            {
                "linear__fit_intercept": [True, False],
                "linear__normalize": [True, False],
                "polys__degree": [1, 2, 3],
            },
            x_train,
            y_train,
        )

        return GridLossLinearModel.get_untrained_model(params).fit(x_train, y_train)

    @staticmethod
    def __grid_search_params(param_grid, x, y):
        grid = GridSearchCV(
            estimator=GridLossLinearModel.get_pipeline(),
            param_grid=param_grid,
            scoring=make_scorer(mean_absolute_error, greater_is_better=False),
            n_jobs=6,
            cv=KFold(n_splits=5, shuffle=True),
        )

        results = grid.fit(x, y)

        selection_indeces = np.where(
            results.best_estimator_.named_steps["selection"].get_support()
        )[0]

        feature_names = results.best_estimator_.named_steps["polys"].get_feature_names(
            x.columns
        )

        selected_features = [feature_names[i] for i in selection_indeces]

        print("all feature names:", selected_features)
        print("optimal parameters:", results.best_params_)

        return results.best_params_

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


def get_data(backfilled_test=False):
    arg = {}
    if backfilled_test:
        arg = {"test_location": "Data/raw/test_backfilled_missing_data.csv"}

    (x_train, y_train), (x_test, y_test) = get_datasets(**arg)
    x_train = x_train.drop(columns=["timestamp"])
    x_test = x_test.drop(columns=["timestamp"])

    return (x_train, y_train), (x_test, y_test)


def run_evaluation(model, backfilled_test=False):
    _, (x_test, y_test) = get_data(backfilled_test=backfilled_test)

    predictions = model.predict(x_test)

    return Evaluation.run(y_test, predictions)


def run_online_evaluation(model, backfilled_test=False):
    _, (x_test, y_test) = get_data(backfilled_test=backfilled_test)

    predictions = model.online_prediction(x_test, y_test)

    y = y_test.iloc[24 : len(predictions) + 24]

    return Evaluation.run(y, predictions)


if __name__ == "__main__":
    print(run_evaluation(Baseline(), backfilled_test=False))
    print(
        run_online_evaluation(
            GridLossLinearModel.get_trained_model(), backfilled_test=False
        )
    )
