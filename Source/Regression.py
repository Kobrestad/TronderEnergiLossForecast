import numpy as np  # linear algebra
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    make_scorer,
)
from sklearn.model_selection import KFold, GridSearchCV
from DataLoading import get_datasets
from Baseline import Baseline
from Metrics import mean_absolute_percentage_error
from RegressionModels.Linear import GridLossLinearModel


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    mape = mean_absolute_percentage_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    return mae, rmse, mape


def evaluate_online_model(model, x_test, y_test):
    predictions = model.online_prediction(x_test, y_test)
    yo = y_test.iloc[24 : len(predictions) + 24]
    mape = mean_absolute_percentage_error(yo, predictions)
    rmse = np.sqrt(mean_squared_error(yo, predictions))
    mae = mean_absolute_error(yo, predictions)

    return mae, rmse, mape


def get_data(backfilled_test=False):
    arg = {}
    if backfilled_test:
        arg = {"test_location": "Data/raw/test_backfilled_missing_data.csv"}

    (x_train, y_train), (x_test, y_test) = get_datasets(
        exclude_columns=[
            "grid1-loss-prophet-daily",
            "grid1-loss-prophet-pred",
            "grid1-loss-prophet-trend",
            "grid1-loss-prophet-weekly",
            "grid1-loss-prophet-yearly",
        ],
        **arg
    )
    x_train = x_train.drop(columns=["timestamp"])
    x_test = x_test.drop(columns=["timestamp"])

    return (x_train, y_train), (x_test, y_test)


def final_evaluation(model, backfilled_test=False):
    _, (x_test, y_test) = get_data(backfilled_test=backfilled_test)

    return evaluate_model(model, x_test, y_test)


def final_online_evaluation(model, backfilled_test=False):
    _, (x_test, y_test) = get_data(backfilled_test=backfilled_test)

    return evaluate_online_model(model, x_test, y_test)


def grid_search_params(pipeline, param_grid, x, y):
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        n_jobs=6,
        cv=KFold(n_splits=5, shuffle=True),
    )

    results = grid.fit(x, y)
    print(results.best_score_)

    return results.best_params_


def final_evaluation_linear(backfilled_test=False):
    (x_train, y_train), _ = get_data()

    params = grid_search_params(
        GridLossLinearModel.get_pipeline(),
        {
            "linear__fit_intercept": [False, True],
            "linear__normalize": [True, False],
            "polys__degree": [1, 2, 3],
        },
        x_train,
        y_train,
    )

    return final_evaluation(
        GridLossLinearModel.get_untrained_model(params).fit(x_train, y_train),
        backfilled_test=backfilled_test,
    ), final_online_evaluation(
        GridLossLinearModel.get_untrained_model(params).fit(x_train, y_train),
        backfilled_test=backfilled_test,
    )


def final_evaluation_baseline(backfilled_test=False):
    return final_evaluation(Baseline(), backfilled_test=backfilled_test)


print(final_evaluation_linear())

# def compare(backfilled_test=False):
#     baseline = final_evaluation_baseline(backfilled_test=backfilled_test)
#     print(baseline)

#     linear = final_evaluation_linear(backfilled_test=backfilled_test)
#     print(linear)

#     random_forest_10 = final_evaluation_random_forest(backfilled_test=backfilled_test)
#     print(random_forest_10)

#     random_forest_50 = final_evaluation_random_forest(
#         n_estimators=50, backfilled_test=backfilled_test
#     )
#     print(random_forest_50)

#     return baseline, linear, random_forest_10, random_forest_50

# def training_set_experiment_random_forest(n_estimators=10):
#     return training_set_experiment(
#         RandomForestRegressor(n_estimators=n_estimators, criterion="mae", n_jobs=6),
#         "forest",
#     )
# def training_set_experiment_gb(n_estimators=10):
#     return training_set_experiment_random_forest(
#         GradientBoostingRegressor(n_estimators=n_estimators)
#     )

# def final_evaluation_random_forest(n_estimators=10, backfilled_test=False):
#     (x_train, y_train), _ = get_data()

#     return final_evaluation(
#         get_trained_random_forest_model(x_train, y_train, n_estimators=n_estimators),
#         backfilled_test=False,
#     )

# print(compare(backfilled_test=False))
# print(training_set_experiment_linear())
# print(final_evaluation_linear())

# sgd_param_grid = {
#     "sgd__loss": [
#         "squared_loss",
#         # "huber",
#         # "epsilon_insensitive",
#         # "squared_epsilon_insensitive",
#     ],
#     "sgd__penalty": ["l1", "l2"],
#     "sgd__alpha": [0.0001, 0.001, 0.00001],
#     "sgd__fit_intercept": [True, False],
#     # "sgd__epsilon": [0.01, 0.1, 1],
#     "sgd__learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
#     "sgd__eta0": [0.01, 0.05, 0.1],
# }

# linear_param_grid = {
#     "linear__fit_intercept": [False, True],
#     "linear__normalize": [True, False],
# }

# (x, y), _ = get_data()
# grid = GridSearchCV(
#     estimator=GridLossLinearModel.get_pipeline(),
#     param_grid=linear_param_grid,
#     scoring=make_scorer(mean_absolute_error, greater_is_better=False),
#     n_jobs=6,
#     cv=KFold(n_splits=5, shuffle=True),
# )
# grid.fit(x, y)
# print(grid)
# # summarize the results of the grid search
# print(grid.best_score_)
# print(grid.best_params_)