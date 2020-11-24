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
    median_absolute_error,
    make_scorer,
)
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from DataLoading import get_datasets
from Baseline import Baseline
from Metrics import mean_absolute_percentage_error, root_mean_squared_error


def get_model_pipeline(model, name, poly_features=2):
    return Pipeline(
        [
            ("imputer", SimpleImputer()),
            # ('scaling', StandardScaler()), #seems to have no effect whatsoever
            (
                "polys",
                PolynomialFeatures(poly_features),
            ),  # increasing above 2 causes significantly longer training time.
            (name, model),
        ]
    )


def model_cross_validation(model, x, y, k=5):
    cv = KFold(n_splits=k, shuffle=True)

    return cross_val_score(
        model,
        x,
        y,
        cv=cv,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
    )


def get_trained_model(model, x_train, y_train, name="model"):
    return get_model_pipeline(model, name).fit(x_train, y_train)


def get_trained_linear_model(x_train, y_train):
    return get_trained_model(
        LinearRegression(copy_X=True, n_jobs=6), x_train, y_train, name="linear"
    )


def get_trained_random_forest_model(x_train, y_train, n_estimators=10):
    return get_trained_model(
        RandomForestRegressor(n_estimators=n_estimators, criterion="mae", n_jobs=6),
        x_train,
        y_train,
        "forest",
    )


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    mape = mean_absolute_percentage_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

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


def training_set_experiment(model, name="model"):
    (x_train, y_train), _ = get_data()
    pipeline = get_model_pipeline(model, name)
    scores = model_cross_validation(pipeline, x_train, y_train)
    print(scores, scores.mean())

    return scores


def training_set_experiment_random_forest(n_estimators=10):
    return training_set_experiment(
        RandomForestRegressor(n_estimators=n_estimators, criterion="mae", n_jobs=6),
        "forest",
    )


def training_set_experiment_linear():
    return training_set_experiment(LinearRegression(copy_X=True, n_jobs=6), "linear")


def training_set_experiment_gb(n_estimators=10):
    return training_set_experiment_random_forest(
        GradientBoostingRegressor(n_estimators=n_estimators)
    )


def final_evaluation(model, backfilled_test=False):
    _, (x_test, y_test) = get_data(backfilled_test=backfilled_test)

    return evaluate_model(model, x_test, y_test)


def final_evaluation_linear(backfilled_test=False):
    (x_train, y_train), _ = get_data()

    return final_evaluation(
        get_trained_linear_model(x_train, y_train), backfilled_test=backfilled_test
    )


def final_evaluation_random_forest(n_estimators=10, backfilled_test=False):
    (x_train, y_train), _ = get_data()

    return final_evaluation(
        get_trained_random_forest_model(x_train, y_train, n_estimators=n_estimators),
        backfilled_test=False,
    )


def final_evaluation_baseline(backfilled_test=False):
    return final_evaluation(Baseline(), backfilled_test=backfilled_test)


def compare(backfilled_test=False):
    baseline = final_evaluation_baseline(backfilled_test=backfilled_test)
    print(baseline)

    linear = final_evaluation_linear(backfilled_test=backfilled_test)
    print(linear)

    random_forest_10 = final_evaluation_random_forest(backfilled_test=backfilled_test)
    print(random_forest_10)

    random_forest_50 = final_evaluation_random_forest(
        n_estimators=50, backfilled_test=backfilled_test
    )
    print(random_forest_50)

    return baseline, linear, random_forest_10, random_forest_50


print(compare(backfilled_test=False))