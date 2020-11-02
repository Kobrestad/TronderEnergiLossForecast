import numpy as np  # linear algebra
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    make_scorer,
)
from sklearn.model_selection import cross_val_score, KFold, cross_validate
import DataLoading


def get_trained_model(x, y):
    return get_model_pipeline().fit(x, y)


def get_model_pipeline(model, name):
    return Pipeline(
        [
            ("imputer", SimpleImputer()),
            # ('scaling', StandardScaler()), #seems to have no effect whatsoever
            (
                "polys",
                PolynomialFeatures(2),
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


def print_predictions(y_true, y_pred):
    x_pred = np.arange(len(y_pred))
    plt.figure(1, figsize=(28, 3))  # the first figure
    plt.plot(x_pred, y_pred, color="green")
    plt.title("Predicted loss on test set")

    plt.figure(2, figsize=(28, 3))  # a second figure
    plt.plot(x_pred, y_true, color="red")
    plt.title("Actual loss on test set")

    plt.show()


(x_train, y_train), (x_test, y_test) = DataLoading.get_datasets(
    "Data/raw/train.csv", "Data/raw/test.csv"
)

# drop timestamps when using regression
x_train = x_train.drop(columns=["timestamp"])
x_test = x_test.drop(columns=["timestamp"])

# regressors = [
#     (LinearRegression(copy_X=True, n_jobs=6), 'linear'),
#     (RandomForestRegressor(n_estimators=10, criterion='mae', n_jobs=6), 'forest'),
# ]

linear_pipeline = get_model_pipeline(LinearRegression(copy_X=True, n_jobs=6), "linear")
random_forest_pipeline = get_model_pipeline(
    RandomForestRegressor(n_estimators=10, criterion="mae", n_jobs=6), "forest"
)

linear_scores = model_cross_validation(linear_pipeline, x_train, y_train)
random_forest_scores = model_cross_validation(random_forest_pipeline, x_train, y_train)

with open("results.txt", "a") as file:
    file.write(f"linear: {linear_scores}\trandom forest: {random_forest_scores}\n")

# model = linear_pipeline.fit(x_train, y_train)

# y_pred = model.predict(x_test)
# print(len(y_test), len(y_pred), len(x_test))
# mae = mean_absolute_error(y_test, y_pred)
# print(mae)
# print_predictions(y_test, y_pred)