import numpy as np
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

    return results.best_params_


def get_trained_linear_model():
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

    return GridLossLinearModel.get_untrained_model(params).fit(x_train, y_train)


def final_evaluation_linear(backfilled_test=False):
    return final_evaluation(
        get_trained_linear_model(),
        backfilled_test=backfilled_test,
    ), final_online_evaluation(
        get_trained_linear_model(),
        backfilled_test=backfilled_test,
    )


def final_evaluation_baseline(backfilled_test=False):
    return final_evaluation(Baseline(), backfilled_test=backfilled_test)


if __name__ == "__main__":
    print(final_evaluation(Baseline(), backfilled_test=False))
    # print(final_evaluation(get_trained_linear_model(), backfilled_test=False))
    print(final_online_evaluation(get_trained_linear_model(), backfilled_test=False))
