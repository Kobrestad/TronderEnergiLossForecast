import pandas as pd


def shift_pandas_column(column, periods=24 * 7):
    return column.shift(periods, fill_value=column.mean())


def get_datasets(
    train_location="/kaggle/input/grid-loss-time-series-dataset/train.csv",
    test_location="/kaggle/input/grid-loss-time-series-dataset/test.csv",
):
    columns_to_drop = [
        "grid2-load",
        "grid2-loss",
        "grid2-loss-prophet-daily",
        "grid2-loss-prophet-pred",
        "grid2-loss-prophet-trend",
        "grid2-loss-prophet-weekly",
        "grid2-loss-prophet-yearly",
        "grid2_1-temp",
        "grid2_2-temp",
        "grid3-load",
        "grid3-loss",
        "grid3-loss-prophet-daily",
        "grid3-loss-prophet-pred",
        "grid3-loss-prophet-trend",
        "grid3-loss-prophet-weekly",
        "grid3-loss-prophet-yearly",
        "grid3-temp",
    ]

    raw_train = pd.read_csv(train_location, parse_dates=True).dropna(0)
    raw_test = pd.read_csv(test_location, parse_dates=True).dropna(0)
    pruned_train = raw_train.drop(columns=columns_to_drop)
    pruned_test = raw_test.drop(columns=columns_to_drop)

    # get y values before shifting
    train_y = pruned_train["grid1-loss"].copy()
    test_y = pruned_test["grid1-loss"].copy()

    # shift grid loss and load features 1 week to emulate real world delay of measurements
    pruned_train["grid1-loss"] = shift_pandas_column(pruned_train["grid1-loss"])
    pruned_train["grid1-load"] = shift_pandas_column(pruned_train["grid1-load"])
    pruned_test["grid1-loss"] = shift_pandas_column(pruned_test["grid1-loss"])
    pruned_test["grid1-load"] = shift_pandas_column(pruned_test["grid1-load"])

    # give name to first column
    pruned_train.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)
    pruned_test.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)
    pruned_train.rename(columns={"grid1-loss": "grid1-loss-lagged"}, inplace=True)
    pruned_test.rename(columns={"grid1-loss": "grid1-loss-lagged"}, inplace=True)

    train = (pruned_train, train_y)
    test = (pruned_test, test_y)

    return train, test


# get_datasets('Data/raw/train.csv', 'Data/raw/test.csv')
