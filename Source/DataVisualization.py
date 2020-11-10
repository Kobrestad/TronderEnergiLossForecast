import DataLoading
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_predictions(y_true, y_pred):
    x_pred = np.arange(len(y_pred))
    plt.figure(1, figsize=(28, 3))  # the first figure
    plt.plot(x_pred, y_pred, color="green")
    plt.title("Predicted loss on test set")

    plt.figure(2, figsize=(28, 3))  # a second figure
    plt.plot(x_pred, y_true, color="red")
    plt.title("Actual loss on test set")

    plt.show()


def plot_timeseries_feature(
    y, x=None, name="Figure", title="A figure", color="red", figsize=(28, 3)
):
    plot_date = True

    if x is None:
        x = np.arange(len(y))
        plot_date = False

    plt.figure(name, figsize=figsize)  # a second figure

    if plot_date:
        plt.plot_date(
            x, y, color=color, xdate=True, linestyle="-", linewidth=1, fmt=","
        )
    else:
        plt.plot(x, y, color=color, linestyle="-", linewidth=1)

    plt.title(title)

    plt.show()


def main():
    train, _ = DataLoading.get_raw_datasets()
    x = pd.to_datetime(train["timestamp"])

    plot_timeseries_feature(train["grid1-loss"], x, title="Training set grid loss")
    plot_timeseries_feature(train["grid1-load"], x, title="Training set grid load")
    plot_timeseries_feature(
        train["grid1-loss-prophet-pred"], x, title="Prophet prediction"
    )
    plot_timeseries_feature(
        train["grid1-loss-prophet-trend"], x, title="Prophet trend component"
    )
    plot_timeseries_feature(
        train["grid1-loss-prophet-daily"], x, title="Prophet daily component"
    )
    plot_timeseries_feature(
        train["grid1-loss-prophet-weekly"], x, title="Prophet weekly component"
    )
    plot_timeseries_feature(
        train["grid1-loss-prophet-yearly"], x, title="Prophet yearly component"
    )
    plot_timeseries_feature(train["grid1-temp"], x, title="Temperature forecast")
    plot_timeseries_feature(train["season_x"], x, title="Season x")
    plot_timeseries_feature(train["season_y"], x, title="Season y")
    plot_timeseries_feature(train["month_x"], x, title="Month x")
    plot_timeseries_feature(train["month_y"], x, title="Month y")
    plot_timeseries_feature(train["week_x"], x, title="Week x")
    plot_timeseries_feature(train["week_y"], x, title="Week y")
    plot_timeseries_feature(train["weekday_x"], x, title="Weekday x")
    plot_timeseries_feature(train["weekday_y"], x, title="Weekday y")
    plot_timeseries_feature(train["hour_x"], x, title="Hour x")
    plot_timeseries_feature(train["hour_y"], x, title="Hour y")
    plot_timeseries_feature(train["holiday"], x, title="Holiday")


main()