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


def create_timeseries_figure(
    y,
    x=None,
    name="Figure",
    title="A figure",
    color="red",
    figsize=(10, 4),
    xlabel="Date",
    ylabel="Value",
):
    plot_date = True

    if x is None:
        x = np.arange(len(y))
        plot_date = False

    fig, ax = plt.subplots()

    if plot_date:
        ax.plot_date(x, y, color=color, xdate=True, linestyle="-", linewidth=1, fmt=",")
    else:
        ax.plot(x, y, color=color, linestyle="-", linewidth=1)

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )
    ax.grid()

    fig.set_size_inches(figsize)

    return fig


def main():
    train, _ = DataLoading.get_raw_datasets()
    x = pd.to_datetime(train["timestamp"])

    features_to_plot = [
        ["grid1-loss", "Training set grid loss", "Grid loss (MW)"],
        ["grid1-load", "Training set grid load", "Grid load (MW)", "green"],
        [
            "grid1-loss-prophet-pred",
            "Prophet predicted grid loss",
            "Predicted grid loss (MW)",
            "blue",
        ],
        [
            "grid1-loss-prophet-trend",
            "Prophet trend component of grid loss prediction",
            "Trend component (MW)",
            "blue",
        ],
        [
            "grid1-loss-prophet-daily",
            "Prophet daily component of grid loss prediction",
            "Daily component (MW)",
            "blue",
        ],
        [
            "grid1-loss-prophet-weekly",
            "Prophet weekly component of grid loss prediction",
            "Weekly component (MW)",
            "blue",
        ],
        [
            "grid1-loss-prophet-yearly",
            "Prophet yearly component of grid loss prediction",
            "Yearly component (MW)",
            "blue",
        ],
        [
            "grid1-temp",
            "Temperature forecast in area around grid",
            "Temperature (K)",
            "black",
        ],
        ["season_x", "Season x"],
        ["season_y", "Season y"],
        ["month_x", "Month x"],
        ["month_y", "Month y"],
        ["week_x", "Week x"],
        ["week_y", "Week y"],
        ["weekday_x", "Weekday x"],
        ["weekday_y", "Weekday y"],
        ["hour_x", "Hour x"],
        ["hour_y", "Hour y"],
        ["holiday", "Holiday"],
    ]

    for (feature, title, *args) in features_to_plot:
        ylabel = args[0] if len(args) else "Value"
        color = args[1] if len(args) > 1 else "red"
        fig = create_timeseries_figure(
            train[feature], x=x, title=title, ylabel=ylabel, color=color
        )
        fig.savefig(f"{title}.png")


main()