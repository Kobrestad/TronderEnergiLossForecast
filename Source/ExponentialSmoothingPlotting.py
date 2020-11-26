import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

WEEK = 167
WEEK_PLUS_24 = 191


def plot_offline_results(y_train, y_test, y_predicted):
    x_total_length = len(y_train) + len(y_test)
    x_train = np.arange(len(y_train))
    x_test = np.arange(start=len(y_train), stop=x_total_length)
    x_pred = np.arange(start=len(y_train), stop=x_total_length)
    f = plt.figure("2", figsize=(10, 4))
    plt.title("Predicted vs actual loss on test set")
    plt.plot(x_train, y_train, color="green", label="Actual loss in training set")
    plt.plot(x_test, y_test, color="blue", label="Actual loss in test set")
    plt.plot(x_pred, y_predicted, color="red", label="Predicted loss")
    plt.xlabel(f"Hour (h)")
    plt.ylabel(f"Grid loss (MWh)")
    plt.grid()
    plt.legend(loc="best")

    plt.show()
    f.savefig("offline_pred.pdf", bbox_inches="tight")


def plot_online_results(y_train, y_test, y_predicted):
    y_total = np.append(y_train, y_test)
    length = len(y_predicted)
    # Since online doesn't include first week, shift x-index by a week.
    x_length = np.arange(start=WEEK, stop=length + WEEK)
    f = plt.figure("1", figsize=(10, 4))
    plt.title("Predicted vs actual loss on test set")
    plt.plot(x_length, y_predicted, color="red", label="Predicted loss")
    plt.plot(
        np.arange(len(y_train)),
        y_train,
        color="green",
        label="Actual loss: training data",
    )
    plt.plot(
        np.arange(start=len(y_train), stop=len(y_total)),
        y_test,
        color="blue",
        label="Actual loss: test data",
    )
    plt.xlabel(f"Hour (h)")
    plt.ylabel(f"Grid loss (MWh)")
    plt.grid()
    plt.legend(loc="best")

    plt.show()
    f.savefig("online_pred.pdf", bbox_inches="tight")


def plot_online_results_only_predictions(y_train, y_test, y_predicted):
    y_total = np.append(y_train, y_test)
    length = len(y_predicted)
    # Since online doesnt include first week, shift x-index by a week.
    x_length = np.arange(start=WEEK, stop=len(y_predicted) - len(y_train) + WEEK)
    f = plt.figure("1", figsize=(10, 4))
    plt.title("Predicted vs actual loss on test set")
    plt.plot(
        x_length,
        y_predicted[len(y_train + WEEK) :],
        color="red",
        label="Predicted loss",
    )
    plt.plot(
        np.arange(start=0, stop=len(y_test)),
        y_test,
        color="green",
        label="Actual loss: test data",
    )
    plt.xlabel(f"Hour (h)")
    plt.ylabel(f"Grid loss (MWh)")
    plt.grid()
    plt.legend(loc="best")

    plt.show()
    f.savefig("online_pred_only_test.pdf", bbox_inches="tight")
