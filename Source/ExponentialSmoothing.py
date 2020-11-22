import DataLoading
from Evaluation import Evaluation
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def initial_trend(data, s_length):
    total = 0.0
    for i in range(s_length):
        total += float(data[i+s_length] - data[i]) / s_length
    return total / s_length

def initial_seasonals(data, s_length):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(data)/s_length)
    # season averages
    for j in range(n_seasons):
        index = s_length * j
        season_averages.append(sum(data[index:index+s_length])/float(s_length))
    # initial values
    for i in range(s_length):
        sum_over_avg = 0.0
        for j in range(n_seasons):
            sum_over_avg += data[s_length*j+i]-season_averages[j]
        seasonals[i] = sum_over_avg/n_seasons
    return seasonals

def exponential_smoothing_triple(data, s_length=24, alpha=0.8, beta=0.025, gamma=0.90, n_preds=24):
    result = []
    seasonals = initial_seasonals(data, s_length)
    for i in range(len(data)+n_preds):
        if i == 0: # initial values
            smooth = data[0]
            trend = initial_trend(data, s_length)
            result.append(data[0])
            continue
        if i >= len(data): # forecasting
            m = i - len(data) + 1
            result.append((smooth + m*trend) + seasonals[i%s_length])
        else: # training values
            val = data[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%s_length]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%s_length] = gamma*(val-smooth) + (1-gamma)*seasonals[i%s_length]
            result.append(smooth+trend+seasonals[i%s_length])
    return result

def plot_results(y_train, y_test, y_predicted):
    x_total_length = len(y_train) + len(y_test)
    x_train = np.arange(len(y_train))
    x_test = np.arange(start=len(y_train), stop=x_total_length)
    x_pred = np.arange(start=len(y_train), stop=x_total_length)
    plt.figure('1', figsize=(15, 3))
    plt.title("Predicted vs actual loss on test set")
    plt.plot(x_train, y_train, color="green", label="Actual loss in training set")
    plt.plot(x_test, y_test, color="blue", label="Actual loss in test set")
    plt.plot(x_pred, y_predicted, color="red", label="Predicted loss")
    plt.xlabel(f"timer")
    plt.ylabel(f"grid_loss")
    plt.grid()
    plt.legend(loc='best')

    plt.show()

def main():
    (x_train, y_train), (x_test, y_test) = DataLoading.get_datasets("Data/raw/train.csv", "Data/raw/test.csv")

    #print(y_train.tail())
    #print(y_test.head())

    y_pred = exponential_smoothing_triple(y_train.values, s_length=24 ,alpha=0.652, beta=0.028, gamma=0.932, n_preds=len(y_test))
    y_predicted = y_pred[-len(y_test):]

    # Evaluation array is mean_absolute_error, mean_squared_error, median_absolute_error respectively
    evaluation = Evaluation.run(y_test.values, y_predicted)
    print(f"Evaluation results from whole prediction: {evaluation}")

    evaluation = Evaluation.run(y_test[:23], y_predicted[:23])
    print(f"Evaluation results for next 24 hrs: {evaluation}")

    evaluation = Evaluation.run(y_test[167:191], y_predicted[167:191])
    print(f"Evaluation results for a next day a week into the future: {evaluation}")

    plot_results(y_train, y_test, y_predicted)

if __name__ == "__main__":
    main()
