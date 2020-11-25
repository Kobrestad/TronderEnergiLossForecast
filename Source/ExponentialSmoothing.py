import numpy as np
from matplotlib import pyplot as plt
import DataLoading
from Evaluation import Evaluation
from Metrics import mean_absolute_percentage_error, root_mean_squared_error

WEEK = 167
WEEK_PLUS_24 = 191


def initial_trend(data, s_length):
    total = 0.0
    for i in range(s_length):
        total += float(data[i + s_length] - data[i]) / s_length
    return total / s_length


def initial_seasonals(data, s_length):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(data) / s_length)
    # season averages
    for j in range(n_seasons):
        index = s_length * j
        season_averages.append(sum(data[index : index + s_length]) / float(s_length))
    # initial values
    for i in range(s_length):
        sum_over_avg = 0.0
        for j in range(n_seasons):
            sum_over_avg += data[s_length * j + i] - season_averages[j]
        seasonals[i] = sum_over_avg / n_seasons
    return seasonals


def holt_winters(data, s_length=24, alpha=0.005, beta=0.005, gamma=0.0625, n_preds=24):
    result = []
    seasonals = initial_seasonals(data, s_length)
    for i in range(len(data) + n_preds):
        if i == 0:  # initial values
            smooth = data[0]
            trend = initial_trend(data, s_length)
            result.append(data[0])
            continue
        if i >= len(data):  # forecasting
            m = i - len(data) + 1
            result.append((smooth + m * trend) + seasonals[i % s_length])
        else:  # training values
            val = data[i]
            last_smooth, smooth = (
                smooth,
                alpha * (val - seasonals[i % s_length])
                + (1 - alpha) * (smooth + trend),
            )
            trend = beta * (smooth - last_smooth) + (1 - beta) * trend
            seasonals[i % s_length] = (
                gamma * (val - smooth) + (1 - gamma) * seasonals[i % s_length]
            )
            result.append(smooth + trend + seasonals[i % s_length])
    return result


def holt_winters_online(
    data,
    test_data,
    s_length=24,
    alpha=0.005,
    beta=0.005,
    gamma=0.0625,
    n_preds=WEEK_PLUS_24,
):
    y_train = data
    y_test = test_data
    y_total = np.append(y_train, y_test)
    acc_res = []
    done = False
    i = 0
    while not done:
        i += 1
        index = i * 24
        if index > WEEK:
            data_so_far = y_total[:index]
            if y_total.size <= index:
                done = True
                break
            y_pred = holt_winters(
                data=data_so_far,
                s_length=s_length,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                n_preds=n_preds,
            )
            next_day_one_week_forward = y_pred[WEEK + index : WEEK_PLUS_24 + 1 + index]
            acc_res.extend(next_day_one_week_forward)

    return acc_res


def plot_online_results(y_train, y_test, y_predicted):
    y_total = np.append(y_train, y_test)
    length = len(y_predicted)
    x_length = np.arange(length)
    plt.figure("1", figsize=(10, 4))
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
    plt.xlabel(f"timer")
    plt.ylabel(f"grid_loss")
    plt.grid()
    plt.legend(loc="best")

    plt.show()


def plot_results(y_train, y_test, y_predicted):
    x_total_length = len(y_train) + len(y_test)
    x_train = np.arange(len(y_train))
    x_test = np.arange(start=len(y_train), stop=x_total_length)
    x_pred = np.arange(start=len(y_train), stop=x_total_length)
    plt.figure("1", figsize=(10, 4))
    plt.title("Predicted vs actual loss on test set")
    plt.plot(x_train, y_train, color="green", label="Actual loss in training set")
    plt.plot(x_test, y_test, color="blue", label="Actual loss in test set")
    plt.plot(x_pred, y_predicted, color="red", label="Predicted loss")
    plt.xlabel(f"timer")
    plt.ylabel(f"grid_loss")
    plt.grid()
    plt.legend(loc="best")

    plt.show()


def main():
    (x_train, y_train), (x_test, y_test) = DataLoading.get_datasets(
        "Data/raw/train.csv", "Data/raw/test.csv"
    )
    y_total = np.append(y_train.values, y_test.values)

    # alpha=0.652, beta=0.028, gamma=0.932 were found by optimizing for next 24 hrs
    # However default params are found after optimizing for whole dataset.
    y_pred = holt_winters(
        y_train.values,
        n_preds=len(y_test),
    )
    y_predicted = y_pred[-len(y_test) :]

    print("__________________Offline learning__________________")
    # Evaluation array is mean_absolute_error, mean_squared_error, median_absolute_error respectively
    evaluation = Evaluation.run(y_test.values, y_predicted)
    mape = mean_absolute_percentage_error(y_test.values, y_predicted)
    rmse = root_mean_squared_error(y_test.values, y_predicted)
    print(
        f"Evaluation results from whole prediction: {evaluation}, mape: {mape}, rmse: {rmse}"
    )

    evaluation = Evaluation.run(y_test[:24], y_predicted[:24])
    mape = mean_absolute_percentage_error(y_test.values[:24], y_predicted[:24])
    rmse = root_mean_squared_error(y_test.values[:24], y_predicted[:24])
    print(
        f"Evaluation results for next 24 hrs: {evaluation}, mape: {mape}, rmse: {rmse}"
    )

    evaluation = Evaluation.run(y_test[167:192], y_predicted[167:192])
    mape = mean_absolute_percentage_error(y_test.values[167:192], y_predicted[167:192])
    rmse = root_mean_squared_error(y_test.values[167:192], y_predicted[167:192])
    print(
        f"Evaluation results for next day a week into the future: {evaluation}, mape: {mape}, rmse: {rmse}"
    )

    plot_results(y_train, y_test, y_predicted)

    print("\n__________________Online learning__________________")
    print("All predictions are made one week into the future.")
    # Feed 24 hours more data at a time. Is better for total prediction, but less accurate on specifics.
    y_pred = holt_winters_online(y_train.values, y_test.values)

    evaluation = Evaluation.run(y_total[len(y_total) - len(y_pred) :], y_pred)
    mape = mean_absolute_percentage_error(y_total[len(y_total) - len(y_pred) :], y_pred)
    rmse = root_mean_squared_error(y_total[len(y_total) - len(y_pred) :], y_pred)
    print(
        f"Evaluation results from whole prediction: {evaluation}, mape: {mape}, rmse: {rmse}"
    )

    plot_online_results(y_train.values, y_test.values, y_pred)


if __name__ == "__main__":
    main()
