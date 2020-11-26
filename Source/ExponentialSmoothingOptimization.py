import numpy as np
import DataLoading
from Evaluation import Evaluation
from ExponentialSmoothing import (
    holt_winters,
    holt_winters_online,
)

(x_train, y_train), (x_test, y_test) = DataLoading.get_datasets(
    "Data/raw/train.csv", "Data/raw/test.csv"
)

# y_pred = holt_winters(y_test)
# y_pred = holt_winters(y_test, alpha=0.64, beta=0.03, gamma=0.89)
y_pred = holt_winters(
    y_train.values, alpha=0.652, beta=0.028, gamma=0.932, n_preds=len(y_test)
)
y_predicted = y_pred[-len(y_test) :]

evaluation = Evaluation.run(y_test.values[:23], y_predicted[:23])
print(f"Evaluation results: {evaluation}")

# Method to brute force testing of different parameters.
def optimizer(data, actual, eval_type=1, iterations=100):
    alpha, beta, gamma = 1.0 / iterations, 1.0 / iterations, 1.0 / iterations
    total_actual = np.append(data, actual)
    old_errors = Evaluation.run(
        total_actual,
        holt_winters(data, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(actual)),
    )
    for i in range(0, iterations):
        temp_alpha = alpha + 1 / iterations
        errors = Evaluation.run(
            total_actual,
            holt_winters(
                data, alpha=temp_alpha, beta=beta, gamma=gamma, n_preds=len(actual)
            ),
        )
        if errors[eval_type] < old_errors[eval_type]:
            alpha = temp_alpha
            old_errors = errors

        temp_beta = beta + 1 / iterations
        errors = Evaluation.run(
            total_actual,
            holt_winters(
                data, alpha=alpha, beta=temp_beta, gamma=gamma, n_preds=len(actual)
            ),
        )
        if errors[eval_type] < old_errors[eval_type]:
            beta = temp_beta
            old_errors = errors

        temp_gamma = gamma + 1 / iterations
        errors = Evaluation.run(
            total_actual,
            holt_winters(
                data, alpha=alpha, beta=beta, gamma=temp_gamma, n_preds=len(actual)
            ),
        )
        if errors[eval_type] < old_errors[eval_type]:
            gamma = temp_gamma
            old_errors = errors

    return alpha, beta, gamma


print("\nOffline parameter optimalization...")

# Split training data into part to train of, and part to predict
# Split is 70/30
train_values = y_train.values[: int(len(y_train.values) * 0.7)]
test_values = y_train.values[len(train_values) :]

alpha, beta, gamma = optimizer(
    train_values,
    test_values,
    eval_type=1,
    iterations=250,
)

print(f"Best alpha: {alpha}\nBest beta: {beta}\nBest gamma: {gamma}")


# takes long time to run, isn't really effective unless iteration count is high
def optimizer_online(data, actual, eval_type=1, iterations=4):
    y_total = np.append(data, actual)
    alpha, beta, gamma = 1.0 / iterations, 1.0 / iterations, 1.0 / iterations
    y_pred = holt_winters_online(data, actual, alpha=alpha, beta=beta, gamma=gamma)
    old_errors = Evaluation.run(y_total[len(y_total) - len(y_pred) :], y_pred)

    for i in range(0, iterations):
        print(f"Iteration No. {i}")
        temp_alpha = alpha + 1 / iterations
        y_pred = holt_winters_online(
            data, actual, alpha=temp_alpha, beta=beta, gamma=gamma
        )
        errors = Evaluation.run(y_total[len(y_total) - len(y_pred) :], y_pred)
        if errors[eval_type] < old_errors[eval_type]:
            alpha = temp_alpha
            old_errors = errors

        temp_beta = beta + 1 / iterations
        y_pred = holt_winters_online(
            data, actual, alpha=alpha, beta=temp_beta, gamma=gamma
        )
        errors = Evaluation.run(y_total[len(y_total) - len(y_pred) :], y_pred)
        if errors[eval_type] < old_errors[eval_type]:
            beta = temp_beta
            old_errors = errors

        temp_gamma = gamma + 1 / iterations
        y_pred = holt_winters_online(
            data, actual, alpha=alpha, beta=beta, gamma=temp_gamma
        )
        errors = Evaluation.run(y_total[len(y_total) - len(y_pred) :], y_pred)
        if errors[eval_type] < old_errors[eval_type]:
            gamma = temp_gamma
            old_errors = errors

    return alpha, beta, gamma


# print("\nOnline parameter optimization...")

# alpha, beta, gamma = optimizer_online(
#    y_train.values,
#    y_test.values,
#    eval_type=1,
#    iterations=4,
# )

# print(f"Best alpha: {alpha}\nBest beta: {beta}\nBest gamma: {gamma}")
