import DataLoading
from Evaluation import Evaluation
from ExponentialSmoothing import exponential_smoothing_triple

(x_train, y_train), (x_test, y_test) = DataLoading.get_datasets(
    "Data/raw/train.csv", "Data/raw/test.csv"
)

# y_pred = exponential_smoothing_triple(y_test)
# y_pred = exponential_smoothing_triple(y_test, alpha=0.64, beta=0.03, gamma=0.89)
y_pred = exponential_smoothing_triple(
    y_train.values, alpha=0.652, beta=0.028, gamma=0.932, n_preds=len(y_test)
)
y_predicted = y_pred[-len(y_test) :]

evaluation = Evaluation.run(y_test.values[:23], y_predicted[:23])
print(f"Evaluation results: {evaluation}")


# use index_1 and index_2 to choose range of actual data to optimize parametres for
def optimizer(data, actual, index_1=0, index_2=47, eval_type=1, iterations=100):
    alpha, beta, gamma = 1.0 / iterations, 1.0 / iterations, 1.0 / iterations
    old_errors = Evaluation.run(
        actual[index_1:index_2],
        exponential_smoothing_triple(
            data, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(actual)
        )[index_1 + len(data) : index_2 + len(data)],
    )
    for i in range(0, iterations):
        temp_alpha = alpha + 1 / iterations
        errors = Evaluation.run(
            actual[index_1:index_2],
            exponential_smoothing_triple(
                data, alpha=temp_alpha, beta=beta, gamma=gamma, n_preds=len(actual)
            )[index_1 + len(data) : index_2 + len(data)],
        )
        if errors[eval_type] < old_errors[eval_type]:
            alpha = temp_alpha
            old_errors = errors

        temp_beta = beta + 1 / iterations
        errors = Evaluation.run(
            actual[index_1:index_2],
            exponential_smoothing_triple(
                data, alpha=alpha, beta=temp_beta, gamma=gamma, n_preds=len(actual)
            )[index_1 + len(data) : index_2 + len(data)],
        )
        if errors[eval_type] < old_errors[eval_type]:
            beta = temp_beta
            old_errors = errors

        temp_gamma = gamma + 1 / iterations
        errors = Evaluation.run(
            actual[index_1:index_2],
            exponential_smoothing_triple(
                data, alpha=alpha, beta=beta, gamma=temp_gamma, n_preds=len(actual)
            )[index_1 + len(data) : index_2 + len(data)],
        )
        if errors[eval_type] < old_errors[eval_type]:
            gamma = temp_gamma
            old_errors = errors

    return alpha, beta, gamma


alpha, beta, gamma = optimizer(
    y_train.values,
    y_test.values,
    index_1=0,
    index_2=24 * 8,
    eval_type=1,
    iterations=250,
)

print(f"Best alpha: {alpha}\nBest beta: {beta}\nBest gamma: {gamma}")
