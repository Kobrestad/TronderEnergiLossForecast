from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, median_absolute_error as medae

class Evaluation:
    def run(y_true, y_pred):
        metrics = [mae, mse, medae]

        return [calc(y_true, y_pred) for calc in metrics]
