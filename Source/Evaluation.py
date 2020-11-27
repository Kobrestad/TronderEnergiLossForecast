from sklearn.metrics import (
    mean_absolute_error as mae,
)
from Metrics import (
    mean_absolute_percentage_error as mape,
    root_mean_squared_error as rmse,
)


class Evaluation:
    @staticmethod
    def run(y_true, y_pred):
        return mae(y_true, y_pred), rmse(y_true, y_pred), mape(y_true, y_pred)
