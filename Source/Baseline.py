import numpy as np
from DataLoading import get_datasets


class Baseline:
    """"""

    def predict(self, X):
        y = X["grid1-loss-lagged"]

        return np.nan_to_num(y, nan=np.mean(y))