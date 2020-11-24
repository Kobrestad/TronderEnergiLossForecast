import numpy as np


class Baseline:
    """"""

    def predict(self, X):
        y = X["grid1-loss-lagged"]

        return np.nan_to_num(y, nan=np.mean(y))