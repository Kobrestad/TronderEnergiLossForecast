import numpy as np
from DataLoading import get_datasets


class Baseline:
    """"""

    def predict(self, X):
        y = X["grid1-loss-lagged"].fillna(method="ffill")

        return y