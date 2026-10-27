"""Min-max scaling fitted on the training split only (no data leakage)."""

from __future__ import annotations

import numpy as np


class MinMax:
    """A small column-wise min-max scaler.

    It is fitted only on the training rows so that validation and test values
    are scaled with parameters learned from the past alone.
    """

    def __init__(self):
        self.mins = None
        self.ranges = None

    def fit(self, X_train: np.ndarray) -> "MinMax":
        mins = np.nanmin(X_train, axis=0)
        ranges = np.nanmax(X_train, axis=0) - mins
        ranges = np.where(ranges == 0, 1.0, ranges)
        self.mins = mins.astype(np.float32)
        self.ranges = ranges.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mins is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.nan_to_num(np.asarray(X, dtype=np.float32))
        return ((X - self.mins) / self.ranges).astype(np.float32)
