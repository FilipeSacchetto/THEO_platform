"""Per-entity temporal windowing — the corrected window function.

This is the windowing used throughout the thesis (reference:
``experiments/data_utils.py``) and is deliberately *per entity*: a window is a
run of ``window_size`` consecutive time steps of a single entity, the label is
the target ``horizon`` steps after the last window element, and a window is
assigned to train/val/test by the time of its *label*. Min-max scaling is fitted
on the training split only, so no future information leaks into the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from .scaling import MinMax


@dataclass
class WindowedDataset:
    """Windows for one feature set, split into train/val/test, ready to train."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    target_min: float
    target_range: float
    train_ids: List = field(default_factory=list)
    val_ids: List = field(default_factory=list)
    test_ids: List = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def inverse_target(self, y):
        """Undo the target scaling back to the original physical scale."""
        return np.asarray(y, dtype=np.float64) * self.target_range + self.target_min


def compute_cutoffs(time_keys, train_frac: float, val_frac: float):
    """Return (train_cutoff, val_cutoff) as actual time-key *values*.

    The cutoffs are expressed in the same space as ``time_keys`` (so they can be
    compared with ``<`` against a row's ``time_key``), computed from the fraction
    of the ordered unique time steps.
    """
    tvals = sorted(pd.unique(time_keys))
    n = len(tvals)
    if n < 2:
        raise ValueError("Need at least two distinct time steps to split.")
    ti = max(1, min(n - 1, int(round(n * train_frac))))
    vi = max(ti + 1, min(n - 1, int(round(n * val_frac))))
    return tvals[ti], tvals[vi]


def _which(pos: int, train_cutoff: int, val_cutoff: int) -> str:
    if pos < train_cutoff:
        return "train"
    if pos < val_cutoff:
        return "val"
    return "test"


def build_windows(table: pd.DataFrame,
                  feature_cols: List[str],
                  target_col: str,
                  window_size: int,
                  train_cutoff: int,
                  val_cutoff: int,
                  horizon: int = 1) -> WindowedDataset:
    """Build per-entity sliding time windows from a long (entity, time) table.

    Parameters
    ----------
    table :
        Long table with columns ``entity``, ``time_key`` (int ordinal) and the
        feature columns plus the target column (``target_col``).
    feature_cols :
        Input feature columns (THEO: node features + edge features).
    target_col :
        Name of the target column in ``table``.
    window_size :
        Number of consecutive time steps per window.
    train_cutoff, val_cutoff :
        Time-step indices delimiting train / val / test.
    horizon :
        Steps ahead to predict (1 = the very next step).
    """
    feature_cols = list(feature_cols)
    n_times = int(table["time_key"].nunique())
    if window_size < 1:
        raise ValueError("window_size must be >= 1.")

    # 1) Scale features on the training split only.
    train_mask = table["time_key"] < train_cutoff
    if not train_mask.any():
        raise ValueError("No training time steps; adjust the split fractions.")
    scaler = MinMax().fit(table.loc[train_mask, feature_cols].to_numpy(np.float32))
    scaled = table.copy()
    scaled[feature_cols] = scaler.transform(scaled[feature_cols].to_numpy(np.float32))

    # 2) Scale the target (fitted on train split only), kept for inverse transform.
    t_train = table.loc[train_mask, target_col].to_numpy(np.float64)
    tgt_min = float(np.nanmin(t_train))
    tgt_range = (float(np.nanmax(t_train)) - tgt_min) or 1.0
    target = (scaled[target_col].to_numpy(np.float64) - tgt_min) / tgt_range

    # 3) Sliding window per entity, labelled by the time of the target step.
    buckets = {"train": ([], [], []), "val": ([], [], []), "test": ([], [], [])}
    for entity, grp in scaled.groupby("entity"):
        grp = grp.sort_values("time_key")
        feats = grp[feature_cols].to_numpy(np.float32)
        pos = grp["time_key"].to_numpy()
        n = len(grp)
        for i in range(n - window_size - horizon + 1):
            window = feats[i:i + window_size]
            label_pos = i + window_size + horizon - 1
            bucket = _which(int(pos[label_pos]), train_cutoff, val_cutoff)
            Xs, ys, ids = buckets[bucket]
            Xs.append(window)
            ys.append(float(target[label_pos]))
            ids.append((str(entity), int(pos[label_pos])))

    if not buckets["train"][0] or not buckets["test"][0]:
        raise ValueError(
            "Not enough time steps to form train/test windows. "
            "Reduce window_size or provide a longer series."
        )

    width = (window_size, len(feature_cols))

    def xa(lst):
        a = np.asarray(lst, np.float32)
        return a if a.size else np.zeros((0, *width), np.float32)

    return WindowedDataset(
        X_train=xa(buckets["train"][0]),
        y_train=np.asarray(buckets["train"][1], np.float32),
        X_val=xa(buckets["val"][0]),
        y_val=np.asarray(buckets["val"][1], np.float32),
        X_test=xa(buckets["test"][0]),
        y_test=np.asarray(buckets["test"][1], np.float32),
        feature_names=feature_cols,
        target_min=tgt_min,
        target_range=tgt_range,
        train_ids=buckets["train"][2],
        val_ids=buckets["val"][2],
        test_ids=buckets["test"][2],
        meta={
            "window_size": window_size,
            "horizon": horizon,
            "n_entities": int(scaled["entity"].nunique()),
            "n_train": len(buckets["train"][0]),
            "n_val": len(buckets["val"][0]),
            "n_test": len(buckets["test"][0]),
        },
    )
