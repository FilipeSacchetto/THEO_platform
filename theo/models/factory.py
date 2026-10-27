"""Factory for the generic (non-graph) predictors."""

from __future__ import annotations

from .ann import ANNRegressor
from .lstm import LSTMRegressor

# Every model offered by the pipeline. "gnn" is built separately (it consumes a
# graph, not a flat window), so it is not part of :func:`build_predictor`.
MODELS = ("ann", "lstm", "gnn")


def build_predictor(name: str, window_size: int, n_features: int):
    """Instantiate an ANN or LSTM from its name (case-insensitive)."""
    name = name.lower()
    if name == "ann":
        return ANNRegressor(window_size, n_features)
    if name == "lstm":
        return LSTMRegressor(n_features)
    raise ValueError(f"Unknown predictor {name!r} (use 'ann' or 'lstm').")
