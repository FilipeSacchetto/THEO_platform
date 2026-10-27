"""LSTM regressor: a stacked LSTM over the temporal window, last-step head."""

from __future__ import annotations

import torch.nn as nn


class LSTMRegressor(nn.Module):
    """Two stacked LSTM layers; the last time step drives a dense head.

    ``x`` has shape ``(batch, window_size, n_features)``.
    """

    def __init__(self, n_features: int, hidden=(64, 32)):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden[0], hidden[1], batch_first=True)
        self.head = nn.Linear(hidden[1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)
