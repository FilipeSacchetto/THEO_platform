"""ANN regressor: a multilayer perceptron over the flattened temporal window."""

from __future__ import annotations

import torch.nn as nn


class ANNRegressor(nn.Module):
    """MLP over the window flattened to a single vector.

    ``x`` has shape ``(batch, window_size, n_features)``; the network flattens it
    and maps it to a single scalar prediction per sample.
    """

    def __init__(self, window_size: int, n_features: int,
                 hidden=(128, 64, 32)):
        super().__init__()
        in_dim = window_size * n_features
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(nn.Flatten(), *layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
