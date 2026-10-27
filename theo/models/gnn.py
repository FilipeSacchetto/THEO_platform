"""Learned-aggregation GNN — the superset of THEO.

THEO fixes the neighbourhood message *outside* the network (the absolute
difference between two adjacent units' features, injected as an edge feature).
The GNN instead *learns* that aggregation with a message-passing layer, then
reads the time steps with an LSTM. Because a frozen absolute-difference
aggregation is just one special case of what this layer can learn, the GNN is a
strict generalisation of THEO.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GraphConv(nn.Module):
    """GCN-style convolution with mean aggregation over neighbours.

    ``h_i' = ReLU( W_self · h_i + W_neigh · mean_{j in N(i)} h_j )``

    Implemented with ``index_add_`` (scatter-mean), so it needs no external GNN
    library. Mean (not sum) aggregation keeps the scale independent of degree.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        agg = torch.zeros(n_nodes, x.size(1), device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, x[src])                      # gather by destination
        deg = torch.zeros(n_nodes, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(src, dtype=x.dtype))
        agg = agg / deg.clamp(min=1.0).unsqueeze(-1)        # normalise -> mean

        return self.act(self.lin_self(x) + self.lin_neigh(agg))


class TemporalGNN(nn.Module):
    """Per-time-step message passing, then an LSTM across the time steps.

    For each time step the graph conv layers mix each unit's features with its
    neighbours' (learned aggregation). The resulting embedding per time step is
    fed to an LSTM; the last step is mapped to a scalar prediction per unit.
    """

    def __init__(self, n_node_features: int, gnn_hidden: int = 16,
                 lstm_hidden: int = 32, n_layers: int = 2):
        super().__init__()
        dims = [n_node_features] + [gnn_hidden] * n_layers
        self.convs = nn.ModuleList(
            GraphConv(dims[i], dims[i + 1]) for i in range(n_layers)
        )
        self.lstm = nn.LSTM(gnn_hidden, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, 1)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Embed every unit in a single time step after message passing."""
        for conv in self.convs:
            x = conv(x, edge_index)
        return x  # (n_nodes, gnn_hidden)

    def forward(self, window_feats, window_edges):
        """Predict each unit's target ``horizon`` steps ahead.

        Parameters
        ----------
        window_feats : list of tensors ``(n_nodes, n_node_features)``, one per step.
        window_edges : list of tensors ``(2, n_edges)``, one per step.

        Returns
        -------
        Tensor ``(n_nodes,)`` — one prediction per unit.
        """
        embeddings = [self.encode(x, e) for x, e in zip(window_feats, window_edges)]
        seq = torch.stack(embeddings, dim=1)   # (n_nodes, window, gnn_hidden)
        out, _ = self.lstm(seq)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)
