"""Assemble per-time-step graph snapshots for :class:`~theo.models.gnn.TemporalGNN`.

Where the ANN/LSTM path works on a flattened per-entity window, the GNN sees the
*whole* graph at each time step: a node-feature matrix (all entities) plus the
edge index (entity -> neighbour). THEO's contribution is preserved: the node
features are the unit's own variables and the graph encodes which units are
neighbours; the learned aggregation replaces THEO's fixed absolute difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..config import ColumnSpec
from .loader import make_time_key
from ..preprocessing.windowing import _which


def build_graph_snapshots(df: pd.DataFrame, spec: ColumnSpec, window_size: int,
                          train_cutoff: int, val_cutoff: int, horizon: int = 1):
    """Build (train, val, test, tgt_min, tgt_range, labels) graph-snapshot samples.

    Parameters
    ----------
    df :
        Raw edge-level table (one row per (entity, time, neighbour)).
    spec :
        Column roles; ``spec.neighbor`` must be set (the graph structure).
    window_size :
        Number of time steps each window looks back over.
    train_cutoff, val_cutoff :
        Time-step indices delimiting train / val / test (shared with the ANN/LSTM
        path so every model is scored on the same temporal split).
    horizon :
        Steps ahead to predict.

    Each sample is a dict with:
      ``feats``   : list of tensors ``(n_nodes, n_node_features)``, one per step;
      ``edges``   : list of tensors ``(2, n_edges)``, one per step;
      ``y``       : tensor ``(n_nodes,)`` target in the original scale;
      ``present`` : boolean mask ``(n_nodes,)`` of units with a value at the step;
      ``tpos``    : the time key of the target step;
      ``split``   : ``"train"`` / ``"val"`` / ``"test"``.
    """
    if not spec.neighbor:
        raise ValueError("The GNN needs a neighbour column (spec.neighbor).")

    d = df.copy()
    d["time_key"] = make_time_key(d, spec.time, spec.year)

    # Integer node ids (order-preserving map of the entity labels), NaN-safe.
    labels = [v for v in pd.unique(d[spec.entity].astype(str)) if v != "nan"]
    node_id = {v: i for i, v in enumerate(labels)}
    n_nodes = len(labels)
    src = d[spec.entity].astype(str).map(node_id)
    dst = d[spec.neighbor].astype(str).map(node_id)

    # Node rows keep every unit; edge rows only where both endpoints are valid.
    node_df = d[src.notna()].copy()
    node_df["_src"] = src[src.notna()].astype(int)
    edge_df = d[src.notna() & dst.notna()].copy()
    edge_df["_src"] = src[src.notna() & dst.notna()].astype(int)
    edge_df["_dst"] = dst[src.notna() & dst.notna()].astype(int)

    node_features = spec.node_features
    train_mask = node_df["time_key"] < train_cutoff
    if not train_mask.any():
        raise ValueError("No training time steps; adjust the split fractions.")
    fmin = node_df.loc[train_mask, node_features].min()
    frange = (node_df.loc[train_mask, node_features].max() - fmin).replace(0, 1.0)
    tgt_min = float(node_df.loc[train_mask, spec.target].min())
    tgt_max = float(node_df.loc[train_mask, spec.target].max())
    tgt_range = (tgt_max - tgt_min) or 1.0

    months = sorted(node_df["time_key"].unique())

    feats_by_t, tgt_by_t, edges_by_t, present_by_t = {}, {}, {}, {}
    for t, g in node_df.groupby("time_key"):
        node_rows = g.drop_duplicates("_src")
        feat = np.zeros((n_nodes, len(node_features)), np.float32)
        raw = np.zeros(n_nodes, np.float32)
        present = np.zeros(n_nodes, bool)
        vals = ((node_rows[node_features] - fmin) / frange).to_numpy(np.float32)
        rawv = node_rows[spec.target].to_numpy(np.float32)
        for nid, row, r in zip(node_rows["_src"].to_numpy(), vals, rawv):
            feat[nid] = row
            raw[nid] = r
            present[nid] = True
        feats_by_t[t] = feat
        tgt_by_t[t] = raw
        present_by_t[t] = present
        eg = edge_df[edge_df["time_key"] == t]
        if len(eg):
            edges_by_t[t] = np.stack([eg["_src"].to_numpy(),
                                     eg["_dst"].to_numpy()]).astype(np.int64)
        else:
            edges_by_t[t] = np.zeros((2, 0), dtype=np.int64)

    samples = []
    for i in range(window_size + horizon - 1, len(months)):
        target_t = months[i]
        win = months[i - horizon - window_size + 1: i - horizon + 1]
        samples.append({
            "feats": [torch.from_numpy(feats_by_t[m]) for m in win],
            "edges": [torch.from_numpy(edges_by_t[m]) for m in win],
            "y": torch.from_numpy(tgt_by_t[target_t]),
            "present": torch.from_numpy(present_by_t[target_t]),
            "tpos": int(target_t),
            "split": _which(int(target_t), train_cutoff, val_cutoff),
        })

    train = [s for s in samples if s["split"] == "train"]
    val = [s for s in samples if s["split"] == "val"]
    test = [s for s in samples if s["split"] == "test"]
    if not train or not test:
        raise ValueError("Not enough time steps for the GNN. Reduce window_size.")
    return train, val, test, tgt_min, tgt_range, labels
