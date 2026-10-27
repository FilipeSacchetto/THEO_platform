"""THEO — data-agnostic, graph-informed time-series forecasting.

The package never assumes a specific domain (solar, wind, load, ...). The user
supplies a :class:`theo.config.ColumnSpec` that maps the columns of *any*
(entity, time) table to the roles the pipeline needs, plus run hyper-parameters.

Every model (ANN / LSTM / GNN) is fed the same THEO feature engineering: the
unit's own variables (node features) plus the neighbour-difference variables
(edge features). The only thing that changes between models is *how* the window
or the graph is consumed.

Public entry point: :func:`theo.pipeline.run_pipeline`.
"""

from .config import ColumnSpec, SplitConfig, TrainConfig
from .pipeline import run_pipeline
from .preprocessing.theo_graph import build_theo_features, delaunay_edges

__all__ = [
    "ColumnSpec", "SplitConfig", "TrainConfig", "run_pipeline",
    "build_theo_features", "delaunay_edges",
]
