"""Configuration objects for the THEO pipeline.

Everything here is domain-agnostic: the user tells the pipeline which columns
of *their* table play which role (entity, time, target, node/edge features) and
how to run the training. Nothing about solar irradiance is hard-coded.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ColumnSpec:
    """Maps the user's table to the pipeline's roles (all values are column names).

    Attributes
    ----------
    entity :
        Unique id of each observation unit (a "node": station, sensor, meter...).
    time :
        Column that orders rows in time. May be a real date, a number, or a label
        such as ``"1_1984_0"`` (see ``theo.data.loader.make_time_key``).
    target :
        The column to predict. It may also be one of ``node_features`` (e.g. the
        unit's own past values are a valid input), which is the usual case.
    node_features :
        The unit's own variables. Always used as THEO node features.
    edge_features :
        Neighbour-difference variables (THEO edge features). May be empty, in
        which case THEO reduces to node-only features.
    neighbor :
        Id of the neighbouring unit. Required only by the GNN (graph structure).
    year :
        Optional explicit year column, used when ``time`` is a non-numeric label.
    theo_variables :
        **Clean-table mode.** The unit's own variables to build THEO features
        from; each becomes ``<v>_node`` and ``<v>_edge``.
    x_col / y_col :
        **Clean-table mode.** The 2D position columns (e.g. lon/lat) used to
        build the Delaunay neighbour graph.
    """

    entity: str
    time: str
    target: str
    node_features: List[str] = field(default_factory=list)
    edge_features: List[str] = field(default_factory=list)
    neighbor: Optional[str] = None
    year: Optional[str] = None
    # Clean-table THEO mode: build the graph from a (entity, time) + positions table.
    theo_variables: List[str] = field(default_factory=list)
    x_col: Optional[str] = None
    y_col: Optional[str] = None

    @property
    def theo_mode(self) -> bool:
        """True when the graph must be built from a clean table (Delaunay)."""
        return (
            bool(self.theo_variables)
            and self.x_col is not None
            and self.y_col is not None
        )

    def to_theo_spec(self) -> "ColumnSpec":
        """The spec that consumes the fixed-name THEO table produced by
        :func:`theo.preprocessing.theo_graph.build_theo_features`."""
        return ColumnSpec(
            entity="source",
            time="time_key",
            target="target",
            node_features=[f"{v}_node" for v in self.theo_variables],
            edge_features=[f"{v}_edge" for v in self.theo_variables],
            neighbor="adjacent_node",
            year=None,
        )

    def all_features(self) -> List[str]:
        """Full THEO feature set = node features + edge features."""
        return list(self.node_features) + list(self.edge_features)

    def validate(self, columns) -> None:
        """Raise a friendly error listing any required column missing from ``columns``."""
        cols = set(columns)
        required = [self.entity, self.time, self.target]
        if self.theo_mode:
            required += [self.x_col, self.y_col, *self.theo_variables]
        else:
            required += list(self.node_features)
            if self.neighbor:
                required.append(self.neighbor)
        missing = [c for c in required if c and c not in cols]
        if missing:
            raise ValueError(f"Columns missing from the table: {missing}")


@dataclass
class SplitConfig:
    """Chronological split of the ordered time axis (fractions, cumulative).

    ``train_frac=0.70`` and ``val_frac=0.85`` means: the first 70% of the time
    steps are training, the next 15% validation, the last 15% test. A window is
    assigned to a split by the time of its *label* (the step it predicts), which
    keeps the split temporally honest (no future information leaks into train).
    """

    train_frac: float = 0.70
    val_frac: float = 0.85

    def __post_init__(self):
        if not (0.0 < self.train_frac < self.val_frac < 1.0):
            raise ValueError("Expected 0 < train_frac < val_frac < 1.")


@dataclass
class TrainConfig:
    """Hyper-parameters shared by every model (kept identical for fair runs).

    ``window_size`` is the number of time steps each window looks back over; the
    user controls it directly from the interface.
    """

    window_size: int = 24
    horizon: int = 1
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 32
    gnn_epochs: int = 15
    seed: int = 42
    device: str = "auto"
