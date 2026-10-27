"""Data-agnostic loading and reshaping of the input table.

The pipeline works on a *long* table: one row per (entity, time) carrying the
unit's own (node) variables, its neighbour-difference (edge) variables and the
target. This module turns the user's raw table — which may carry one row per
(entity, time, neighbour) edge — into that long form. Nothing about the domain
is assumed; the user's :class:`~theo.config.ColumnSpec` names the columns.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from ..config import ColumnSpec

logger = logging.getLogger(__name__)

# Domain-neutral names the auto-detection looks for first, in priority order.
_ENTITY_NAMES = ("source", "entity", "node_id", "station", "device", "sensor", "id")
_TIME_NAMES = ("month", "time", "date", "period", "day")
_TARGET_NAMES = ("ALLSKY_SFC_SW_DWN", "radiation_node", "radiation", "target",
                 "label", "value", "y")
_X_NAMES = ("lon", "longitude", "x", "x_pos", "px")
_Y_NAMES = ("lat", "latitude", "y", "y_pos", "py")


def load_table(source) -> pd.DataFrame:
    """Read a table from a path or an upload object (e.g. Streamlit file).

    A leading unnamed index column (``Unnamed: 0`` from a ``to_csv`` with an
    index) is dropped so it is never mistaken for a data variable.
    """
    if isinstance(source, (str, os.PathLike)):
        source = str(source)
    try:
        df = pd.read_csv(source)
    except pd.errors.ParserError:
        df = pd.read_csv(source, sep=";")
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def auto_detect_columns(df: pd.DataFrame) -> ColumnSpec:
    """Best-effort :class:`ColumnSpec` from column names.

    Understands the project's native flattened-graph schema (``source``/``target``
    ids, ``*_node`` and ``*_edge`` variables, a ``month``/``year`` time) and,
    failing that, sensible generic fallbacks. The result is always editable in
    the UI, so a wrong guess is only a starting point.
    """
    cols = list(df.columns)

    def pick(*names):
        return next((c for c in names if c in cols), None)

    node = [c for c in cols if c.lower().endswith("_node")]
    edge = [c for c in cols if c.lower().endswith("_edge")]

    entity = pick(*_ENTITY_NAMES)
    # In the native flattened-graph schema the neighbour id is named "target";
    # in a generic table that name more likely holds the prediction target,
    # so prefer neutral names there.
    if entity == "source":
        neighbor = pick("target", "neighbor", "dst")
    else:
        neighbor = pick("neighbor", "dst", "neighbor_id")
    time = pick(*_TIME_NAMES)
    year = pick("year", "yr")
    x = pick(*_X_NAMES)
    y = pick(*_Y_NAMES)

    # Clean-table THEO mode: positions are present and there are no pre-built
    # ``_node``/``_edge`` columns, so the graph must be built (Delaunay) and the
    # THEO features computed from the unit's own numeric variables.
    if x and y and not node and not edge:
        excluded = {entity, time, year, x, y}
        theo_variables = [c for c in cols
                          if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
        target = pick(*_TARGET_NAMES) or (theo_variables[0] if theo_variables else None)
        return ColumnSpec(
            entity=entity, time=time, target=target, year=year,
            theo_variables=theo_variables, x_col=x, y_col=y,
        )

    # Pre-built THEO table (``_node``/``_edge`` already present) or generic fallback.
    target = pick(*_TARGET_NAMES) or (node[0] if node else (cols[0] if cols else None))
    return ColumnSpec(
        entity=entity,
        time=time,
        target=target,
        node_features=node,
        edge_features=edge,
        neighbor=neighbor,
        year=year,
    )


def make_time_key(df: pd.DataFrame, time_col: str,
                  year_col: str | None = None) -> pd.Series:
    """Build a monotonically orderable time key from the user's time column.

    Handles datetimes, numerics, the ``"month_year_index"`` label style, an
    explicit year column, and falls back to first-appearance order. The result
    only needs to be orderable (it is compared with ``<``).
    """
    s = df[time_col]
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        # (period, year) pattern: a small integer period (month, week, day-in-year)
        # that repeats across years, paired with a separate year column. Without
        # this, a bare "month" column (1-12) would collapse every year together.
        if (
            year_col is not None and year_col in df.columns
            and pd.api.types.is_numeric_dtype(df[year_col])
            and s.nunique(dropna=True) <= 120
            and df[year_col].nunique(dropna=True) >= 2
        ):
            periods = sorted(pd.unique(s.dropna()))
            rank = {p: i for i, p in enumerate(periods)}
            return df[year_col].astype(float) * len(periods) + s.map(rank).astype(float)
        return s
    # "month_year_index" style label (e.g. "1_1984_0") -> ordinal months.
    # Checked before date parsing because to_datetime may not raise on labels.
    sample = s.dropna().iloc[0] if s.notna().any() else None
    if isinstance(sample, str) and sample.count("_") == 2:
        p = s.astype(str).str.split("_")
        return (p.str[1].astype(int) * 12) + p.str[0].astype(int)
    # Real dates or date-strings.
    try:
        parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().any():
            return parsed
    except (ValueError, TypeError):
        pass
    if (year_col and year_col in df.columns
            and pd.api.types.is_numeric_dtype(df[year_col])):
        return df[year_col].astype(float)
    return pd.factorize(s)[0].astype(int)


def aggregate_entity_time(df: pd.DataFrame, spec: ColumnSpec) -> pd.DataFrame:
    """Collapse an edge-level table to one row per (entity, time).

    Node and edge variables are averaged over the neighbours present at each
    (entity, time) — the same neighbourhood aggregation used by the thesis. The
    target column keeps its original name (it may also be one of the node
    features, e.g. the unit's own past values).

    Returns a long table with columns ``entity``, ``time_key`` and the original
    feature/target column names.
    """
    spec.validate(df.columns)
    out = df.copy()
    out["time_key"] = make_time_key(out, spec.time, spec.year)
    feats = spec.all_features()
    agg = {c: "mean" for c in feats}
    agg[spec.target] = "mean"
    grouped = out.groupby([spec.entity, "time_key"], as_index=False).agg(agg)
    out_df = grouped.rename(columns={spec.entity: "entity"})
    logger.info(
        "Aggregated %d edge-level rows -> %d (entity, time) rows "
        "(entity=%s, time=%s, target=%s)",
        len(df), len(out_df), spec.entity, spec.time, spec.target,
    )
    return out_df
