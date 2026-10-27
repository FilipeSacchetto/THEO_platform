"""THEO graph feature construction — Delaunay triangulation over entity positions.

Given a clean ``(entity, time)`` table where each entity has a 2D position and a
set of variables, this builds THEO's graph:

1. a **Delaunay triangulation** of the entities' positions — the fixed neighbour
   structure (the same construction as ``TSGraphGenerator.py`` in the project
   root, which derives the edges from ``scipy.spatial.Voronoi(...).ridge_points``);
2. for every ``(entity, time)`` step and every Delaunay neighbour, one row with:

   ==============  =============================================================
   ``source``      the entity's own id
   ``adjacent_node``  the neighbour's id (one Delaunay edge endpoint)
   ``target``        the entity's own value of the prediction variable
   ``<v>_node``      the entity's own value of variable ``v``
   ``<v>_edge``      ``abs(entity_v - neighbour_v)``  (THEO's edge feature)
   ==============  =============================================================

The column names are **fixed** (``source``, ``adjacent_node``, ``target``,
``<v>_node``, ``<v>_edge``) so the downstream pipeline (windowing, models)
consumes every input uniformly, no matter the user's original schema. The
prediction ``target`` and the ``<v>_node`` feature for the same variable carry
the same *value* but at different time steps (the target is the step to predict,
the node feature sits in the look-back window), so there is no leakage.

This module is the graph-construction half of THEO; the consumer half is
``theo.pipeline.run_pipeline``.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from ..data.loader import make_time_key

logger = logging.getLogger(__name__)

# Fixed column names of the table this module produces.
SOURCE = "source"
ADJACENT = "adjacent_node"
TARGET = "target"
TIME_KEY = "time_key"


def delaunay_edges(points: np.ndarray) -> List[tuple[int, int]]:
    """Return the unique undirected edges of a Delaunay triangulation.

    ``points`` is an ``(n, 2)`` array of 2D positions (one row per entity). The
    result is a sorted list of index pairs ``(i, j)`` with ``i < j`` — the fixed
    neighbour structure THEO propagates information across.
    """
    from scipy.spatial import Delaunay  # imported here to keep the dependency optional

    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        raise ValueError(
            "Delaunay needs at least 3 distinct positions (got "
            f"{points.shape[0]}). THEO builds the graph from a 2D position per "
            "entity — check the position columns."
        )
    tri = Delaunay(points)
    pairs = set()
    for simplex in tri.simplices:
        a, b, c = (int(x) for x in simplex)
        pairs.add((min(a, b), max(a, b)))
        pairs.add((min(a, c), max(a, c)))
        pairs.add((min(b, c), max(b, c)))
    return sorted(pairs)


def build_theo_features(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    x_col: str,
    y_col: str,
    variables: List[str],
    target_col: str,
    year_col: Optional[str] = None,
) -> pd.DataFrame:
    """Build THEO's long feature table from a clean ``(entity, time)`` table.

    Parameters
    ----------
    df :
        One row per ``(entity, time)`` carrying each entity's position
        (``x_col``, ``y_col``) and its variables (``variables``) plus the
        prediction variable (``target_col``).
    variables :
        The unit's own variables. Each becomes a ``<v>_node`` (own value) and a
        ``<v>_edge`` (absolute difference to the Delaunay neighbour) column.
    target_col :
        The variable to predict. It is carried into the fixed ``target`` column;
        it is always included among the node/edge features as well (THEO uses the
        predicted variable as both a feature and the label, at different steps).
    year_col :
        Optional separate year column, combined with a repeating period in
        ``time_col`` to form a unique time order (see ``make_time_key``).

    Returns
    -------
    A long table — one row per ``(entity, time, neighbour)`` — with the fixed
    columns ``source``, ``adjacent_node``, ``target``, ``time_key`` and, for each
    variable, ``<v>_node`` and ``<v>_edge``.
    """
    variables = list(dict.fromkeys(list(variables) + [target_col]))  # unique, keep target
    for c in (entity_col, time_col, x_col, y_col, target_col, *variables):
        if c not in df.columns:
            raise ValueError(f"Column not in the table: {c}")

    # --- 1) one (x, y) position per entity ---------------------------------
    pos = (
        df[[entity_col, x_col, y_col]]
        .dropna()
        .groupby(entity_col, as_index=False)
        .agg("first")
        .drop_duplicates(entity_col)
    )
    entities = pos[entity_col].astype(str).to_list()
    points = pos[[x_col, y_col]].to_numpy(np.float64)
    ent_id = {e: i for i, e in enumerate(entities)}

    # --- 2) Delaunay neighbour structure (fixed, shared by every step) -----
    edges = delaunay_edges(points)
    src_idx = np.array([e[0] for e in edges], dtype=int)
    dst_idx = np.array([e[1] for e in edges], dtype=int)
    n_edges = len(edges)

    # --- 3) the time axis as a unique ordinal per row ----------------------
    tk = make_time_key(df, time_col, year_col).to_numpy()
    T = np.sort(pd.unique(tk))
    time_id = {t: i for i, t in enumerate(T)}
    row_t = np.array([time_id[t] for t in tk], dtype=int)
    row_e = df[entity_col].astype(str).map(ent_id).to_numpy(dtype=int)
    n_ent, n_t = len(entities), len(T)

    # --- 4) value matrices: V[v][entity, time] = value --------------------
    def value_matrix(col: str) -> np.ndarray:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
        M = np.full((n_ent, n_t), np.nan, dtype=np.float64)
        valid = ~np.isnan(vals)
        M[row_e[valid], row_t[valid]] = vals[valid]
        return M

    mats = {v: value_matrix(v) for v in variables}

    # --- 5) emit one row per (directed edge, time step), vectorised --------
    source = np.array(entities)[np.repeat(src_idx, n_t)]
    adjacent = np.array(entities)[np.repeat(dst_idx, n_t)]
    time_out = np.tile(T, n_edges)

    data = {
        SOURCE: source,
        ADJACENT: adjacent,
        TIME_KEY: time_out,
        TARGET: mats[target_col][src_idx, :].ravel(),
    }
    for v in variables:
        data[f"{v}_node"] = mats[v][src_idx, :].ravel()
        data[f"{v}_edge"] = np.abs(mats[v][src_idx, :] - mats[v][dst_idx, :]).ravel()

    out = pd.DataFrame(data)
    logger.info(
        "THEO graph built | entities=%d | delaunay_edges=%d | time_steps=%d | "
        "rows=%d | variables=%s",
        n_ent, n_edges, n_t, len(out), variables,
    )
    return out
