"""End-to-end THEO pipeline: load -> aggregate -> window/snapshot -> train -> eval.

This is the single orchestration point used by the Streamlit interface and by
any script. Given a table, a :class:`ColumnSpec` and a model name, it returns a
result dict with validation/test metrics and the per-(entity, time) predictions.
"""

from __future__ import annotations

import logging

import pandas as pd

from .config import ColumnSpec, SplitConfig, TrainConfig
from .data.loader import aggregate_entity_time, make_time_key
from .data.snapshots import build_graph_snapshots
from .evaluation.metrics import collect_gnn, collect_predictor
from .models.factory import build_predictor
from .models.gnn import TemporalGNN
from .preprocessing.theo_graph import build_theo_features
from .preprocessing.windowing import build_windows, compute_cutoffs
from .training.engine import choose_device, set_seed, train_gnn, train_predictor

logger = logging.getLogger(__name__)


def _emit(progress, frac: float, label: str) -> None:
    """Call the progress callback (if any) with a clamped fraction and label."""
    if progress:
        progress(float(min(max(frac, 0.0), 1.0)), label)


def run_pipeline(df: pd.DataFrame, spec: ColumnSpec, model: str,
                 train_cfg: TrainConfig, split_cfg: SplitConfig,
                 progress=None) -> dict:
    """Run THEO feature engineering + the chosen model and report metrics.

    Parameters
    ----------
    df :
        The user's table (edge-level or already per-entity).
    spec :
        Column roles (entity, time, target, node/edge features, neighbour).
    model :
        ``"ann"``, ``"lstm"`` or ``"gnn"``.
    train_cfg, split_cfg :
        Run hyper-parameters and the chronological split fractions.
    progress :
        Optional callback ``progress(fraction, label)`` with ``fraction`` in
        ``[0, 1]``; used by the interface to show a live progress bar.
    """
    model = model.lower()
    if model == "gnn" and not spec.neighbor and not spec.theo_mode:
        raise ValueError("The GNN needs a neighbour column (or THEO positions to build one).")

    device = choose_device(train_cfg.device)
    set_seed(train_cfg.seed)

    logger.info(
        "Run start | model=%s | rows=%d | entities=%d | window=%d | epochs=%d | "
        "seed=%d | device=%s | target=%s | node_features=%s | edge_features=%s",
        model, len(df), df[spec.entity].nunique() if spec.entity in df.columns else -1,
        train_cfg.window_size, train_cfg.epochs, train_cfg.seed, device,
        spec.target, spec.node_features, spec.edge_features,
    )

    # THEO (clean-table mode): build the Delaunay graph + node/edge features from
    # a clean (entity, time, position, variables) table, then switch to the spec
    # that consumes the fixed-name table. Downstream is identical to pre-built mode.
    if spec.theo_mode:
        _emit(progress, 0.03, "Building THEO graph (Delaunay triangulation) …")
        df = build_theo_features(
            df, entity_col=spec.entity, time_col=spec.time,
            x_col=spec.x_col, y_col=spec.y_col,
            variables=spec.theo_variables, target_col=spec.target,
            year_col=spec.year,
        )
        spec = spec.to_theo_spec()
        logger.info("THEO built | n_features=%d | entity=%s | target=%s",
                    len(spec.all_features()), spec.entity, spec.target)

    # Shared chronological split, computed once so every model is scored on the
    # same temporal boundaries.
    time_key = make_time_key(df, spec.time, spec.year)
    n_times = int(time_key.nunique())
    train_cutoff, val_cutoff = compute_cutoffs(
        time_key, split_cfg.train_frac, split_cfg.val_frac
    )
    logger.info(
        "Split | unique_times=%d | train_cutoff=%s | val_cutoff=%s | "
        "train_frac=%.2f | val_frac=%.2f",
        n_times, train_cutoff, val_cutoff, split_cfg.train_frac, split_cfg.val_frac,
    )

    result = {
        "model": model,
        "device": device,
        "n_entities": int(df[spec.entity].nunique()),
        "n_times": n_times,
        "n_features": len(spec.all_features()) if model != "gnn" else len(spec.node_features),
        "window_size": train_cfg.window_size,
        "split": {
            "train_cutoff": train_cutoff,
            "val_cutoff": val_cutoff,
            "train_frac": split_cfg.train_frac,
            "val_frac": split_cfg.val_frac,
        },
    }

    if model in ("ann", "lstm"):
        # THEO = node features + edge features, consumed as a per-entity window.
        _emit(progress, 0.05, "Aggregating (entity, time) rows …")
        long = aggregate_entity_time(df, spec)
        features = spec.all_features()
        _emit(progress, 0.15, "Building THEO temporal windows …")
        ds = build_windows(long, features, spec.target, train_cfg.window_size,
                           train_cutoff, val_cutoff, horizon=train_cfg.horizon)
        logger.info(
            "Windows | %s | train=%d val=%d test=%d | features=%d",
            ds.meta.get("n_entities"), ds.meta.get("n_train"),
            ds.meta.get("n_val"), ds.meta.get("n_test"), len(features),
        )
        result["meta"] = ds.meta
        model_obj = build_predictor(model, train_cfg.window_size, len(features))

        def _epoch(done: int, total: int) -> None:
            _emit(progress, 0.25 + 0.65 * done / total,
                  f"Training epoch {done}/{total}")

        _emit(progress, 0.25, f"Training … (epoch 0/{train_cfg.epochs})")
        m, trained = train_predictor(model_obj, ds, train_cfg, device,
                                     epoch_cb=_epoch if progress else None)
        _emit(progress, 0.93, "Evaluating on the validation and test splits …")
        preds = collect_predictor(trained, ds, device)
    else:  # gnn
        _emit(progress, 0.05, "Building per-time-step graph snapshots …")
        samples = build_graph_snapshots(df, spec, train_cfg.window_size,
                                        train_cutoff, val_cutoff,
                                        horizon=train_cfg.horizon)
        logger.info(
            "Graph snapshots | train=%d val=%d test=%d | node_features=%d",
            len(samples[0]), len(samples[1]), len(samples[2]),
            len(spec.node_features),
        )
        _, _, test, tgt_min, tgt_range, labels = samples
        model_obj = TemporalGNN(n_node_features=len(spec.node_features))

        def _epoch_gnn(done: int, total: int) -> None:
            _emit(progress, 0.25 + 0.65 * done / total,
                  f"Training epoch {done}/{total}")

        m, trained = train_gnn(model_obj, samples, train_cfg, device,
                               epoch_cb=_epoch_gnn if progress else None)
        _emit(progress, 0.93, "Evaluating on the validation and test splits …")
        preds = collect_gnn(trained, test, tgt_min, tgt_range, device, labels)

    result["val"] = m["val"]
    result["test"] = m["test"]
    result["n_test_pairs"] = len(preds)
    result["predictions"] = preds  # (entity, time_key) -> (y_true, y_pred)
    logger.info(
        "Run done | val=%s | test=%s | test_pairs=%d",
        m["val"], m["test"], len(preds),
    )
    _emit(progress, 1.0, "Done")
    return result
