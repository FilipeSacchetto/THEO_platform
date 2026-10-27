"""Streamlit interface for the THEO forecasting pipeline.

The user:
  1. uploads (or points to) *any* (entity, time) table,
  2. maps the columns (auto-detected, then editable) — including the **target**,
  3. chooses the **window size** and a model (ANN / LSTM / GNN),
  4. runs it and reads validation/test metrics plus a per-(entity, time) table.

Every model is fed the same THEO feature engineering (node features +
neighbour-difference features); only the consumer changes.

Run from inside this folder:  ``streamlit run app.py``
"""

from __future__ import annotations

import logging
import traceback

import pandas as pd
import streamlit as st

from theo.config import ColumnSpec, SplitConfig, TrainConfig
from theo.data.loader import auto_detect_columns, load_table
from theo.logging_setup import setup_logging
from theo.pipeline import run_pipeline

logger = logging.getLogger("theo.app")

MODEL_HELP = {
    "lstm": "Stacked LSTM over the temporal window.",
    "ann": "MLP over the flattened window.",
    "gnn": "Learned message-passing GNN (superset of THEO); needs a neighbour column.",
}


def _idx(cols, name):
    return cols.index(name) if name in cols else 0


def render_inputs() -> None:
    st.set_page_config(page_title="THEO Forecasting", layout="wide")
    st.title("THEO — graph-informed time-series forecasting")
    st.caption("Upload any (entity, time) table, choose the **target** and a "
               "model, and forecast with THEO feature engineering. Every model "
               "uses the same node + neighbour-difference features.")

    # ---------------------------------------------------------------- data ---
    st.subheader("1 · Data")
    upl = st.file_uploader("Data (CSV)", type=["csv"], key="upl")
    path = st.text_input("…or a path to a CSV file", placeholder="optional")
    if upl is not None:
        df = load_table(upl)
    elif path:
        df = load_table(path)
    else:
        st.info("Upload a CSV or provide a path to continue.")
        return

    st.dataframe(df.head(8), width="stretch")

    # ------------------------------------- mapping + model + run (one form) --
    # Everything lives in a single form with one "Run" submit button, so the
    # click that starts the prediction is the same click that commits every
    # widget value. (Splitting this into an "Apply mapping" step + a separate
    # "Run" button meant the Run click re-ran the script with the form's submit
    # flag already False, so the early-return below swallowed it.)
    st.subheader("2 · Mapping, model & run")
    spec0 = auto_detect_columns(df)
    cols = list(df.columns)
    opt = ["(none)"] + cols

    with st.form("run"):
        st.markdown("**Column mapping**")
        c1, c2 = st.columns(2)
        entity = c1.selectbox("Entity / node id", cols, index=_idx(cols, spec0.entity))
        time_col = c2.selectbox("Time column", cols, index=_idx(cols, spec0.time))
        target = c1.selectbox("Target (what to predict)", cols,
                              index=_idx(cols, spec0.target))
        year = c2.selectbox("Year column (if time is a period like month)",
                            opt, index=opt.index(spec0.year) if spec0.year in opt else 0)

        st.markdown("**THEO graph — build it from a clean table** "
                    "(Delaunay triangulation over the positions below)")
        g1, g2 = st.columns(2)
        x_col = g1.selectbox("Position X (e.g. lon)",
                             opt, index=opt.index(spec0.x_col) if spec0.x_col in opt else 0)
        y_col = g2.selectbox("Position Y (e.g. lat)",
                             opt, index=opt.index(spec0.y_col) if spec0.y_col in opt else 0)
        theo_vars = st.multiselect(
            "THEO variables (unit's own values → `<v>_node` and `<v>_edge`)",
            cols, default=spec0.theo_variables or [])

        st.markdown("**…or consume a pre-built THEO table** "
                    "(``_node``/``_edge`` columns already present)")
        p1, p2 = st.columns(2)
        node_feats = p1.multiselect("Node features (`_node`)", cols,
                                    default=spec0.node_features or [])
        edge_feats = p2.multiselect("Edge features (`_edge`)", cols,
                                    default=spec0.edge_features or [])
        neighbor = c2.selectbox("Neighbour id (pre-built, for the GNN)",
                                opt, index=opt.index(spec0.neighbor) if spec0.neighbor in opt else 0)

        st.markdown("**Model**")
        r1, r2, r3, r4 = st.columns(4)
        window = r1.number_input("Window size (time steps)", 2, 480, 24, step=1)
        model = r2.selectbox("Model", ["lstm", "ann", "gnn"],
                             format_func=lambda m: m.upper())
        epochs = r3.number_input("Epochs", 1, 500, 10, step=1)
        seed = r4.number_input("Seed", 0, 999999, 42, step=1)
        st.caption(MODEL_HELP[model])

        with st.expander("Split & advanced"):
            a1, a2 = st.columns(2)
            train_frac = a1.slider("Train fraction", 0.5, 0.9, 0.70, 0.05)
            val_frac = a2.slider("Validation end fraction", train_frac + 0.05, 0.95, 0.85, 0.05)
            device = st.selectbox("Device", ["auto", "cpu", "mps", "cuda"], index=0)

        # Resolve the mode from what the user filled: build THEO from positions,
        # or consume an already-built ``_node``/``_edge`` table.
        build_mode = (x_col != "(none)" and y_col != "(none)" and bool(theo_vars))
        year_val = None if year == "(none)" else year
        if build_mode:
            spec = ColumnSpec(entity=entity, time=time_col, target=target, year=year_val,
                              theo_variables=list(theo_vars), x_col=x_col, y_col=y_col)
        else:
            spec = ColumnSpec(entity=entity, time=time_col, target=target,
                              node_features=list(node_feats), edge_features=list(edge_feats),
                              neighbor=None if neighbor == "(none)" else neighbor, year=year_val)
        has_features = bool(spec.theo_variables) or bool(spec.node_features)
        has_neighbor = spec.theo_mode or spec.neighbor is not None

        run_clicked = st.form_submit_button(
            "Run", type="primary",
            disabled=(not has_features) or (model == "gnn" and not has_neighbor),
        )

    if not run_clicked:
        st.info("Map the columns, choose the **target**, **window size** and **model** above, "
                "then press **Run** to start the prediction.")
        return
    if not has_features:
        st.warning("Select at least one feature: a THEO variable, or node/edge features.")
        return
    if model == "gnn" and not has_neighbor:
        st.error("The GNN needs a neighbour column (or THEO positions to build one).")
        return

    train_cfg = TrainConfig(window_size=int(window), epochs=int(epochs),
                            seed=int(seed), device=device)
    split_cfg = SplitConfig(train_frac=train_frac, val_frac=val_frac)

    logger.info("Run requested | model=%s | window=%d | epochs=%d | seed=%d "
                "| train_frac=%.2f | val_frac=%.2f | device=%s",
                model, window, epochs, seed, train_frac, val_frac, device)
    bar = st.progress(0.0, text="Starting …")

    def _progress(frac: float, label: str) -> None:
        frac = min(max(frac, 0.0), 1.0)
        bar.progress(frac, text=f"{label} — {int(round(frac * 100))}%")

    try:
        result = run_pipeline(df, spec, model, train_cfg, split_cfg,
                              progress=_progress)
    except Exception as exc:  # surface a readable error, log the full traceback
        logger.exception("Run failed")
        st.error(f"Run failed: {exc}")
        with st.expander("Full traceback", expanded=True):
            st.code(traceback.format_exc(), language="python")
        return
    _progress(1.0, "Done")
    logger.info("Run finished | test=%s | val=%s", result["test"], result["val"])
    render_result(result)


def render_result(result: dict) -> None:
    st.subheader("Results")
    t, v = result["test"], result["val"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test RMSE", f"{t['rmse']:.4f}", f"val {v['rmse']:.4f}")
    c2.metric("Test MAE", f"{t['mae']:.4f}", f"val {v['mae']:.4f}")
    c3.metric("Test R²", f"{t['r2']:.4f}", f"val {v['r2']:.4f}")
    c4.metric("Test pairs", f"{result['n_test_pairs']:,}")
    st.caption(f"{result['model'].upper()} · {result['n_entities']:,} entities · "
               f"{result['n_times']:,} time steps · window {result['window_size']} · "
               f"{result['n_features']} features · device {result['device']}")

    # Per-(entity, time) predictions table.
    rows = [(ent, tpos, yt, yp) for (ent, tpos), (yt, yp)
            in sorted(result["predictions"].items(), key=lambda kv: kv[0])]
    pred_df = pd.DataFrame(rows, columns=["entity", "time_key", "actual", "predicted"])
    st.dataframe(pred_df, width="stretch")
    st.download_button("Download predictions (CSV)",
                       pred_df.to_csv(index=False).encode("utf-8"),
                       file_name="theo_predictions.csv", mime="text/csv")

    # One sample series as a sanity plot.
    if not pred_df.empty:
        first_entity = pred_df["entity"].iloc[0]
        sample = pred_df[pred_df["entity"] == first_entity].sort_values("time_key")
        st.pyplot(_plot_series(sample))


def _plot_series(sample: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sample["time_key"], sample["actual"], label="actual", marker="o")
    ax.plot(sample["time_key"], sample["predicted"], label="predicted", marker="x")
    ax.set_title(f"Entity {sample['entity'].iloc[0]} (test split)")
    ax.set_xlabel("time step")
    ax.legend()
    fig.tight_layout()
    return fig


def _log_file():
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            return h.baseFilename
    return None


def main() -> None:
    setup_logging()
    render_inputs()
    st.divider()
    st.caption("THEO feature engineering: node features (the unit's own variables) "
               "+ edge features (neighbour differences), per-entity temporal windows, "
               "min-max fitted on the training split only.")
    log_path = _log_file()
    if log_path:
        st.caption(f"Run logs: `{log_path}`")


if __name__ == "__main__":
    main()
