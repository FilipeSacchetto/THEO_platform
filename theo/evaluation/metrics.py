"""Error metrics and per-(entity, time) prediction collection.

Predictions are keyed by ``(entity, time_key_of_target_step)`` so that a result
is always interpretable as "this unit, this month" and can be turned into a
plain table in the interface.
"""

from __future__ import annotations

import numpy as np
import torch


def metrics(y_true, y_pred) -> dict:
    """RMSE, MAE and R^2 on the original (physical) scale."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) or 1.0
    return {"rmse": rmse, "mae": mae, "r2": 1.0 - ss_res / ss_tot}


def predictions_to_metrics(preds: dict) -> dict:
    """Metrics over a ``(entity, tpos) -> (y_true, y_pred)`` mapping."""
    yt = np.array([v[0] for v in preds.values()], np.float64)
    yp = np.array([v[1] for v in preds.values()], np.float64)
    return metrics(yt, yp)


def collect_predictor(model, ds, device) -> dict:
    """Test-split predictions of an ANN/LSTM, keyed by (entity, time key)."""
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(ds.X_test).to(device)).cpu().numpy()
    yp = ds.inverse_target(pred)
    yt = ds.inverse_target(ds.y_test)
    return {ds.test_ids[i]: (float(yt[i]), float(yp[i])) for i in range(len(yp))}


def collect_gnn(model, test_samples, tgt_min, tgt_range, device, labels) -> dict:
    """Test-split predictions of the GNN, keyed by (entity label, time key)."""
    model.eval()
    out = {}
    with torch.no_grad():
        for s in test_samples:
            f = [x.to(device) for x in s["feats"]]
            e = [x.to(device) for x in s["edges"]]
            pred = model(f, e).cpu().numpy()
            p = s["present"].cpu().numpy()
            yp = pred * tgt_range + tgt_min
            yt = s["y"].cpu().numpy()
            for nid in range(len(yp)):
                if p[nid]:
                    out[(labels[nid], s["tpos"])] = (float(yt[nid]), float(yp[nid]))
    return out
