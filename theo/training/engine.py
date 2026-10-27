"""Model training with best-epoch selection by validation RMSE.

Both the generic predictors (ANN / LSTM) and the GNN are trained the same way:
the best epoch is chosen by validation error and restored, and the test split is
touched only at the end. This mirrors the thesis reference implementation in
``experiments/run_ablation.py``.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import TrainConfig
from ..evaluation.metrics import metrics
from ..preprocessing.windowing import WindowedDataset

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(pref: str = "auto") -> str:
    """Pick a compute device, preferring the user's choice, then CUDA > MPS > CPU."""
    if pref and pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except (AttributeError, RuntimeError):
        pass
    return "cpu"


def _eval(model, X, y, ds: WindowedDataset, device) -> dict:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(device)).cpu().numpy()
    return metrics(ds.inverse_target(y), ds.inverse_target(pred))


def train_predictor(model, ds: WindowedDataset, cfg: TrainConfig, device,
                    epoch_cb=None):
    """Train an ANN/LSTM, restoring the weights of the best validation epoch.

    ``epoch_cb(epoch_done, total_epochs)`` is called after every epoch (used for
    UI progress reporting); it may be ``None``.
    """
    set_seed(cfg.seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(ds.X_train), torch.from_numpy(ds.y_train)),
        batch_size=cfg.batch_size, shuffle=True,
    )

    best_val = float("inf")
    best = copy.deepcopy(model.state_dict())
    for ep in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        vm = _eval(model, ds.X_val, ds.y_val, ds, device)
        improved = vm["rmse"] < best_val
        if improved:
            best_val = vm["rmse"]
            best = copy.deepcopy(model.state_dict())
        logger.info(
            "Epoch %d/%d | val_rmse=%.5f | val_r2=%.4f%s",
            ep + 1, cfg.epochs, vm["rmse"], vm["r2"], " | best" if improved else "",
        )
        if epoch_cb:
            epoch_cb(ep + 1, cfg.epochs)

    logger.info("Best epoch selected | val_rmse=%.5f", best_val)
    model.load_state_dict(best)
    result = {
        "val": _eval(model, ds.X_val, ds.y_val, ds, device),
        "test": _eval(model, ds.X_test, ds.y_test, ds, device),
    }
    return result, model


def train_gnn(model, samples, cfg: TrainConfig, device, epoch_cb=None):
    """Train the GNN on graph snapshots, restoring the best validation epoch.

    Only units present at the target step contribute to the loss/evaluation, so
    the prediction is always over a unit's own target series.
    ``epoch_cb`` works as in :func:`train_predictor`.
    """
    train, val, test, tgt_min, tgt_range = samples[:5]
    set_seed(cfg.seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    def to_dev(s):
        return ([f.to(device) for f in s["feats"]],
                [e.to(device) for e in s["edges"]],
                s["y"].to(device),
                s["present"].to(device))

    def ev(samples_):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for s in samples_:
                f, e, y, p = to_dev(s)
                pred = model(f, e)
                preds.append((pred[p] * tgt_range + tgt_min).cpu().numpy())
                trues.append(y[p].cpu().numpy())
        if not preds:
            return None
        return metrics(np.concatenate(trues), np.concatenate(preds))

    best_val = float("inf")
    best = copy.deepcopy(model.state_dict())
    for ep in range(cfg.gnn_epochs):
        model.train()
        for s in train:
            f, e, y, p = to_dev(s)
            opt.zero_grad()
            loss = loss_fn(model(f, e)[p], (y[p] - tgt_min) / tgt_range)
            loss.backward()
            opt.step()
        vm = ev(val)
        improved = bool(vm) and vm["rmse"] < best_val
        if improved:
            best_val = vm["rmse"]
            best = copy.deepcopy(model.state_dict())
        logger.info(
            "GNN epoch %d/%d | val_rmse=%s%s",
            ep + 1, cfg.gnn_epochs,
            f"{vm['rmse']:.5f}" if vm else "n/a",
            " | best" if improved else "",
        )
        if epoch_cb:
            epoch_cb(ep + 1, cfg.gnn_epochs)

    logger.info("GNN best epoch selected | val_rmse=%.5f", best_val)
    model.load_state_dict(best)
    return {"val": ev(val), "test": ev(test)}, model
