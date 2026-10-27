"""Training routines shared by every model (kept identical for fair runs)."""

from .engine import choose_device, set_seed, train_gnn, train_predictor

__all__ = ["choose_device", "set_seed", "train_gnn", "train_predictor"]
