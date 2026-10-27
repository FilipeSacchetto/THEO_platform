"""Logging for the THEO pipeline and the Streamlit interface.

Every run leaves a readable trail in a timestamped file under ``logs/`` (created
next to the folder the app is started from): data size, column mapping, split
cutoffs, window counts, per-epoch validation losses, final metrics — and the
complete traceback of any failure. The same lines are echoed to the console.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

_CONFIGURED = False


def setup_logging(log_dir: str | os.PathLike | None = None) -> logging.Logger:
    """Configure (once) the ``theo`` logger with console + file handlers.

    Returns the root ``theo`` logger; package modules get their own children
    via ``logging.getLogger(__name__)`` and inherit this configuration.
    """
    global _CONFIGURED
    logger = logging.getLogger("theo")
    if _CONFIGURED:
        return logger

    base = Path(log_dir) if log_dir else Path.cwd() / "logs"
    base.mkdir(parents=True, exist_ok=True)
    log_file = base / f"theo_{datetime.now():%Y%m%d_%H%M%S}.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_h = logging.FileHandler(log_file, encoding="utf-8")
    file_h.setFormatter(fmt)
    logger.addHandler(file_h)

    console_h = logging.StreamHandler()
    console_h.setFormatter(fmt)
    logger.addHandler(console_h)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    _CONFIGURED = True
    logger.info("Logging to %s", log_file)
    return logger
