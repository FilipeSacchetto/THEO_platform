"""Data loading and reshaping (domain-agnostic)."""

from .loader import (
    aggregate_entity_time,
    auto_detect_columns,
    load_table,
    make_time_key,
)

__all__ = [
    "aggregate_entity_time",
    "auto_detect_columns",
    "load_table",
    "make_time_key",
]
