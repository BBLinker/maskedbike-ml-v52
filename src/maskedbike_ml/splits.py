"""Chronological complete-bundle splitting and leakage checks."""
from __future__ import annotations

import numpy as np


def trace_indices(bundle_ids: np.ndarray, selected) -> np.ndarray:
    return np.flatnonzero(np.isin(bundle_ids, list(selected)))


def chronological_split(bundle_ids: list[str], train_fraction: float) -> tuple[list[str], list[str]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    ordered = list(bundle_ids)
    cut = round(len(ordered) * train_fraction)
    if cut == 0 or cut == len(ordered):
        raise ValueError("both chronological partitions require at least one complete bundle")
    return ordered[:cut], ordered[cut:]


def assert_disjoint(left, right, name: str) -> None:
    overlap = set(map(str, left)) & set(map(str, right))
    if overlap:
        raise ValueError(f"{name} overlap: {sorted(overlap)[:5]}")
