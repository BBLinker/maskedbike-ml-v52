"""Streaming Guo-Johansson-Stankovski distance-spectrum estimation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def cyclic_distance(left: int, right: int, block_length: int) -> int:
    delta = abs(int(left) - int(right)) % block_length
    return min(delta, block_length - delta)


def distance_spectrum(positions: np.ndarray, block_length: int) -> np.ndarray:
    """Unique nonzero cyclic distances of one sparse error block."""
    values = np.unique(np.asarray(positions, dtype=np.int64))
    values = values[(values >= 0) & (values < block_length)]
    if len(values) < 2:
        return np.empty(0, np.int32)
    left, right = np.triu_indices(len(values), 1)
    delta = np.abs(values[left] - values[right])
    return np.unique(np.minimum(delta, block_length - delta)).astype(np.int32)


@dataclass
class GJSAggregator:
    """Accumulate conditional oracle statistics for every cyclic distance."""

    block_length: int
    queries: np.ndarray
    soft_sum: np.ndarray
    soft_square_sum: np.ndarray
    hard_sum: np.ndarray
    traces_seen: int = 0

    @classmethod
    def create(cls, block_length: int) -> "GJSAggregator":
        if block_length < 3:
            raise ValueError("block_length must be at least three")
        size = block_length // 2 + 1
        zeros = np.zeros(size, np.float64)
        return cls(block_length, np.zeros(size, np.int64), zeros.copy(), zeros.copy(), zeros.copy())

    def update(self, error_positions: np.ndarray, soft_oracle: np.ndarray, threshold: float) -> None:
        """Update from padded `[N,weight]` positions; negative values are padding."""
        positions = np.asarray(error_positions, dtype=np.int64)
        scores = np.asarray(soft_oracle, dtype=np.float64).reshape(-1)
        if positions.ndim != 2 or len(positions) != len(scores):
            raise ValueError("error_positions must be [N,weight] and match the oracle rows")
        if np.any((positions >= self.block_length)):
            raise ValueError("error position is outside the cyclic block")
        present = np.zeros((len(positions), len(self.queries)), dtype=np.bool_)
        for row, values in enumerate(positions):
            present[row, distance_spectrum(values, self.block_length)] = True
        self.queries += present.sum(axis=0, dtype=np.int64)
        self.soft_sum += present.T @ scores
        self.soft_square_sum += present.T @ np.square(scores)
        self.hard_sum += present.T @ (scores >= threshold).astype(np.float64)
        self.traces_seen += len(scores)

    def report(self, min_queries: int = 1000, expected_direction: str = "low") -> dict[str, Any]:
        if expected_direction not in {"low", "high"}:
            raise ValueError("expected_direction must be low or high")
        count = self.queries.astype(np.float64)
        soft = np.divide(self.soft_sum, count, out=np.full_like(count, np.nan), where=count > 0)
        hard = np.divide(self.hard_sum, count, out=np.full_like(count, np.nan), where=count > 0)
        second = np.divide(self.soft_square_sum, count, out=np.full_like(count, np.nan), where=count > 0)
        variance = np.maximum(second - np.square(soft), 0.0)
        se = np.sqrt(np.divide(variance, count, out=np.full_like(count, np.nan), where=count > 1))
        eligible = np.flatnonzero((self.queries >= min_queries) & (np.arange(len(count)) > 0))
        order = np.argsort(soft[eligible])
        if expected_direction == "high":
            order = order[::-1]
        ranked = eligible[order]
        rows = [{
            "distance": int(distance), "queries": int(self.queries[distance]),
            "soft_rate": float(soft[distance]), "hard_rate": float(hard[distance]),
            "soft_standard_error": float(se[distance]),
            "soft_ci95_low": float(soft[distance] - 1.96 * se[distance]),
            "soft_ci95_high": float(soft[distance] + 1.96 * se[distance]),
        } for distance in ranked]
        return {"block_length": self.block_length, "traces_seen": self.traces_seen,
                "min_queries": min_queries, "expected_direction": expected_direction,
                "eligible_distances": len(ranked), "ranked_distances": rows}

    def state(self) -> dict[str, np.ndarray]:
        return {"block_length": np.asarray(self.block_length), "traces_seen": np.asarray(self.traces_seen),
                "queries": self.queries, "soft_sum": self.soft_sum,
                "soft_square_sum": self.soft_square_sum, "hard_sum": self.hard_sum}

    @classmethod
    def from_state(cls, state) -> "GJSAggregator":
        return cls(int(state["block_length"]), state["queries"].copy(), state["soft_sum"].copy(),
                   state["soft_square_sum"].copy(), state["hard_sum"].copy(), int(state["traces_seen"]))


def reconstruct_support(distance_set: set[int], block_length: int, weight: int,
                        max_solutions: int = 32, max_nodes: int = 2_000_000) -> list[list[int]]:
    """Recover supports up to cyclic rotation/reflection from a candidate distance set."""
    spectrum = {int(value) for value in distance_set if 0 < int(value) <= block_length // 2}
    if weight < 2 or not spectrum:
        return []
    candidates = [value for value in range(1, block_length)
                  if cyclic_distance(0, value, block_length) in spectrum]
    solutions: list[list[int]] = []
    nodes = 0

    def visit(support: list[int], start: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes or len(solutions) >= max_solutions:
            return
        if len(support) == weight:
            found = set(distance_spectrum(np.asarray(support), block_length).tolist())
            if found == spectrum:
                canonical = min(tuple(sorted((value - shift) % block_length for value in support))
                                for shift in support)
                reflected = tuple(sorted((-value) % block_length for value in canonical))
                representative = list(min(canonical, reflected))
                if representative not in solutions:
                    solutions.append(representative)
            return
        for index in range(start, len(candidates)):
            value = candidates[index]
            if all(cyclic_distance(value, old, block_length) in spectrum for old in support):
                visit(support + [value], index + 1)

    visit([0], 0)
    return solutions
