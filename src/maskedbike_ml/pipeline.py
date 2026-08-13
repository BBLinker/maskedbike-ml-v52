"""Geometry-aware q-share centered-product preprocessing and CSCAE model."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch import nn

from .models import CSCAE, PaperClassifier


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class TraceGeometry:
    """Capture geometry derived from one ready-manifest contract."""

    masking_order: int
    samples: int
    share_intervals: tuple[tuple[int, int], ...]
    j_count: int = 14
    j_stride_samples: int = 0
    cycle_samples: int = 0

    @property
    def share_count(self) -> int:
        return self.masking_order + 1

    @property
    def product_shape(self) -> tuple[int, int]:
        return self.j_count, self.cycle_samples

    @property
    def product_length(self) -> int:
        return self.j_count * self.cycle_samples

    @property
    def share_starts(self) -> tuple[int, ...]:
        return tuple(start for start, _ in self.share_intervals)

    @classmethod
    def from_manifest(cls, manifest: dict, j_count: int = 14) -> "TraceGeometry":
        config = manifest["config"]
        provenance = manifest["provenance"]
        order = int(config.get("masking_order", provenance.get("masking_order")))
        samples = int(config["samples"])
        rows = sorted(provenance.get("share_intervals", []), key=lambda row: int(row.get("share", 0)))
        intervals = tuple(tuple(map(int, row["stored"])) for row in rows)
        if len(intervals) != order + 1:
            raise ValueError(f"masking order {order} requires {order + 1} shares, manifest has {len(intervals)}")
        widths = [end - start for start, end in intervals]
        if not widths or any(width <= 0 for width in widths) or len(set(widths)) != 1:
            raise ValueError(f"share intervals must have one equal positive width: {intervals}")
        if any(start < 0 or end > samples for start, end in intervals):
            raise ValueError(f"share interval outside trace with {samples} samples")
        matching = provenance.get("matching_j_positions", [])
        if len(matching) != j_count:
            raise ValueError(f"expected {j_count} matching_j_positions, got {len(matching)}")
        for expected_j, row in enumerate(matching):
            stored = row.get("stored", [])
            if int(row.get("j", expected_j)) != expected_j or len(stored) != order + 1:
                raise ValueError("matching_j_positions do not match the declared share count")
        anchors = np.asarray([row["stored"] for row in matching], dtype=np.int64)
        periods = np.diff(anchors, axis=0)
        if periods.size == 0 or not np.all(periods == periods[0, 0]):
            raise ValueError("matching_j_positions must have one constant period across j and shares")
        period = int(periods[0, 0])
        if period <= 0:
            raise ValueError("matching_j_positions must declare a positive period")
        cycle_samples = min(period, min(widths) - (j_count - 1) * period)
        if cycle_samples <= 0:
            raise ValueError("share intervals do not contain a uniform positive window for every j")
        return cls(order, samples, intervals, j_count, period, cycle_samples)

    def as_dict(self) -> dict[str, Any]:
        return {
            "masking_order": self.masking_order,
            "share_count": self.share_count,
            "samples": self.samples,
            "share_intervals": self.share_intervals,
            "j_count": self.j_count,
            "j_stride_samples": self.j_stride_samples,
            "cycle_samples": self.cycle_samples,
            "cycles_shape": ["N", self.share_count, self.j_count, self.cycle_samples],
            "product_shape": ["N", self.j_count, self.cycle_samples],
        }


@dataclass
class TraceDataset:
    traces: np.ndarray
    labels: np.ndarray
    hamming_weights: np.ndarray
    bundle_ids: np.ndarray
    case_ids: np.ndarray
    trace_ids: np.ndarray
    bundle_order: list[str]
    bundle_rows: list[dict[str, Any]]
    geometry: TraceGeometry


@dataclass
class CommonAlignment:
    template: np.ndarray
    candidate_shifts: np.ndarray
    sample_indices: np.ndarray

    @classmethod
    def fit(
        cls,
        traces: np.ndarray,
        train_indices: np.ndarray,
        share_intervals: tuple[tuple[int, int], ...],
        max_shift: int = 4,
        stride: int = 2,
    ) -> "CommonAlignment":
        template = traces[train_indices].astype(np.float64).mean(axis=0)
        indices = np.concatenate([
            np.arange(start + max_shift, end - max_shift, stride, dtype=np.int64)
            for start, end in share_intervals
        ])
        return cls(template, np.arange(-max_shift, max_shift + 1, dtype=np.int16), indices)

    def shifts(self, traces: np.ndarray) -> np.ndarray:
        reference = self.template[self.sample_indices].astype(np.float32)
        reference -= reference.mean(); reference /= np.linalg.norm(reference) + 1e-12
        scores = np.empty((len(traces), len(self.candidate_shifts)), np.float32)
        for column, shift in enumerate(self.candidate_shifts):
            values = traces[:, self.sample_indices + int(shift)].astype(np.float32)
            values -= values.mean(axis=1, keepdims=True)
            scores[:, column] = values @ reference
        return self.candidate_shifts[np.argmax(scores, axis=1)]


def extract_share_cycles(traces: np.ndarray, global_shifts: np.ndarray, geometry: TraceGeometry) -> np.ndarray:
    """Return exact local-time cycles as [N,q,j,time], using one shift per trace."""
    traces = np.asarray(traces)
    shifts = np.asarray(global_shifts, dtype=np.int64).reshape(-1)
    if traces.ndim != 2 or traces.shape[1] != geometry.samples:
        raise ValueError(f"traces must have shape [N,{geometry.samples}]")
    if len(traces) != len(shifts):
        raise ValueError("one common global shift is required per trace")
    result = []
    local = (
        np.arange(geometry.j_count, dtype=np.int64)[:, None] * geometry.j_stride_samples
        + np.arange(geometry.cycle_samples, dtype=np.int64)[None, :]
    )
    rows = np.arange(len(traces))[:, None, None]
    for start in geometry.share_starts:
        positions = start + local[None, :, :] + shifts[:, None, None]
        result.append(traces[rows, positions].astype(np.float32))
    return np.stack(result, axis=1)


@dataclass
class CenteredProductPreprocessor:
    """Training-only q-variate pointwise centered-product standardization."""

    geometry: TraceGeometry
    share_center: np.ndarray | None = None
    product_center: np.ndarray | None = None
    product_scale: np.ndarray | None = None
    epsilon: float = 1e-6

    def _validate(self, cycles: np.ndarray) -> np.ndarray:
        values = np.asarray(cycles, dtype=np.float32)
        expected = (self.geometry.share_count, self.geometry.j_count, self.geometry.cycle_samples)
        if values.ndim != 4 or values.shape[1:] != expected:
            raise ValueError(f"cycles must have shape [N,{expected[0]},{expected[1]},{expected[2]}]")
        return values

    def fit(self, training_cycles: np.ndarray) -> "CenteredProductPreprocessor":
        values = self._validate(training_cycles)
        self.share_center = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        raw = np.prod(values.astype(np.float64) - self.share_center.astype(np.float64), axis=1).astype(np.float32)
        self.product_center = raw.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.product_scale = np.maximum(raw.std(axis=0, dtype=np.float64), self.epsilon).astype(np.float32)
        return self

    def raw_product(self, cycles: np.ndarray) -> np.ndarray:
        values = self._validate(cycles)
        if self.share_center is None:
            raise RuntimeError("fit training statistics first")
        return np.prod(values.astype(np.float64) - self.share_center.astype(np.float64), axis=1).astype(np.float32)

    def transform(self, cycles: np.ndarray) -> np.ndarray:
        if self.product_center is None or self.product_scale is None:
            raise RuntimeError("fit training statistics first")
        return (self.raw_product(cycles) - self.product_center) / self.product_scale


class CenteredProductCSCAENN(nn.Module):
    """One CSCAE consumes the complete ordered [j,time] q-share product."""

    def __init__(self, product_shape: tuple[int, int], latent_ratio: float, channels: int, widths: list[int]):
        super().__init__()
        self.product_shape = tuple(map(int, product_shape))
        self.autoencoder = CSCAE(int(np.prod(self.product_shape)), latent_ratio, channels)
        self.classifier = PaperClassifier(self.autoencoder.latent_size, widths)

    def reconstruct(self, product: torch.Tensor) -> torch.Tensor:
        _, rebuilt = self.autoencoder(product.flatten(1))
        return rebuilt.reshape_as(product)

    def forward(self, product: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        code, rebuilt = self.autoencoder(product.flatten(1))
        return self.classifier(code), rebuilt.reshape_as(product)

    def autoencoder_parameters(self):
        return self.autoencoder.parameters()


def threshold_from_validation(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.quantile(probabilities, np.linspace(.01, .99, 199)))
    scores = [balanced_accuracy_score(labels, probabilities >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (probabilities >= threshold).astype(np.uint8)
    return {
        "traces": len(labels),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, prediction)),
        "confusion_matrix": confusion_matrix(labels, prediction, labels=[0, 1]),
        "threshold": threshold,
    }
