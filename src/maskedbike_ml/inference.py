"""Frozen-preprocessing batch inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .artifacts import load_model, load_preprocessing
from .pipeline import extract_share_cycles


class Predictor:
    def __init__(self, artifact_dir: str | Path, device: str | None = None):
        root = Path(artifact_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.alignment, self.preprocessor, self.clip_scale = load_preprocessing(root / "preprocessing.npz")
        self.model, self.threshold, self.checkpoint = load_model(root / "model.pt", self.device)
        self.positive_event = str(self.checkpoint.get("positive_event", "hw_nonzero"))
        if self.positive_event not in {"hw_zero", "hw_nonzero"}:
            raise ValueError(f"unsupported model positive_event: {self.positive_event}")
        if tuple(self.checkpoint["product_shape"]) != self.preprocessor.geometry.product_shape:
            raise ValueError("model and preprocessing product shapes differ")

    def transform(self, traces: np.ndarray) -> np.ndarray:
        cycles = extract_share_cycles(traces, self.alignment.shifts(traces), self.preprocessor.geometry)
        values = self.preprocessor.transform(cycles)
        return np.clip(values / self.clip_scale, -1, 1).astype(np.float32)

    def probabilities(self, traces: np.ndarray, batch_size: int = 512) -> np.ndarray:
        values = self.transform(traces); rows = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                logits, _ = self.model(torch.from_numpy(values[start:start + batch_size]).to(self.device))
                rows.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        return np.concatenate(rows) if rows else np.empty(0, np.float32)

    def predict(self, traces: np.ndarray, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
        probability = self.probabilities(traces, batch_size)
        return probability, (probability >= self.threshold).astype(np.uint8)
