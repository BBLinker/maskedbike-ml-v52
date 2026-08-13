"""Save, hash, verify, and restore a complete trained pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .pipeline import CenteredProductCSCAENN, CenteredProductPreprocessor, CommonAlignment, TraceGeometry


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_preprocessing(path: Path, alignment: CommonAlignment, preprocessor: CenteredProductPreprocessor,
                       clip_scale: float) -> None:
    geometry = preprocessor.geometry
    np.savez_compressed(
        path, alignment_template=alignment.template, candidate_shifts=alignment.candidate_shifts,
        sample_indices=alignment.sample_indices, share_center=preprocessor.share_center,
        product_center=preprocessor.product_center, product_scale=preprocessor.product_scale,
        product_clip_scale=float(clip_scale), masking_order=geometry.masking_order,
        samples=geometry.samples, share_intervals=np.asarray(geometry.share_intervals),
        j_count=geometry.j_count, j_stride_samples=geometry.j_stride_samples,
        cycle_samples=geometry.cycle_samples,
    )


def load_preprocessing(path: Path) -> tuple[CommonAlignment, CenteredProductPreprocessor, float]:
    with np.load(path, allow_pickle=False) as data:
        geometry = TraceGeometry(int(data["masking_order"]), int(data["samples"]),
                                 tuple(map(tuple, data["share_intervals"].astype(int).tolist())),
                                 int(data["j_count"]), int(data["j_stride_samples"]), int(data["cycle_samples"]))
        alignment = CommonAlignment(data["alignment_template"], data["candidate_shifts"], data["sample_indices"])
        preprocessor = CenteredProductPreprocessor(geometry, data["share_center"], data["product_center"], data["product_scale"])
        return alignment, preprocessor, float(data["product_clip_scale"])


def load_model(path: Path, device: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = CenteredProductCSCAENN(tuple(checkpoint["product_shape"]), config["model"]["latent_ratio"],
                                   config["model"]["channels"], config["model"]["classifier_widths"])
    model.load_state_dict(checkpoint["model_state"]); model.to(device); model.eval()
    return model, float(checkpoint["threshold"]), checkpoint


def write_artifact_manifest(output: Path) -> Path:
    names = ["dataset-snapshot.json", "preprocessing.npz", "model.pt", "predictions.npz", "results.json"]
    rows = [{"path": name, "sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
            for name in names if (output / name).is_file()]
    path = output / "artifact-manifest.json"
    path.write_text(json.dumps({"schema": "maskedbike-artifacts.v1", "files": rows}, indent=2) + "\n")
    return path


def verify_artifact_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text())
    for row in manifest["files"]:
        target = path.parent / row["path"]
        if target.stat().st_size != row["bytes"] or sha256_file(target) != row["sha256"]:
            raise ValueError(f"artifact verification failed: {target}")
