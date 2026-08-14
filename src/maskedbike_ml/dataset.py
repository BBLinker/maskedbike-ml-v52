"""Manifest-audited HDF5 dataset loading for q-share captures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .pipeline import TraceDataset, TraceGeometry, write_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _distribution(values: np.ndarray) -> dict[str, int]:
    keys, counts = np.unique(values, return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(keys, counts)}


def load_rounds(
    dataset: Path,
    ready_paths: list[Path],
    rounds: set[int],
    masking_order: int,
    adc_source: str,
    j_count: int,
    audit_path: Path | None = None,
) -> TraceDataset:
    """Load only bundles satisfying the complete manifest contract."""
    xs, ys, hws, bundles, cases, tids, rows, rejected = [], [], [], [], [], [], [], []
    common_geometry = None
    for ready_path in ready_paths:
        try:
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            manifest = ready["run_manifest"]
            config = manifest["config"]
            provenance = manifest["provenance"]
            round_id = int(provenance["decoder_round"])
            reasons = []
            if round_id not in rounds:
                reasons.append(f"decoder_round not in {sorted(rounds)}")
            if int(config.get("masking_order", -1)) != masking_order:
                reasons.append(f"masking_order != {masking_order}")
            if config.get("adc_source") != adc_source:
                reasons.append(f"adc_source != {adc_source}")
            try:
                geometry = TraceGeometry.from_manifest(manifest, j_count=j_count)
            except (KeyError, TypeError, ValueError) as error:
                geometry = None
                reasons.append(f"invalid capture geometry: {error}")
            if reasons:
                rejected.append({"ready_path": str(ready_path.relative_to(dataset)), "reasons": reasons})
                continue
            assert geometry is not None
            if common_geometry is None:
                common_geometry = geometry
            elif geometry != common_geometry:
                raise ValueError(f"mixed capture geometries: {common_geometry} versus {geometry}")
            h5_path = ready_path.parent / ready.get("file", ready_path.name.removesuffix(".ready.json"))
            actual = sha256_file(h5_path)
            declared = ready.get("sha256") or ready.get("integrity", {}).get("bundle_sha256")
            if declared and actual.lower() != str(declared).lower():
                raise ValueError(f"SHA-256 mismatch: {h5_path}")
            with h5py.File(h5_path, "r") as handle:
                x = np.asarray(handle["traces"], dtype=np.int16)
                hw = np.asarray(handle["hamming_weights"], dtype=np.int64)
                case = np.asarray(handle["case_ids"]).astype("U")
                trace_id = np.asarray(handle["trace_ids"]).astype("U")
            if (x.ndim != 2 or x.shape[1] != geometry.samples or len(x) == 0
                    or not (len(x) == len(hw) == len(case) == len(trace_id))):
                raise ValueError(f"invalid H5 shape or empty bundle: {h5_path}")
            bundle_id = h5_path.name
            label = 0 if round_id == 0 else 1 if round_id == 7 else 255
            xs.append(x); ys.append(np.full(len(x), label, np.uint8)); hws.append(hw)
            cases.append(case); tids.append(trace_id); bundles.append(np.full(len(x), bundle_id, dtype="U256"))
            rows.append({
                "run_id": _text(ready.get("run_id", manifest.get("run_id", ""))),
                "bundle_id": bundle_id,
                "decoder_round": round_id,
                "trace_count": int(len(x)),
                "capture_timestamp": manifest.get("created_at") or ready.get("created_at"),
                "sha256": actual,
                "ready_path": str(ready_path.relative_to(dataset)),
                "h5_path": str(h5_path.relative_to(dataset)),
                "hw_distribution": _distribution(hw),
                "global_label_distribution": {"0": int((hw == 0).sum()), "1": int((hw != 0).sum())},
                "bit_label_distribution": [
                    {"0": int((((hw >> j) & 1) == 0).sum()), "1": int((((hw >> j) & 1) == 1).sum())}
                    for j in range(j_count)
                ],
                "case_ids": case.tolist(),
                "trace_ids": trace_id.tolist(),
            })
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            rejected.append({"ready_path": str(ready_path.relative_to(dataset)), "reasons": [str(error)]})
    if not rows or common_geometry is None:
        raise ValueError(f"no accepted bundles for rounds {sorted(rounds)} and masking order {masking_order}")
    result = TraceDataset(
        np.concatenate(xs), np.concatenate(ys), np.concatenate(hws), np.concatenate(bundles),
        np.concatenate(cases), np.concatenate(tids), [row["bundle_id"] for row in rows], rows, common_geometry,
    )
    if audit_path is not None:
        write_json(audit_path, {
            "schema": "maskedbike-bundle-audit.v3",
            "ready_files_scanned": len(ready_paths),
            "selection": {"rounds": sorted(rounds), "masking_order": masking_order,
                          "share_count": masking_order + 1, "adc_source": adc_source, "j_count": j_count},
            "geometry": common_geometry.as_dict(),
            "accepted_bundles": len(rows), "rejected_bundles": len(rejected),
            "accepted": [{k: v for k, v in row.items() if k not in {"case_ids", "trace_ids"}} for row in rows],
            "rejected": rejected,
        })
    return result


def load_r0_r7(dataset: Path, ready_paths: list[Path], masking_order: int, adc_source: str,
               j_count: int, audit_path: Path | None = None) -> TraceDataset:
    return load_rounds(dataset, ready_paths, {0, 7}, masking_order, adc_source, j_count, audit_path)


def load_r2(dataset: Path, ready_paths: list[Path], masking_order: int, adc_source: str,
            j_count: int, audit_path: Path | None = None) -> TraceDataset:
    return load_rounds(dataset, ready_paths, {2}, masking_order, adc_source, j_count, audit_path)
