#!/usr/bin/env python3
"""Train the q-share centered-product CSCAE model on an immutable snapshot."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from maskedbike_ml.dataset import load_r0_r7, load_r2
from maskedbike_ml.training import clip_with_scale, fit_model
from maskedbike_ml.artifacts import save_preprocessing, write_artifact_manifest, verify_artifact_manifest
from maskedbike_ml.pipeline import CommonAlignment, CenteredProductPreprocessor, extract_share_cycles, write_json
from maskedbike_ml.splits import assert_disjoint, chronological_split, trace_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path, help="dataset root containing *.h5.ready.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    dataset = args.dataset.resolve(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    contract = config["data_contract"]
    masking_order = int(contract["masking_order"])
    ready_paths = sorted(dataset.rglob("*.h5.ready.json"))

    r07 = load_r0_r7(dataset, ready_paths, masking_order, contract["adc_source"], contract["j_count"], output / "bundle-audit.json")
    r2 = load_r2(dataset, ready_paths, masking_order, contract["adc_source"], contract["j_count"])
    if r07.geometry != r2.geometry:
        raise ValueError("R0/R7 and R2 capture geometries differ")
    geometry = r07.geometry

    by_class = {0: [], 1: []}
    for bundle in r07.bundle_order:
        row = trace_indices(r07.bundle_ids, {bundle})
        by_class[int(r07.labels[row[0]])].append(bundle)
    allocation = config["profiling"]
    r0 = sorted(by_class[0])[: int(allocation["r0_bundles"])]
    r7 = sorted(by_class[1])[: int(allocation["r7_bundles"])]
    ordered_r2_rows = sorted(r2.bundle_rows, key=lambda row: (str(row.get("capture_timestamp") or ""), row["bundle_id"]))
    r2_rows = ordered_r2_rows[: int(allocation["r2_bundles"])]
    heldout_rows = ordered_r2_rows[int(allocation["r2_bundles"]):]
    if len(r0) != allocation["r0_bundles"] or len(r7) != allocation["r7_bundles"] or len(r2_rows) != allocation["r2_bundles"] or not heldout_rows:
        raise RuntimeError("insufficient complete bundles for profiling plus independent R2 test")

    fraction = float(config["split"]["train_fraction"])
    r0_train, r0_validation = chronological_split(r0, fraction)
    r7_train, r7_validation = chronological_split(r7, fraction)
    r2_profiling_ids = [row["bundle_id"] for row in r2_rows]
    r2_train_ids, r2_validation_ids = chronological_split(r2_profiling_ids, fraction)
    train_bundles = set(r0_train + r7_train)
    validation_bundles = set(r0_validation + r7_validation)
    r2_train_bundles = set(r2_train_ids); r2_validation_bundles = set(r2_validation_ids)
    heldout_bundles = {row["bundle_id"] for row in heldout_rows}
    train = trace_indices(r07.bundle_ids, train_bundles)
    validation = trace_indices(r07.bundle_ids, validation_bundles)
    r2_train = trace_indices(r2.bundle_ids, r2_train_bundles)
    r2_validation = trace_indices(r2.bundle_ids, r2_validation_bundles)
    r2_test = trace_indices(r2.bundle_ids, heldout_bundles)

    assert_disjoint(train_bundles, validation_bundles, "R0/R7 bundle")
    assert_disjoint(r2_train_bundles, r2_validation_bundles | heldout_bundles, "R2 bundle")
    profiling_r2 = np.r_[r2_train, r2_validation]
    assert set(r2.trace_ids[profiling_r2]).isdisjoint(set(r2.trace_ids[r2_test]))
    assert set(r2.case_ids[profiling_r2]).isdisjoint(set(r2.case_ids[r2_test]))

    snapshot = {
        "schema": "maskedbike-qshare-snapshot.v2", "created_before_training_unix": time.time(),
        "ready_files_frozen": len(ready_paths), "dataset": str(dataset), "geometry": geometry.as_dict(),
        "profiling": {
            "R0": sum(row["trace_count"] for row in r07.bundle_rows if row["bundle_id"] in set(r0)),
            "R2_unlabeled": sum(row["trace_count"] for row in r2_rows),
            "R7": sum(row["trace_count"] for row in r07.bundle_rows if row["bundle_id"] in set(r7)),
            "train_traces": len(train) + len(r2_train), "validation_traces": len(validation) + len(r2_validation),
        },
        "r2_heldout": {
            "bundles": len(heldout_rows), "traces": len(r2_test),
            "hw0": int((r2.hamming_weights[r2_test] == 0).sum()),
            "hw_nonzero": int((r2.hamming_weights[r2_test] != 0).sum()), "bundle_rows": heldout_rows,
        },
        "selected_bundles": {"R0": r0, "R2_profiling": r2_profiling_ids, "R7": r7},
        "bundle_rows": [row for row in r07.bundle_rows if row["bundle_id"] in set(r0 + r7)] + r2_rows + heldout_rows,
        "classification_labels": {"R0": 0, "R7": 1, "R2_profiling": "unused"},
        "no_r2_test_overlap": {"bundle_ids": True, "case_ids": True, "trace_ids": True},
    }
    write_json(output / "dataset-snapshot.json", snapshot)

    common_training = np.concatenate([r07.traces[train], r2.traces[r2_train]])
    alignment = CommonAlignment.fit(
        common_training, np.arange(len(common_training)), geometry.share_intervals,
        max_shift=int(config["preprocessing"]["alignment_max_shift"]),
        stride=int(config["preprocessing"]["alignment_stride"]),
    )
    cycles07 = extract_share_cycles(r07.traces, alignment.shifts(r07.traces), geometry)
    cycles2 = extract_share_cycles(r2.traces, alignment.shifts(r2.traces), geometry)
    preprocessor = CenteredProductPreprocessor(geometry).fit(np.concatenate([cycles07[train], cycles2[r2_train]]))
    z07 = preprocessor.transform(cycles07); z2 = preprocessor.transform(cycles2)
    clip_percentile = float(config["preprocessing"]["clip_percentile"])
    clip_scale = float(np.percentile(np.abs(np.concatenate([z07[train], z2[r2_train]])), clip_percentile)) or 1.0
    product07 = clip_with_scale(z07, clip_scale); product2 = clip_with_scale(z2, clip_scale)

    result = fit_model(
        output, f"masking-order-{masking_order}",
        np.concatenate([product07[train], product2[r2_train]]),
        np.concatenate([product07[validation], product2[r2_validation]]),
        product07[train], r07.labels[train], product07[validation], r07.labels[validation],
        product2[r2_test], (r2.hamming_weights[r2_test] == 0).astype(np.uint8), config,
    )
    write_json(output / "results.json", {"config": config, "result": result})
    save_preprocessing(output / "preprocessing.npz", alignment, preprocessor, clip_scale)
    manifest_path = write_artifact_manifest(output)
    verify_artifact_manifest(manifest_path)
    print(json.dumps({
        "status": "complete", "output": str(output), "geometry": geometry.as_dict(),
        "r0r7_validation_auc": result["r0_r7_validation"]["roc_auc"],
        "r2_test_auc": result["r2_heldout_test"]["roc_auc"],
        "r2_test_balanced_accuracy": result["r2_heldout_test"]["balanced_accuracy"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
