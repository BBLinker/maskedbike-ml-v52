#!/usr/bin/env python3
"""Run fixed-test profiling-count sensitivity experiments for any masking order."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from maskedbike_ml.artifacts import save_preprocessing, write_artifact_manifest, verify_artifact_manifest
from maskedbike_ml.dataset import load_r0_r7, load_r2
from maskedbike_ml.training import clip_with_scale, fit_model
from maskedbike_ml.pipeline import CommonAlignment, CenteredProductPreprocessor, extract_share_cycles, write_json


CASES = {
    10000: {"train": (27, 16, 27), "validation": (11, 8, 11)},
    20000: {"train": (53, 34, 53), "validation": (22, 16, 22)},
    40000: {"train": (105, 70, 105), "validation": (45, 30, 45)},
    60000: {"train": (158, 104, 158), "validation": (67, 46, 67)},
    80000: {"train": (210, 140, 210), "validation": (90, 60, 90)},
}


def select(bundle_ids: np.ndarray, values) -> np.ndarray:
    return np.flatnonzero(np.isin(bundle_ids, list(values)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path, help="dataset-snapshot.json from the 80k run")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(CASES))
    args = parser.parse_args()
    dataset = args.dataset.resolve(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text()); frozen = json.loads(args.snapshot.read_text())
    if any(budget not in CASES for budget in args.budgets):
        raise ValueError(f"budgets must be chosen from {sorted(CASES)}")
    contract = config["data_contract"]; order = int(contract["masking_order"])
    ready_paths = sorted(dataset.rglob("*.h5.ready.json"))
    r07 = load_r0_r7(dataset, ready_paths, order, contract["adc_source"], contract["j_count"])
    r2 = load_r2(dataset, ready_paths, order, contract["adc_source"], contract["j_count"])
    if (r07.geometry != r2.geometry
            or int(frozen["geometry"]["masking_order"]) != r07.geometry.masking_order
            or tuple(map(tuple, frozen["geometry"]["share_intervals"])) != r07.geometry.share_intervals):
        raise ValueError("current data geometry differs from the frozen final snapshot")
    geometry = r07.geometry
    selected = frozen["selected_bundles"]
    heldout_ids = {row["bundle_id"] for row in frozen["r2_heldout"]["bundle_rows"]}
    r0 = list(selected["R0"]); r2_profiling = list(selected["R2_profiling"]); r7 = list(selected["R7"])
    full_train = {"r0": r0[:210], "r2": r2_profiling[:140], "r7": r7[:210]}
    full_validation = {"r0": r0[210:], "r2": r2_profiling[140:], "r7": r7[210:]}
    r2_test = select(r2.bundle_ids, heldout_ids)
    if len(r2_test) != frozen["r2_heldout"]["traces"]:
        raise RuntimeError("held-out R2 snapshot mismatch")

    for budget in args.budgets:
        nr0, nr2, nr7 = CASES[budget]["train"]
        vr0, vr2, vr7 = CASES[budget]["validation"]
        train = select(r07.bundle_ids, full_train["r0"][:nr0] + full_train["r7"][:nr7])
        validation = select(r07.bundle_ids, full_validation["r0"][:vr0] + full_validation["r7"][:vr7])
        r2_train = select(r2.bundle_ids, full_train["r2"][:nr2])
        r2_validation = select(r2.bundle_ids, full_validation["r2"][:vr2])
        if len(train) + len(validation) + len(r2_train) + len(r2_validation) != budget:
            raise RuntimeError(f"{budget}: profiling count mismatch")
        case_output = output / f"profiling-{budget}"; case_output.mkdir(exist_ok=True)
        common_training = np.concatenate([r07.traces[train], r2.traces[r2_train]])
        alignment = CommonAlignment.fit(
            common_training, np.arange(len(common_training)), geometry.share_intervals,
            int(config["preprocessing"]["alignment_max_shift"]), int(config["preprocessing"]["alignment_stride"]),
        )
        cycles07 = extract_share_cycles(r07.traces, alignment.shifts(r07.traces), geometry)
        cycles2 = extract_share_cycles(r2.traces, alignment.shifts(r2.traces), geometry)
        preprocessor = CenteredProductPreprocessor(geometry).fit(np.concatenate([cycles07[train], cycles2[r2_train]]))
        z07 = preprocessor.transform(cycles07); z2 = preprocessor.transform(cycles2)
        scale = float(np.percentile(np.abs(np.concatenate([z07[train], z2[r2_train]])), config["preprocessing"]["clip_percentile"])) or 1.0
        product07 = clip_with_scale(z07, scale); product2 = clip_with_scale(z2, scale)
        result = fit_model(
            case_output, f"masking-order-{order}-sensitivity-{budget}",
            np.concatenate([product07[train], product2[r2_train]]),
            np.concatenate([product07[validation], product2[r2_validation]]),
            product07[train], r07.labels[train], product07[validation], r07.labels[validation],
            product2[r2_test], (r2.hamming_weights[r2_test] != 0).astype(np.uint8), config,
        )
        write_json(case_output / "results.json", {
            "schema": "maskedbike-qshare-sensitivity-case.v2", "budget": budget,
            "geometry": geometry.as_dict(), "bundle_allocation": CASES[budget], "config": config,
            "result": result, "same_r2_heldout_snapshot": str(args.snapshot.resolve()),
            "completed_at_unix": time.time(),
        })
        save_preprocessing(case_output / "preprocessing.npz", alignment, preprocessor, scale)
        artifact_path = write_artifact_manifest(case_output)
        verify_artifact_manifest(artifact_path)
        print(json.dumps({"budget": budget, "geometry": geometry.as_dict(),
                          "r2_test_auc": result["r2_heldout_test"]["roc_auc"],
                          "r2_test_balanced_accuracy": result["r2_heldout_test"]["balanced_accuracy"]}), flush=True)


if __name__ == "__main__":
    main()
