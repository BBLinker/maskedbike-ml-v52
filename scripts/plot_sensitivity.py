#!/usr/bin/env python3
"""Aggregate sensitivity cases into CSV, JSON, and a publication-ready plot."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.input.resolve(); output = (args.output or root).resolve(); output.mkdir(parents=True, exist_ok=True)
    rows = []
    for budget in (10000, 20000, 40000, 60000, 80000):
        document = json.loads((root / f"profiling-{budget}" / "results.json").read_text())
        result = document["result"]
        rows.append({
            "profiling_traces": budget,
            "r0r7_validation_auc": result["r0_r7_validation"]["roc_auc"],
            "r0r7_validation_balanced_accuracy": result["r0_r7_validation"]["balanced_accuracy"],
            "r2_test_auc": result["r2_heldout_test"]["roc_auc"],
            "r2_test_pr_auc": result["r2_heldout_test"]["pr_auc"],
            "r2_test_balanced_accuracy": result["r2_heldout_test"]["balanced_accuracy"],
            "r2_test_mcc": result["r2_heldout_test"]["mcc"],
            "r2_test_traces": result["r2_heldout_test"]["traces"],
            "classifier_epochs": len(result["classifier_history"]),
        })
    with (output / "sensitivity-results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "sensitivity-results.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")

    x = np.array([row["profiling_traces"] for row in rows]) / 1000
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, key, color, marker in (
        ("R0/R7 validation ROC-AUC", "r0r7_validation_auc", "#377eb8", "o"),
        ("R2 held-out ROC-AUC", "r2_test_auc", "#e41a1c", "s"),
    ):
        y = np.array([row[key] for row in rows]); axes[0].plot(x, y, marker=marker, lw=2.2, label=label, color=color)
        for xx, yy in zip(x, y): axes[0].text(xx, yy + .004, f"{yy * 100:.2f}%", ha="center", fontsize=8, color=color)
    for label, key, color, marker in (
        ("R0/R7 validation", "r0r7_validation_balanced_accuracy", "#4daf4a", "o"),
        ("R2 held-out test", "r2_test_balanced_accuracy", "#984ea3", "s"),
    ):
        y = np.array([row[key] for row in rows]); axes[1].plot(x, y, marker=marker, lw=2.2, label=label, color=color)
        for xx, yy in zip(x, y): axes[1].text(xx, yy + .004, f"{yy * 100:.2f}%", ha="center", fontsize=8, color=color)
    axes[0].set_ylim(.85, 1); axes[0].set_ylabel("ROC-AUC")
    axes[1].set_ylim(.75, .95); axes[1].set_ylabel("Balanced accuracy")
    for axis in axes:
        axis.set_xticks(x); axis.set_xlabel("Profiling traces (thousands)"); axis.legend(loc="lower right"); axis.grid(alpha=.25)
    figure.suptitle("Masked BIKE q-share profiling-trace sensitivity analysis\nFixed model, split, seed, and independent R2 test", fontweight="bold")
    figure.tight_layout(); figure.savefig(output / "profiling-count-sensitivity.png", dpi=180); plt.close(figure)


if __name__ == "__main__":
    main()
