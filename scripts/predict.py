#!/usr/bin/env python3
"""Run a frozen pipeline on an NPZ containing a traces array."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from maskedbike_ml.inference import Predictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        traces = data["traces"]
    probability, prediction = Predictor(args.artifacts).predict(traces, args.batch_size)
    np.savez_compressed(args.output, probability=probability, prediction=prediction)


if __name__ == "__main__":
    main()
