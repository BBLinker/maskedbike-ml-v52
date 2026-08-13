#!/usr/bin/env python3
"""Evaluate frozen artifacts on an NPZ containing traces and hamming_weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from maskedbike_ml.inference import Predictor
from maskedbike_ml.pipeline import metrics, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="NPZ with traces and hamming_weights")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        traces = data["traces"]; labels = (data["hamming_weights"] != 0).astype(np.uint8)
    predictor = Predictor(args.artifacts)
    probabilities = predictor.probabilities(traces, args.batch_size)
    result = metrics(labels, probabilities, predictor.threshold)
    write_json(args.output, result); print(json.dumps(result, default=lambda value: value.tolist()))


if __name__ == "__main__":
    main()
