#!/usr/bin/env python3
"""Run streaming GJS statistics using a frozen R2 trace classifier."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from maskedbike_ml.gjs import GJSAggregator, reconstruct_support
from maskedbike_ml.inference import Predictor
from maskedbike_ml.pipeline import write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_ids(values) -> np.ndarray:
    return np.asarray(values).astype("U")


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming GJS distance-spectrum estimator")
    parser.add_argument("--artifacts", required=True, type=Path, help="frozen CSCAE/NN run directory")
    parser.add_argument("--query-manifest", required=True, type=Path, help="JSONL, one aligned trace/query H5 pair per line")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-length", required=True, type=int, help="BIKE cyclic block length r")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--min-queries", type=int, default=1000)
    parser.add_argument("--oracle-event", choices=["hw_nonzero", "hw_zero"], default="hw_nonzero")
    parser.add_argument("--expected-direction", choices=["low", "high"], default="low")
    parser.add_argument("--top-distances", type=int, default=0)
    parser.add_argument("--key-weight", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "gjs-state.npz"
    manifest_hash = sha256(args.query_manifest)
    contract = {
        "query_manifest_sha256": manifest_hash,
        "model_sha256": sha256(args.artifacts / "model.pt"),
        "preprocessing_sha256": sha256(args.artifacts / "preprocessing.npz"),
        "block_length": args.block_length,
        "oracle_event": args.oracle_event,
    }
    contract_json = json.dumps(contract, sort_keys=True)
    completed_entries = 0
    processed = []
    if args.resume and state_path.is_file():
        with np.load(state_path, allow_pickle=False) as state:
            if str(state["contract_json"].item()) != contract_json:
                raise ValueError("resume state input/model contract differs")
            aggregator = GJSAggregator.from_state(state)
            completed_entries = int(state["completed_entries"])
            processed = json.loads(str(state["processed_json"].item()))
        if aggregator.block_length != args.block_length:
            raise ValueError("resume state block length differs")
    else:
        aggregator = GJSAggregator.create(args.block_length)
    predictor = Predictor(args.artifacts)
    entries = [json.loads(line) for line in args.query_manifest.read_text().splitlines() if line.strip()]
    same_event = args.oracle_event == predictor.positive_event
    threshold = predictor.threshold if same_event else 1.0 - predictor.threshold
    for entry_index, entry in enumerate(entries):
        if entry_index < completed_entries:
            continue
        ready_path = Path(entry["ready_json"]) if entry.get("ready_json") else None
        if ready_path is not None:
            ready = json.loads(ready_path.read_text())
            if int(ready["run_manifest"]["provenance"]["decoder_round"]) != 2:
                raise ValueError(f"ready manifest is not decoder_round=2: {ready_path}")
            trace_path = ready_path.parent / ready.get("file", ready_path.name.removesuffix(".ready.json"))
            declared_trace_hash = ready.get("sha256") or ready.get("integrity", {}).get("bundle_sha256")
        else:
            if int(entry.get("decoder_round", -1)) != 2:
                raise ValueError("direct H5 entries must explicitly declare decoder_round=2")
            trace_path = Path(entry["trace_h5"])
            declared_trace_hash = entry.get("trace_sha256")
        query_path = Path(entry.get("query_h5", trace_path))
        if declared_trace_hash and sha256(trace_path) != declared_trace_hash:
            raise ValueError(f"trace SHA-256 mismatch: {trace_path}")
        if entry.get("query_sha256") and sha256(query_path) != entry["query_sha256"]:
            raise ValueError(f"query SHA-256 mismatch: {query_path}")
        with h5py.File(trace_path, "r") as traces, h5py.File(query_path, "r") as queries:
            trace_ds = traces[entry.get("trace_dataset", "traces")]
            position_ds = queries[entry.get("positions_dataset", "error_positions")]
            if len(trace_ds) != len(position_ds):
                raise ValueError(f"row-count mismatch: {trace_path} vs {query_path}")
            if "trace_ids" in traces and "trace_ids" in queries:
                if not np.array_equal(text_ids(traces["trace_ids"][:]), text_ids(queries["trace_ids"][:])):
                    raise ValueError(f"trace-id mismatch: {trace_path} vs {query_path}")
            trace_count = len(trace_ds)
            for start in range(0, len(trace_ds), args.batch_size):
                end = min(start + args.batch_size, len(trace_ds))
                probability = predictor.probabilities(np.asarray(trace_ds[start:end]), args.batch_size)
                if not same_event:
                    probability = 1.0 - probability
                aggregator.update(np.asarray(position_ds[start:end]), probability, threshold)
        processed.append({"ready_json": str(ready_path) if ready_path else None,
                          "trace_h5": str(trace_path), "query_h5": str(query_path), "traces": trace_count})
        temporary_state = state_path.with_suffix(".npz.tmp")
        with temporary_state.open("wb") as handle:
            np.savez_compressed(handle, **aggregator.state(), completed_entries=entry_index + 1,
                                processed_json=json.dumps(processed), contract_json=contract_json)
        temporary_state.replace(state_path)

    report = aggregator.report(args.min_queries, args.expected_direction)
    report.update({"schema": "maskedbike-gjs.v1", "oracle_event": args.oracle_event,
                   "model_positive_event": predictor.positive_event,
                   "model_threshold": threshold, "processed": processed,
                   "input_contract": contract})
    ranked = report["ranked_distances"]
    if args.top_distances and args.key_weight:
        selected = {row["distance"] for row in ranked[:args.top_distances]}
        report["support_reconstruction"] = {
            "selected_distances": sorted(selected), "key_weight": args.key_weight,
            "solutions": reconstruct_support(selected, args.block_length, args.key_weight),
        }
    write_json(args.output / "gjs-report.json", report)
    with (args.output / "gjs-distances.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]) if ranked else ["distance"])
        writer.writeheader(); writer.writerows(ranked)
    if ranked:
        ordered = sorted(ranked, key=lambda row: row["distance"])
        x = np.asarray([row["distance"] for row in ordered])
        y = np.asarray([row["soft_rate"] for row in ordered])
        err = np.asarray([1.96 * row["soft_standard_error"] for row in ordered])
        fig, axis = plt.subplots(figsize=(12, 4.8))
        axis.plot(x, y, linewidth=.8); axis.fill_between(x, y - err, y + err, alpha=.2)
        axis.set(xlabel="cyclic distance d", ylabel=f"P({args.oracle_event} | d in query)", title="GJS distance-spectrum estimate")
        axis.grid(alpha=.25); fig.tight_layout(); fig.savefig(args.output / "gjs-spectrum.png", dpi=180); plt.close(fig)
    print(json.dumps({"status": "complete", "traces": aggregator.traces_seen,
                      "eligible_distances": report["eligible_distances"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
