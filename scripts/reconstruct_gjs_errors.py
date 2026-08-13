#!/usr/bin/env python3
"""Replay R2 case seeds and produce verified e0/e1 GJS sidecars."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

R_BITS = 12323
R_BYTES = (R_BITS + 7) // 8
TRANSPORT_BYTES = 197 * 8
SYNDROME_TRANSPORT_BYTES = 193 * 8
T1 = 134


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_positions(text: str) -> np.ndarray:
    if not text:
        return np.empty(0, np.int32)
    values = np.fromiter((int(item) for item in text.split(",")), dtype=np.int32)
    if len(values) and (values.min() < 0 or values.max() >= R_BITS or len(np.unique(values)) != len(values)):
        raise ValueError("helper returned invalid error positions")
    return values


def transport_ciphertext(canonical: bytes) -> bytes:
    if len(canonical) != R_BYTES + 32:
        raise ValueError("unexpected canonical ciphertext length")
    output = bytearray(TRANSPORT_BYTES)
    output[:R_BYTES] = canonical[:R_BYTES]
    output[SYNDROME_TRANSPORT_BYTES:] = canonical[R_BYTES:]
    return bytes(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--helper", required=True, type=Path)
    parser.add_argument("--key-seed", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit-bundles", type=int)
    args = parser.parse_args()
    key_seed = bytes.fromhex(args.key_seed)
    if len(key_seed) != 32 or not any(key_seed):
        raise ValueError("key seed must be one nonzero 32-byte hexadecimal value")
    key_fingerprint = hashlib.sha256(key_seed).hexdigest()
    candidates = sorted(args.dataset.resolve().rglob("*.h5.ready.json"))
    ready_paths = []
    for path in candidates:
        ready = json.loads(path.read_text())
        provenance = ready["run_manifest"]["provenance"]
        if int(provenance["decoder_round"]) == 2:
            if provenance.get("key_fingerprint") != key_fingerprint:
                raise ValueError(f"key fingerprint mismatch: {path}")
            ready_paths.append(path)
    if args.limit_bundles is not None:
        ready_paths = ready_paths[:args.limit_bundles]
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = {"e0": [], "e1": []}
    helper_hash = sha256(args.helper)
    process = subprocess.Popen([str(args.helper.resolve()), args.key_seed], text=True,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert process.stdin is not None and process.stdout is not None
    totals = {"bundles": 0, "traces": 0, "ciphertexts_verified": 0}
    try:
        for ready_path in ready_paths:
            ready = json.loads(ready_path.read_text())
            source = ready_path.parent / ready.get("file", ready_path.name.removesuffix(".ready.json"))
            source_hash = sha256(source)
            declared = ready.get("sha256") or ready.get("integrity", {}).get("bundle_sha256")
            if declared and source_hash != declared:
                raise ValueError(f"source H5 SHA-256 mismatch: {source}")
            with h5py.File(source, "r") as h5:
                seeds = np.asarray(h5["case_seeds"], dtype=np.uint8)
                trace_ids = np.asarray(h5["trace_ids"]).astype("S64")
                ciphertexts = np.asarray(h5["ciphertexts"], dtype=np.uint8)
            e0 = np.full((len(seeds), T1), -1, np.int32)
            e1 = np.full((len(seeds), T1), -1, np.int32)
            w0 = np.empty(len(seeds), np.uint16); w1 = np.empty(len(seeds), np.uint16)
            for index, seed in enumerate(seeds):
                seed_hex = seed.tobytes().hex()
                process.stdin.write(seed_hex + "\n"); process.stdin.flush()
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError("reconstruction helper ended early")
                returned_seed, ciphertext_hex, e0_text, e1_text = line.rstrip("\n").split("\t")
                if returned_seed != seed_hex:
                    raise ValueError("helper case-seed order mismatch")
                if transport_ciphertext(bytes.fromhex(ciphertext_hex)) != ciphertexts[index].tobytes():
                    raise ValueError(f"reconstructed ciphertext mismatch: {source} row {index}")
                p0, p1 = parse_positions(e0_text), parse_positions(e1_text)
                if len(p0) + len(p1) != T1:
                    raise ValueError(f"reconstructed total error weight is not {T1}")
                e0[index, :len(p0)] = p0; e1[index, :len(p1)] = p1
                w0[index] = len(p0); w1[index] = len(p1)
                totals["ciphertexts_verified"] += 1
            relative_parent = ready_path.parent.relative_to(args.dataset.resolve())
            destination_dir = args.output / relative_parent
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / (source.stem + ".gjs.h5")
            with h5py.File(destination, "w") as sidecar:
                sidecar["trace_ids"] = trace_ids
                sidecar["case_seeds"] = seeds
                sidecar["error_positions_e0"] = e0
                sidecar["error_positions_e1"] = e1
                sidecar["error_weight_e0"] = w0
                sidecar["error_weight_e1"] = w1
                sidecar.attrs["schema"] = "maskedbike-gjs-errors.v1"
                sidecar.attrs["source_h5_sha256"] = source_hash
                sidecar.attrs["helper_sha256"] = helper_hash
                sidecar.attrs["key_fingerprint"] = key_fingerprint
                sidecar.attrs["r_bits"] = R_BITS
                sidecar.attrs["total_error_weight"] = T1
            sidecar_hash = sha256(destination)
            with h5py.File(destination, "r") as check:
                if not np.array_equal(np.asarray(check["trace_ids"]), trace_ids):
                    raise ValueError("sidecar verification failed")
            for block in ("e0", "e1"):
                manifests[block].append({"ready_json": str(ready_path.resolve()),
                                         "query_h5": str(destination.resolve()),
                                         "query_sha256": sidecar_hash,
                                         "positions_dataset": f"error_positions_{block}"})
            totals["bundles"] += 1; totals["traces"] += len(seeds)
            print(json.dumps({"bundle": source.name, "traces": len(seeds),
                              "e0_weight_mean": float(w0.mean()), "e1_weight_mean": float(w1.mean())}), flush=True)
    finally:
        process.stdin.close(); return_code = process.wait()
        if return_code:
            raise RuntimeError(f"reconstruction helper exited with {return_code}")
    for block, rows in manifests.items():
        path = args.output / f"r2-gjs-{block}-snapshot.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report = {"schema": "maskedbike-gjs-reconstruction.v1", **totals,
              "key_fingerprint": key_fingerprint, "helper_sha256": helper_hash,
              "dataset": str(args.dataset.resolve())}
    (args.output / "reconstruction-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
