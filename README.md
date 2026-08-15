# Masked BIKE q-share machine-learning pipeline

This repository implements the centered-product CSCAE pipeline for multiple masking orders. It is not limited to two shares.

## Supported masking orders

For masking order `d`, the pipeline requires and jointly processes:

```text
q = d + 1 shares
```

| Masking order | Shares | Combining order |
|---:|---:|---|
| 1 | 2 | second-order centered product |
| 2 | 3 | third-order centered product |
| 3 | 4 | fourth-order centered product |
| 4 | 5 | fifth-order centered product |
| 5 | 6 | sixth-order centered product |

The configuration files `configs/masking-order-1.json` through `configs/masking-order-5.json` contain the complete data, preprocessing, model, optimization, early-stopping, evaluation, and output settings for each order.

## Geometry comes from the capture manifest

The code does not hard-code two share intervals or a fixed trace length. Every accepted ready manifest supplies:

- masking order and therefore the required share count;
- complete stored interval for every share;
- trace sample count;
- 14 matching-position records;
- ADC source and decoder round.

All bundles in one run must have exactly the same geometry. The `j` stride is inferred from consecutive `matching_j_positions`. The extracted cycle width is the largest uniform window that remains inside every declared share interval, capped at the `j` stride. Matching landmarks are diagnostic and may have different local offsets across shares; they never rotate a share. Every cycle is anchored at its share-interval start:

```text
cycles.shape = [N, q, 14, cycle_samples]
```

For every trace, all `q` shares use one common global shift. Each share is extracted without phase rotation:

```text
S_i[j,t] = trace[share_start_i + j_stride_samples*j + t + global_shift]
```

The training-only share mean has shape `[q,14,cycle_samples]`. The higher-order feature is:

```text
P[j,t] = product_i(S_i[j,t] - mean_i[j,t]),  i=0..q-1
```

The same `(j,t)` coordinate is multiplied across every share. There is no share-wise alignment, per-`j` alignment, pooling, phase rotation, or cross-time multiplication.

After the q-share product is formed, its shape is `[N,14,cycle_samples]` for every masking order. It is standardized, clipped with training-only statistics, flattened in ordered `[j,time]` order, denoised by one CSCAE, and classified from the CSCAE latent representation by a four-block neural network.

## Semi-supervised profiling design

The default complete configuration allocates 80,000 profiling traces:

| Round | Role | Traces |
|---|---|---:|
| R0 | labeled class 0 | 30,000 |
| R2 | unlabeled CSCAE reconstruction | 20,000 |
| R7 | labeled class 1 | 30,000 |

Complete bundles are split chronologically into 70% fitting and 30% validation partitions. R2 profiling labels are never supplied to the classifier. Later R2 bundles form an independent chronological test partition and do not contribute to alignment, preprocessing, CSCAE fitting, checkpoint selection, or threshold selection.

## Trace dataset

The raw H5 bundles are published separately in
[`BBLinker/maskedbike-traces`](https://github.com/BBLinker/maskedbike-traces).
They are GitHub Release assets rather than files in this repository, so cloning
the ML code does not download hundreds of megabytes of immutable captures.

Select D1, D2, or D3 explicitly when downloading. For the currently published
D1 release:

```bash
git clone https://github.com/BBLinker/maskedbike-traces.git
cd maskedbike-traces
python scripts/download_dataset.py --dataset D1 --release v1.0.0 --output datasets
python scripts/verify_dataset.py datasets/maskedbike-traces-D1-v1.0.0 --dataset D1
```

The extracted directory contains `BIKE/` and `manifests/`. Pass that extracted
directory to `--dataset`; training recursively discovers the paired
`*.h5.ready.json` manifests. The D1 v1.0.0 release contains 150,000 traces:
50,000 each from R0, R2, and R7. Its summary, frozen bundle snapshot, SHA-256,
and exact capture contract are maintained by the dataset repository. D2 and D3
use the same archive, snapshot, checksum, download, and verification interface;
their manifests supply their own share count and geometry rather than reusing D1
preprocessing parameters.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Training

Choose the configuration matching the dataset's declared masking order. For example, four shares means masking order 3:

```bash
python scripts/train.py \
  --dataset /path/to/masking-order-3-dataset \
  --config configs/masking-order-3.json \
  --output /path/to/output
```

Two-share example:

```bash
python scripts/train.py \
  --dataset /path/to/masking-order-1-dataset \
  --config configs/masking-order-1.json \
  --output /path/to/output
```

The run rejects bundles whose manifest masking order, share count, geometry, ADC source, or H5 trace width differs from the selected configuration.

## Profiling-count sensitivity analysis

After producing the final snapshot:

```bash
python scripts/sensitivity.py \
  --dataset /path/to/dataset \
  --config configs/masking-order-3.json \
  --snapshot /path/to/final-run/dataset-snapshot.json \
  --output /path/to/sensitivity-output \
  --budgets 10000 20000 40000 60000 80000

python scripts/plot_sensitivity.py --input /path/to/sensitivity-output
```

All sensitivity cases reuse the same architecture, seed, chronological bundle pools, and independent R2 test snapshot. Only the profiling-trace count changes.

## GJS analysis with a large R2 campaign

`scripts/gjs.py` turns the frozen R2 classifier into the oracle used by the
Guo--Johansson--Stankovski distance-spectrum procedure. It consumes bundles in
streaming batches, so the complete R2 campaign is not loaded into RAM.

GJS requires the error positions used to construct every chosen query. Existing
trace bundles that contain only traces, ciphertexts, syndromes, and case seeds do
not contain enough information by themselves. Save an aligned sidecar H5 with:

```text
error_positions: int array [N, max_weight], padded with -1
trace_ids:       strings [N], identical and in the same order as the trace H5
```

Freeze the input set as JSONL, one line per R2 bundle:

```json
{"ready_json":"/data/R2/bundle.h5.ready.json","query_h5":"/data/R2/bundle.gjs.h5","query_sha256":"SHA256"}
```

For a direct H5 entry without a ready manifest, explicitly include
`"trace_h5"`, `"trace_sha256"`, and `"decoder_round": 2`. The runner checks the
ready-manifest round, declared hashes, row counts, and trace IDs before updating
statistics.

```bash
python scripts/gjs.py \
  --artifacts /path/to/frozen-model-run \
  --query-manifest /path/to/r2-gjs-snapshot.jsonl \
  --output /path/to/gjs-output \
  --block-length R \
  --batch-size 512 \
  --min-queries 1000 \
  --oracle-event hw_nonzero \
  --expected-direction low
```

The output includes resumable accumulators, per-distance query counts, soft and
hard conditional oracle rates, standard errors, 95% confidence intervals, CSV,
JSON, and `gjs-spectrum.png`. If the number of selected spectrum distances and
the block weight are already fixed, add `--top-distances COUNT --key-weight W`
to run support reconstruction up to cyclic rotation/reflection.

The soft statistic is `P(oracle event | distance d occurs in the query error
block)`. `--expected-direction low` matches the original GJS observation when
the measured event rate decreases for secret-spectrum distances; use `high`
when calibration on a held-out synthetic-key campaign shows the opposite sign.

## Optional GJS error reconstruction

This is not part of normal ML training or evaluation. Only an analyst actually
running a GJS experiment needs these derived data. Source H5 bundles remain
unchanged and no error sidecars are generated in advance.

When needed, build the deterministic replay helper from the BIKE v5.2 reference
archive recorded by the capture provenance:

```bash
python scripts/build_gjs_reconstructor.py \
  --reference-archive /path/to/Reference_Implementation.2024.10.10.1.zip \
  --output /path/to/bike-v52-error-reconstructor
```

Then replay the stored case seeds for the selected dataset:

```bash
python scripts/reconstruct_gjs_errors.py \
  --dataset /path/to/dataset \
  --helper /path/to/bike-v52-error-reconstructor \
  --key-seed KEY_SEED_HEX \
  --output /path/to/gjs-errors
```

The runner regenerates encapsulation `e0/e1`, requires total error weight 134,
and verifies every regenerated ciphertext byte-for-byte against its source H5
before writing an independent sidecar. The resulting e0/e1 snapshot can then be
passed to `scripts/gjs.py`.

## Metrics

- **ROC-AUC** evaluates score-ranking discrimination across all thresholds.
- **Balanced accuracy** evaluates mean class recall at the threshold selected only from R0/R7 validation data.

R2 test labels are opened only after the model, preprocessing statistics, checkpoint, and threshold are frozen.

## Frozen evaluation and inference

Curated frozen checkpoints are published in [`published/model_D1`](published/model_D1),
[`published/model_D2`](published/model_D2), and [`published/model_D3`](published/model_D3).
Each directory contains only the model, preprocessing parameters, metrics,
checksums, and model card; dataset snapshots, raw traces, and per-trace
predictions are intentionally excluded.

```python
from maskedbike_ml.inference import Predictor

predictor = Predictor("published/model_D1")
probability, prediction = predictor.predict(traces)
print(predictor.positive_event)
```

`positive_event` records which event class the stored probability and threshold
represent, so callers and the GJS runner invert scores only when their requested
event differs from the model calibration.

Training writes a complete, hash-verified artifact set: the immutable dataset snapshot,
training-only alignment and centering statistics, model checkpoint, predictions, metrics,
and `artifact-manifest.json`. Reuse those frozen parameters instead of fitting anything on
evaluation data:

```bash
python scripts/evaluate.py --artifacts /path/to/run --input evaluation.npz --output metrics.json
python scripts/predict.py --artifacts /path/to/run --input traces.npz --output predictions.npz
```

`evaluation.npz` contains `traces` and `hamming_weights`; `traces.npz` contains `traces`.
The global target is always `hamming_weight != 0`.
