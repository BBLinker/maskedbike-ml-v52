# D3 80,000-trace semi-supervised model

Frozen four-share centered-product -> CSCAE -> NN checkpoint for masking order 3.

## Reporting note

For manuscript and cross-run discussion, describe held-out R2 balanced accuracy as
**approximately 70%**. The exact value for this frozen snapshot is **71.82%**.
This result is specific to the frozen capture campaign, chronological split, and
random seed; it does not imply that increasing the masking order makes the target
easier or harder in general.

## Frozen training and evaluation

| Round | Role | Traces |
|---|---|---:|
| R0 | labeled HW != 0 | 29,890 |
| R2 | unlabeled CSCAE adaptation | 19,902 |
| R7 | labeled HW = 0 | 29,950 |

- Profiling total: 79,742 traces from 800 legal bundles.
- Masking order: 3; shares: 4.
- Trace width: 3,488 samples.
- Share intervals: `[128,944]`, `[944,1760]`, `[1760,2576]`, `[2576,3392]`.
- Cycles: `[N,4,14,56]`; pointwise product: `[N,14,56]`.
- R2 chronological held-out set: 17,530 traces.
- Positive event represented by the model probability: `hw_zero`.
- Frozen threshold: `0.999719133016`.
- R0/R7 validation ROC-AUC: 79.42%.
- R0/R7 validation balanced accuracy: 72.39%.
- R2 held-out ROC-AUC: 78.73%.
- R2 held-out balanced accuracy: 71.82%.

The four shares are centered independently using training-only means at every
`[share,j,time]`, then multiplied pointwise at identical `[j,time]` coordinates.
No time or `j` pooling is used.

## Inference

```python
from maskedbike_ml.inference import Predictor

predictor = Predictor("published/model_D3")
probability, prediction = predictor.predict(traces)
print(predictor.positive_event)  # hw_zero
```

`preprocessing.npz` contains the frozen global alignment, four-share centering,
pointwise-product normalization, clipping, and geometry parameters. Exact
machine-readable training histories and metrics are preserved in `results.json`.
