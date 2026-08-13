# D2 80,000-trace semi-supervised model

Frozen three-share centered-product -> CSCAE -> NN checkpoint for masking order 2.

## Conservative reporting note

For manuscript and general cross-run description, interpret held-out R2 balanced
accuracy as **approximately 90%**, not as a guaranteed value. The exact result
for this frozen dataset snapshot is **94.80%**.
The difference may reflect capture-campaign quality, chronological split,
dataset snapshot, or random-seed variation. This single D2 result is not evidence
that increasing the masking order systematically makes the target easier to
classify.

## Frozen training and evaluation

| Round | Role | Traces |
|---|---|---:|
| R0 | labeled HW != 0 | 30,000 |
| R2 | unlabeled CSCAE adaptation | 20,000 |
| R7 | labeled HW = 0 | 30,000 |

- Masking order: 2; shares: 3.
- Trace width: 2,672 samples.
- Share intervals: `[128,944]`, `[944,1760]`, `[1760,2576]`.
- Cycles: `[N,3,14,56]`; pointwise product: `[N,14,56]`.
- R2 chronological held-out set: 25,700 traces.
- Positive event represented by the model probability: `hw_zero`.
- Frozen threshold: `0.886666147829`.
- R0/R7 validation ROC-AUC: 98.27%.
- R0/R7 validation balanced accuracy: 93.52%.
- R2 held-out ROC-AUC: 98.90%.
- R2 held-out balanced accuracy: 94.80%.

## Inference

```python
from maskedbike_ml.inference import Predictor

predictor = Predictor("published/model_D2")
probability, prediction = predictor.predict(traces)
print(predictor.positive_event)  # hw_zero
```

`preprocessing.npz` contains the frozen alignment, three-share centering,
product standardization, clipping, and geometry parameters. Exact machine-readable
metrics are preserved in `results.json`.
