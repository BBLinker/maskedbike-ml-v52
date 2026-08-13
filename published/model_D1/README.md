# D1 80,000-trace semi-supervised model

Frozen centered-product -> CSCAE -> NN model trained with 80,000 profiling traces.

| Round | Role | Traces |
|---|---|---:|
| R0 | labeled classifier class 0 | 30,000 |
| R2 | unlabeled CSCAE adaptation | 20,000 |
| R7 | labeled classifier class 1 | 30,000 |

- Complete-bundle chronological split: 70% fit / 30% validation.
- CSCAE training: 56,000 traces; supervised NN training: 42,000 R0/R7 traces.
- Independent chronological R2 test: 11,200 traces.
- R0/R7 validation ROC-AUC: 0.969119.
- R0/R7 validation balanced accuracy: 0.900889.
- R2 held-out ROC-AUC: 0.975135.
- R2 held-out balanced accuracy: 0.910249.
- Frozen threshold: 0.540302355867, selected only from R0/R7 validation.

`model.pt` and `preprocessing.npz` use the current public inference schema and can be loaded with:

```python
from maskedbike_ml.inference import Predictor
predictor = Predictor("published/model_D1")
probability, prediction = predictor.predict(traces)
```

The published output probability was evaluated on R2 using `hamming_weight == 0` as the
positive event. `Predictor.positive_event` records this calibration explicitly.
