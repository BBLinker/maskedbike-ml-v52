# D1 held-out stratified error-rate analysis

- Frozen R2 held-out traces: 11200
- Positive event: `delta=1 iff HW=0`
- Frozen threshold: `0.5403023559`
- Literal overlap range: 1626–1886 unique key-spectrum distances per ciphertext.

## Requested groups

|group|n|TPR (95% bootstrap CI)|TNR (95% bootstrap CI)|
|---:|---:|---:|---:|
|0|0|NA [NA, NA]|NA [NA, NA]|
|1|0|NA [NA, NA]|NA [NA, NA]|
|2|0|NA [NA, NA]|NA [NA, NA]|
|>=3|11200|0.8939 [0.8870, 0.9007]|0.9266 [0.9178, 0.9353]|

## Exploratory overlap-count quartiles

|group|n|TPR|TNR|Fisher TPR vs Q1|Fisher TNR vs Q1|
|---|---:|---:|---:|---:|---:|
|Q1|2920|0.8884|0.9259|1|1|
|Q2|2743|0.9052|0.9222|0.0912|0.789|
|Q3|2751|0.8959|0.9272|0.473|0.926|
|Q4|2786|0.8870|0.9314|0.92|0.709|

The requested categorical test is not identifiable on this valid-ciphertext held-out set because every ciphertext falls in `>=3`. Quartile results are exploratory and do not replace a GJS chosen-ciphertext stratification.

## Interpretation

The requested `0/1/2/>=3` trace-level bins are degenerate for this valid-ciphertext set: every sample contains at least three matching key-spectrum distances. The overlap-count quartiles are therefore a sensitivity analysis over the observed high-overlap regime, not a direct validation of non-overlap or low-multiplicity GJS queries.

Across these quartiles, all raw Fisher exact p-values are at least 0.091 for TPR and 0.709 for TNR. This supports the constant class-conditional error-rate approximation within the observed regime, but does not establish it for deliberately constructed low-overlap ciphertexts.

## Reproduction

Use `scripts/stratified_error_rates.py` with a frozen model run, its prediction archive and dataset snapshot, the D1 H5 dataset, the deterministic BIKE reconstruction helper, and the experimental key seed. The script verifies every replayed ciphertext before calculating the spectrum overlap.
