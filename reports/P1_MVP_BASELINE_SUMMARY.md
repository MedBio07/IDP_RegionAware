# P1 MVP Baseline Summary

Date: 2026-07-31

## Scope

P1 was started with a minimal, reproducible baseline stack:

- ESM2 embedding extraction using `fair-esm` from `/opt/anaconda3/bin/python`.
- Shallow residue classifier: `sklearn.SGDClassifier(loss=log_loss)` with class-balanced sample weights.
- Features tested:
  - `onehot,position`
  - `esm,position`
  - `esm,position,onehot`
- Fixed threshold selected only on `data/DM1229_Validation.fasta` by validation Fmax.
- Evaluation on SL329, MXD494, DISORDER723 with `-1` labels masked.
- First-pass NR25 training for each target test set.

ProtT5 and LoRA were not run in this first P1 pass. The local conda environment has `torch`, `fair-esm`, `numpy`, and `sklearn`; it does not currently have `transformers`, and no local ProtT5 cache was found during the environment check.

## New P1 Assets

Scripts and model utilities:

- `scripts/extract_plm_embeddings.py`
- `scripts/train_disorder_model.py`
- `scripts/predict_disorder_model.py`
- `models/features.py`
- `configs/baseline_esm2_t12_terminal_sgd.yaml`

Generated ESM2 cache:

- `data/features/esm2_embeddings/esm2_t12_35M_UR50D_layer12_fp16/`
- Unique sequences: 5471
- Cache size: about 1.4GB
- Model: `esm2_t12_35M_UR50D`, layer 12, fp16 per-protein `.npy`

Main result files:

- `results/baselines/P1_BASELINE_COMPARISON.tsv`
- `results/baselines/p1_esm2_t12_position_onehot_*_metrics.tsv`
- `results/stratified/p1_esm2_t12_position_onehot_*_stratified.tsv`
- `results/baselines/p1_nr25_*_metrics.tsv`
- `results/stratified/p1_nr25_*_stratified.tsv`

## Overall Results

| Experiment | Train | Dataset | AUC | AUPR | MCC | BACC | SOTA AUC | AUC gap |
|---|---|---|---:|---:|---:|---:|---:|---:|
| B0 onehot+position | DM3000 | SL329 | 0.579422 | 0.509702 | 0.107943 | 0.536671 | 0.915 | -0.335578 |
| B0 onehot+position | DM3000 | MXD494 | 0.591639 | 0.300714 | 0.118937 | 0.547245 | 0.842 | -0.250361 |
| B0 onehot+position | DM3000 | DISORDER723 | 0.758854 | 0.240468 | 0.257143 | 0.667930 | 0.943 | -0.184146 |
| ESM2+position | DM3000 | SL329 | 0.890594 | 0.874041 | 0.658724 | 0.826472 | 0.915 | -0.024406 |
| ESM2+position | DM3000 | MXD494 | 0.814964 | 0.532735 | 0.448361 | 0.752922 | 0.842 | -0.027036 |
| ESM2+position | DM3000 | DISORDER723 | 0.899222 | 0.516817 | 0.493658 | 0.774147 | 0.943 | -0.043778 |
| ESM2+position+onehot | DM3000 | SL329 | 0.894307 | 0.887082 | 0.664334 | 0.828056 | 0.915 | -0.020693 |
| ESM2+position+onehot | DM3000 | MXD494 | 0.817631 | 0.538147 | 0.449810 | 0.752390 | 0.842 | -0.024369 |
| ESM2+position+onehot | DM3000 | DISORDER723 | 0.900286 | 0.536881 | 0.500994 | 0.770947 | 0.943 | -0.042714 |

The current best P1 baseline is `ESM2+position+onehot`. It is a useful starting point but not yet competitive with IDP-EDL/FusionEncoder by AUC. SL329 BACC is already close to the current literature baseline, but AUC and MCC still lag.

## NR25 First-Pass Results

| Train | Dataset | AUC | AUPR | MCC | BACC | Full-train AUC | AUC drop |
|---|---|---:|---:|---:|---:|---:|---:|
| NR25-vs-SL329 | SL329 | 0.891377 | 0.883087 | 0.660777 | 0.826587 | 0.894307 | -0.002930 |
| NR25-vs-MXD494 | MXD494 | 0.805165 | 0.523738 | 0.434687 | 0.739079 | 0.817631 | -0.012466 |
| NR25-vs-DISORDER723 | DISORDER723 | 0.898145 | 0.536394 | 0.503762 | 0.773404 | 0.900286 | -0.002141 |

Interpretation:

- SL329 and DISORDER723 are stable under NR25 filtering.
- MXD494 drops more, suggesting either more useful train/test homology was removed or this test set needs stronger features/fusion.
- NR25 stability is good enough to keep leakage-controlled evaluation as a central paper asset.

## Hard-Case Findings

For the best full-training P1 baseline, terminal regions remain much easier than internal regions:

| Dataset | Terminal IDR AUC | Internal IDR AUC | Gap |
|---|---:|---:|---:|
| SL329 | 0.915156 | 0.846417 | 0.068739 |
| MXD494 | 0.832379 | 0.760439 | 0.071940 |
| DISORDER723 | 0.939863 | 0.804272 | 0.135591 |

SDR/LDR behavior is dataset-dependent:

| Dataset | SDR AUC | LDR AUC | Observation |
|---|---:|---:|---|
| SL329 | 0.809246 | 0.907266 | LDR easier |
| MXD494 | 0.765948 | 0.826862 | LDR easier |
| DISORDER723 | 0.905978 | 0.882508 | SDR easier |

These results support the planned P2 direction: region-aware experts are justified, especially for internal IDRs and dataset-specific SDR/LDR differences.

## Issues Observed

- The SGD models reached `max_iter` before convergence. Results are usable for P1, but P2 should add better optimization or an incremental training loop.
- Validation-selected thresholds are close to 1.0, indicating poor raw probability calibration. This reinforces the P3 calibration plan.
- No ProtT5 baseline was run yet because `transformers` is not installed in the current conda environment and no local ProtT5 weights were found.

## P1 Decision

Go to P2 method development, but do not claim SOTA yet.

Rationale:

1. The P1 pipeline is fully operational: embeddings, training, prediction, overall evaluation, stratified evaluation, and NR25 evaluation.
2. The ESM2 baseline is within about 0.02-0.04 AUC of current direct SOTA, close enough to justify model/fusion work.
3. The hard-case analysis found a clear weakness in internal IDRs, which gives P2 a concrete target.
4. NR25 results are mostly stable, so leakage control is feasible for the manuscript.

Immediate P2 priorities:

1. Replace linear SGD with a small sequence model: CNN/TCN or BiGRU head over frozen ESM2 features.
2. Add region-aware heads for terminal/internal and SDR/LDR.
3. Add a validation-calibrated probability layer or at least temperature scaling.
4. Add ProtT5 only after the ESM2 sequence model baseline is stable.
