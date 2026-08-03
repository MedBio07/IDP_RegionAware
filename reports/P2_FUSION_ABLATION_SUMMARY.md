# P2 Fusion/Ablation Summary

Date: 2026-07-31

## Scope

P2 has started with a frozen-feature sequence model over the existing ESM2 cache:

- Model: `RegionAwareTCN`
- Features: `esm2_t12,position,onehot`
- Train/validation: `data/DM3000_Train.fasta` / `data/DM1229_Validation.fasta`
- Test sets: SL329, MXD494, DISORDER723
- Threshold: validation Fmax threshold from DM1229, `0.777697`
- Best checkpoint selection: validation AUC

The current run is a first full-model pass, not a completed ablation suite.

## New P2 Assets

- `models/sequence_models.py`
- `scripts/train_sequence_disorder_model.py`
- `scripts/predict_sequence_disorder_model.py`
- `configs/p2_region_aware_tcn_esm2.yaml`
- `models/p2_region_aware_tcn_esm2_position_onehot_seed1.pt`
- `predictions/fusion/p2_region_aware_tcn_*.tsv`
- `results/fusion/p2_region_aware_tcn_*_metrics.tsv`
- `results/stratified/p2_region_aware_tcn_*_stratified.tsv`
- `results/fusion/P2_FIRST_PASS_COMPARISON.tsv`
- `results/fusion/P2_ABLATION_COMPARISON.tsv`
- `results/fusion/P2_MULTISEED_RESULTS.tsv`
- `results/fusion/P2_MULTISEED_SUMMARY.tsv`
- `results/fusion/P2_NR25_COMPARISON.tsv`
- `scripts/ensemble_disorder_predictions.py`
- `results/fusion/P2_ENSEMBLE_COMPARISON.tsv`

Implementation note: the prediction script was patched to accept both list-form and comma-separated string-form `features` metadata, because the first checkpoint was saved after metrics metadata overwrote the feature list.

## Validation Result

Best epoch was epoch 3:

| Metric | Value |
|---|---:|
| Validation AUC | 0.898529 |
| Validation AUPR | 0.646963 |
| Validation Fmax | 0.606999 |
| Validation MCC | 0.571000 |
| Threshold | 0.777697 |

## Overall Test Results

| Dataset | P1 Best AUC | P2 AUC | Delta AUC | P1 Best MCC | P2 MCC | Delta MCC | P2 AUPR | Literature Best AUC | Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SL329 | 0.894307 | 0.912342 | +0.018035 | 0.664334 | 0.714500 | +0.050166 | 0.913725 | 0.915 | -0.002658 |
| MXD494 | 0.817631 | 0.840312 | +0.022681 | 0.449810 | 0.483621 | +0.033811 | 0.579960 | 0.842 | -0.001688 |
| DISORDER723 | 0.900286 | 0.917155 | +0.016869 | 0.500994 | 0.550765 | +0.049771 | 0.613054 | 0.943 | -0.025845 |

Interpretation:

- The sequence model improves all three test sets over the P1 shallow ESM2 baseline.
- SL329 and MXD494 are now very close to the strongest collected literature AUC values.
- DISORDER723 improves clearly over P1 and reaches IDP-Fusion-level AUC, but remains below the strongest recent methods.
- This is strong enough to continue P2, but not enough to claim a final SOTA method without ablations, repeated seeds, calibration, and NR25-controlled training.

## Hard-Case Results

Terminal/internal IDR AUC:

| Dataset | P1 Terminal | P2 Terminal | Delta | P1 Internal | P2 Internal | Delta |
|---|---:|---:|---:|---:|---:|---:|
| SL329 | 0.915156 | 0.929797 | +0.014641 | 0.846417 | 0.872247 | +0.025830 |
| MXD494 | 0.832379 | 0.854916 | +0.022537 | 0.760439 | 0.783681 | +0.023242 |
| DISORDER723 | 0.939863 | 0.950101 | +0.010238 | 0.804272 | 0.837225 | +0.032953 |

SDR/LDR AUC:

| Dataset | P1 SDR | P2 SDR | Delta | P1 LDR | P2 LDR | Delta |
|---|---:|---:|---:|---:|---:|---:|
| SL329 | 0.809246 | 0.836327 | +0.027081 | 0.907266 | 0.923923 | +0.016657 |
| MXD494 | 0.765948 | 0.781608 | +0.015660 | 0.826862 | 0.850797 | +0.023935 |
| DISORDER723 | 0.905978 | 0.923002 | +0.017024 | 0.882508 | 0.898889 | +0.016381 |

The region-aware TCN improves the exact weak cases identified in P1, especially internal IDRs. This supports the paper's proposed focus on region-aware modeling, but the expert/auxiliary contribution still needs to be isolated.

## First Ablation Results

Three minimal P2 controls were added:

- `GenericTCN`: same frozen ESM2/position/onehot features and TCN context, but no region experts and no auxiliary heads.
- `AuxiliaryTCN`: same TCN context plus SDR/LDR/terminal/internal auxiliary heads, but no expert gate.
- `RegionAwareTCN no-aux`: keeps the gated expert architecture, but sets `aux_loss_weight=0.0`.

| Variant | Dataset | AUC | AUPR | MCC | BACC | Validation AUC |
|---|---|---:|---:|---:|---:|---:|
| GenericTCN | SL329 | 0.916141 | 0.918419 | 0.718719 | 0.848098 | 0.899191 |
| GenericTCN | MXD494 | 0.830158 | 0.523015 | 0.481100 | 0.764916 | 0.899191 |
| GenericTCN | DISORDER723 | 0.917828 | 0.602995 | 0.549657 | 0.783655 | 0.899191 |
| AuxiliaryTCN | SL329 | 0.911850 | 0.911366 | 0.711510 | 0.845288 | 0.898859 |
| AuxiliaryTCN | MXD494 | 0.840191 | 0.565101 | 0.487300 | 0.768667 | 0.898859 |
| AuxiliaryTCN | DISORDER723 | 0.917525 | 0.610705 | 0.546892 | 0.784522 | 0.898859 |
| RegionAwareTCN no-aux | SL329 | 0.916079 | 0.919629 | 0.716827 | 0.847636 | 0.898374 |
| RegionAwareTCN no-aux | MXD494 | 0.828550 | 0.513951 | 0.479102 | 0.764176 | 0.898374 |
| RegionAwareTCN no-aux | DISORDER723 | 0.917909 | 0.603357 | 0.548213 | 0.783154 | 0.898374 |
| RegionAwareTCN + aux | SL329 | 0.912342 | 0.913725 | 0.714500 | 0.842342 | 0.898529 |
| RegionAwareTCN + aux | MXD494 | 0.840312 | 0.579960 | 0.483621 | 0.763332 | 0.898529 |
| RegionAwareTCN + aux | DISORDER723 | 0.917155 | 0.613054 | 0.550765 | 0.779175 | 0.898529 |

Interpretation:

- Generic sequence context is already a major source of improvement over P1. On SL329 it reaches AUC 0.916141, slightly above the collected literature best AUC 0.915.
- The current expert gate does not give a uniform AUC gain. It helps MXD494 AUC only when auxiliary region supervision is enabled.
- Auxiliary region supervision alone lifts MXD494 AUC from 0.830158 to 0.840191 and AUPR from 0.523015 to 0.565101 versus `GenericTCN`.
- Adding expert gate on top of auxiliary supervision provides a small further MXD494 gain in AUC/AUPR, but the effect is not visible on SL329 or DISORDER723 in seed1.
- Region-aware claims should be framed around ranking positives in difficult datasets and hard-case behavior, not as a universal AUC improvement yet.

## P2.4 Multi-Seed Results

`GenericTCN` and `RegionAwareTCN + aux` were repeated with seeds 1, 2, and 3.

| Variant | Dataset | AUC mean | AUC SD | AUC min-max | AUPR mean | MCC mean | Best seed | Literature best AUC | Mean gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GenericTCN | SL329 | 0.914920 | 0.001967 | 0.912651-0.916141 | 0.916747 | 0.716810 | 1 | 0.915 | -0.000080 |
| GenericTCN | MXD494 | 0.836278 | 0.005301 | 0.830158-0.839384 | 0.541100 | 0.490145 | 2 | 0.842 | -0.005722 |
| GenericTCN | DISORDER723 | 0.919118 | 0.001284 | 0.917828-0.920395 | 0.609868 | 0.553643 | 3 | 0.943 | -0.023882 |
| RegionAwareTCN + aux | SL329 | 0.912806 | 0.001513 | 0.911579-0.914497 | 0.912684 | 0.703383 | 3 | 0.915 | -0.002194 |
| RegionAwareTCN + aux | MXD494 | 0.843771 | 0.003868 | 0.840312-0.847948 | 0.594100 | 0.493757 | 3 | 0.842 | +0.001771 |
| RegionAwareTCN + aux | DISORDER723 | 0.920744 | 0.003195 | 0.917155-0.923276 | 0.610806 | 0.552824 | 3 | 0.943 | -0.022256 |

Interpretation:

- MXD494 is the strongest P2 claim so far: `RegionAwareTCN + aux` has a 3-seed mean AUC above the collected FusionEncoder AUC and mean MCC near the target threshold.
- SL329 is competitive but not stable enough for a strong SOTA claim: `GenericTCN` has one seed above 0.915, but the 3-seed mean is effectively tied with the literature value.
- DISORDER723 improves over P1 but remains far below IDP-EDL AUC 0.943; this dataset should drive the next methodological step.

## P2.5 NR25 Results

Each NR25 model was trained only against its matching target test set.

| Variant | Target | NR25 AUC | NR25 AUPR | NR25 MCC | AUC delta vs full seed1 | P1 NR25 AUC | AUC delta vs P1 NR25 |
|---|---|---:|---:|---:|---:|---:|---:|
| GenericTCN | SL329 | 0.913728 | 0.916194 | 0.711858 | -0.002413 | 0.891377 | +0.022351 |
| RegionAwareTCN + aux | SL329 | 0.912394 | 0.913004 | 0.705392 | +0.000052 | 0.891377 | +0.021017 |
| GenericTCN | MXD494 | 0.827122 | 0.552129 | 0.482370 | -0.003036 | 0.805165 | +0.021957 |
| RegionAwareTCN + aux | MXD494 | 0.831274 | 0.554917 | 0.484029 | -0.009038 | 0.805165 | +0.026109 |
| GenericTCN | DISORDER723 | 0.912272 | 0.599786 | 0.544892 | -0.005556 | 0.898145 | +0.014127 |
| RegionAwareTCN + aux | DISORDER723 | 0.913417 | 0.576808 | 0.544381 | -0.003738 | 0.898145 | +0.015272 |

Interpretation:

- NR25 performance does not collapse. AUC drops versus full-train seed1 are small, usually within about 0.003-0.009.
- All P2 NR25 variants beat the P1 NR25 shallow ESM2 baseline by AUC and MCC on their target test sets.
- MXD494 remains the best high-impact angle: even after NR25 filtering, the region-aware model keeps the best P2 NR25 AUC/MCC for that dataset.

## P2.6 Seed-Ensemble Results

Three score-averaging ensembles were evaluated. Thresholds were selected on the corresponding DM1229 validation ensemble.

| Ensemble | Dataset | AUC | AUPR | MCC | BACC | Literature best AUC | Gap |
|---|---|---:|---:|---:|---:|---:|---:|
| GenericTCN 3-seed | SL329 | 0.916804 | 0.919803 | 0.718277 | 0.849106 | 0.915 | +0.001804 |
| GenericTCN 3-seed | MXD494 | 0.839586 | 0.547817 | 0.487135 | 0.769458 | 0.842 | -0.002414 |
| GenericTCN 3-seed | DISORDER723 | 0.921224 | 0.619473 | 0.556987 | 0.788714 | 0.943 | -0.021776 |
| RegionAwareTCN + aux 3-seed | SL329 | 0.915272 | 0.916345 | 0.705418 | 0.844005 | 0.915 | +0.000272 |
| RegionAwareTCN + aux 3-seed | MXD494 | 0.845856 | 0.600208 | 0.497426 | 0.775607 | 0.842 | +0.003856 |
| RegionAwareTCN + aux 3-seed | DISORDER723 | 0.923132 | 0.625509 | 0.559496 | 0.789425 | 0.943 | -0.019868 |
| 6-model TCN ensemble | SL329 | 0.916941 | 0.918902 | 0.713501 | 0.847167 | 0.915 | +0.001941 |
| 6-model TCN ensemble | MXD494 | 0.845034 | 0.593451 | 0.492306 | 0.772545 | 0.842 | +0.003034 |
| 6-model TCN ensemble | DISORDER723 | 0.922928 | 0.625439 | 0.558295 | 0.789355 | 0.943 | -0.020072 |

Interpretation:

- The seed ensemble strengthens the performance claim on MXD494: `RegionAwareTCN + aux 3-seed` reaches AUC 0.845856, AUPR 0.600208, MCC 0.497426, exceeding the collected FusionEncoder AUC/MCC targets.
- SL329 is also above the collected literature AUC with ensemble models, but the margin is small. It should be reported as competitive or slightly above, with caution.
- DISORDER723 improves further but remains below IDP-EDL AUC 0.943. This remains the main blocker for a broad SOTA claim.

## Required Next P2 Runs

Completed:

1. `P2.3`: Auxiliary heads without gated expert delta to isolate multi-task regularization.
2. `P2.4`: Repeat `GenericTCN` and `RegionAwareTCN + aux` with seeds 2 and 3.
3. `P2.5`: Train NR25-filtered variants against SL329, MXD494, and DISORDER723.
4. `P2.6`: Seed-ensemble the strongest P2 candidates.

Remaining:

1. `P3`: Add probability calibration and uncertainty analysis after final model selection.
2. DISORDER723-specific improvement work if the manuscript needs a broad SOTA claim rather than a strong MXD494-focused contribution.
