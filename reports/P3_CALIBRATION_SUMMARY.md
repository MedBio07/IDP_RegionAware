# P3 Calibration and Uncertainty Summary

Date: 2026-07-31

## Scope

P3 was run as post-hoc calibration on the current P2 ensemble candidates:

- `p2_generic_tcn_3seed_ensemble`
- `p2_region_aware_tcn_3seed_ensemble`
- `p2_tcn_6model_ensemble`

Calibration was fitted only on `data/DM1229_Validation.fasta` and then applied to SL329, MXD494, and DISORDER723. Test labels were used only for final evaluation.

Methods:

- `raw`: uncalibrated score baseline.
- `temperature`: one-parameter logit temperature scaling.
- `platt`: logistic calibration on the raw logit score.
- `isotonic`: non-parametric monotonic calibration.

Primary recommendation: use `platt` as the main calibrated probability output. It preserves ranking metrics and gives robust external calibration improvements. Use `isotonic` as a supplement because it sometimes gives slightly lower ECE/Brier, but it can overfit validation calibration and may introduce score ties.

## New P3 Assets

- `scripts/calibrate_disorder_predictions.py`
- `results/calibration/calibration_metrics.tsv`
- `results/calibration/P3_CALIBRATION_COMPARISON.tsv`
- `results/calibration/P3_BEST_CALIBRATION_BY_DATASET.tsv`
- `results/calibration/P3_SELECTED_PLATT_RESULTS.tsv`
- `results/calibration/reliability_bins.tsv`
- `results/calibration/uncertainty_error_enrichment.tsv`
- `results/calibration/unknown_neighbor_uncertainty.tsv`
- `predictions/calibration/`
- `figures/calibration/reliability_*.pdf`
- `figures/calibration/reliability_selected_region_aware_tcn_ensemble.pdf`
- `figures/calibration/uncertainty_error_enrichment.pdf`

## Selected Calibration Results

Platt calibration keeps AUC/AUPR/MCC unchanged because it is monotonic, while substantially improving probability quality.

| Model | Dataset | AUC | AUPR | MCC | Raw Brier | Platt Brier | Raw ECE | Platt ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GenericTCN 3-seed | SL329 | 0.916804 | 0.919803 | 0.718277 | 0.121789 | 0.114493 | 0.101914 | 0.087654 |
| GenericTCN 3-seed | MXD494 | 0.839586 | 0.547817 | 0.487135 | 0.210511 | 0.157174 | 0.244085 | 0.113844 |
| GenericTCN 3-seed | DISORDER723 | 0.921224 | 0.619473 | 0.556987 | 0.093222 | 0.036092 | 0.191864 | 0.018107 |
| RegionAwareTCN + aux 3-seed | SL329 | 0.915272 | 0.916345 | 0.705418 | 0.120225 | 0.115747 | 0.088591 | 0.083598 |
| RegionAwareTCN + aux 3-seed | MXD494 | 0.845856 | 0.600208 | 0.497426 | 0.199652 | 0.154194 | 0.225297 | 0.107488 |
| RegionAwareTCN + aux 3-seed | DISORDER723 | 0.923132 | 0.625509 | 0.559496 | 0.082109 | 0.035833 | 0.168850 | 0.018852 |
| 6-model TCN ensemble | SL329 | 0.916941 | 0.918902 | 0.713501 | 0.120105 | 0.114584 | 0.095253 | 0.085988 |
| 6-model TCN ensemble | MXD494 | 0.845034 | 0.593451 | 0.492306 | 0.204286 | 0.155064 | 0.234691 | 0.109817 |
| 6-model TCN ensemble | DISORDER723 | 0.922928 | 0.625439 | 0.558295 | 0.087113 | 0.035793 | 0.180357 | 0.018399 |

Key interpretation:

- MXD494 remains the strongest performance claim: RegionAwareTCN + aux 3-seed gives AUC 0.845856, AUPR 0.600208, MCC 0.497426, and Platt reduces ECE from 0.225297 to 0.107488.
- DISORDER723 probability calibration is very strong even though ranking performance remains below IDP-EDL. For RegionAwareTCN + aux 3-seed, ECE drops from 0.168850 to 0.018852 and Brier drops from 0.082109 to 0.035833.
- SL329 already has comparatively better raw calibration; Platt still improves ECE/Brier modestly.

## Method Comparison

Temperature scaling was not enough for this task. The fitted temperatures were below 1, which sharpens predictions but cannot correct the main bias caused by disorder prevalence differences between validation and tests.

Platt scaling is the best main-text choice:

- It preserves AUC/AUPR because it is monotonic.
- It corrects both slope and intercept.
- It avoids isotonic's validation overfitting risk.

Isotonic regression can be reported in supplementary results:

- It often gives the best Brier/ECE on MXD494.
- It occasionally reduces AUC/AUPR slightly because it creates tied calibrated scores.
- Validation ECE can become artificially perfect, so it should not be the only calibration claim.

## Uncertainty Findings

For the selected candidate `p2_region_aware_tcn_3seed_ensemble` with Platt calibration, residues in the top 10% uncertainty group are enriched for errors:

| Dataset | Overall error rate | Top 10% uncertain error rate | Error enrichment |
|---|---:|---:|---:|
| DM1229_Validation | 0.070291 | 0.370729 | 5.274224 |
| SL329 | 0.145306 | 0.385073 | 2.650084 |
| MXD494 | 0.203108 | 0.547809 | 2.697128 |
| DISORDER723 | 0.053775 | 0.357292 | 6.644172 |

This supports an uncertainty-aware use case: the calibrated probability can flag residues likely to need caution or manual review.

## Unknown Label Neighborhood

Only SL329 has a large number of `-1` labels in the current test sets. For `p2_region_aware_tcn_3seed_ensemble` with Platt calibration:

| SL329 group | Residues | Mean uncertainty | Known error rate |
|---|---:|---:|---:|
| Unknown residues | 89582 | 0.405332 | NA |
| Known within 5 aa of unknown labels | 2857 | 0.568334 | 0.290515 |
| Known far from unknown labels | 87979 | 0.367189 | 0.140590 |

Known residues adjacent to unknown-label segments are both more uncertain and more error-prone. This is useful manuscript evidence that uncertainty is not only a calibration statistic but also tracks annotation ambiguity.

## P3 Decision

Use `RegionAwareTCN + aux 3-seed ensemble + Platt calibration` as the current main candidate when discussing calibrated probabilities. Keep `6-model ensemble + Platt` as a strong alternative because it has the best SL329 AUC and similar DISORDER723 behavior, but the region-aware ensemble has the clearest MXD494 performance and method narrative.

Remaining before paper-level finalization:

1. Run the final selected model on NR25 with calibration if NR25-calibrated probability tables are needed.
2. Add DISORDER723-specific improvement work if broad SOTA ranking is required.
3. Convert key TSV tables into manuscript figures and statistical summaries.
