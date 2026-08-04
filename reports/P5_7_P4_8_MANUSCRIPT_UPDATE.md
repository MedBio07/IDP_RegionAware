# P5.7 P4.8 Manuscript Update

Date: 2026-08-04

## Main Replacement

The manuscript main model should now be the P4.8 warm-start `RegionAdapterMoETCN` ensemble with validation-fitted Platt calibration. P4.6 remains the warm-start backbone and strongest architecture/representation baseline.

## P4.8 Full Benchmark Table

| dataset | method | sn | sp | bacc | mcc | auc | aupr | fmax | sota_method | sota_auc | auc_gap_vs_sota | p4_8_minus_p4_6_auc | p4_8_minus_p4_6_mcc | auc_delta_ci_95 | paired_permutation_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SL329 | RegionAdapterMoETCN warm-start + Platt | 0.780902 | 0.935526 | 0.858214 | 0.733181 | 0.920545 | 0.920569 | 0.841076 | IDP-EDL | 0.915000 | 0.005545 | 0.001218 | -0.003300 | 0.000298 to 0.002092 | 0.253493 |
| MXD494 | RegionAdapterMoETCN warm-start + Platt | 0.752104 | 0.807806 | 0.779955 | 0.501629 | 0.850471 | 0.602829 | 0.624979 | FusionEncoder | 0.842000 | 0.008471 | -0.000166 | 0.002190 | -0.001384 to 0.001085 | 0.497006 |
| DISORDER723 | RegionAdapterMoETCN warm-start + Platt | 0.659175 | 0.972673 | 0.815924 | 0.613151 | 0.945409 | 0.693164 | 0.642751 | IDP-EDL | 0.943000 | 0.002409 | 0.000798 | 0.002689 | 0.000015 to 0.001502 | 0.321357 |

## P4.8 NR25 Table

| dataset | removed_train_proteins | kept_train_proteins | full_auc | nr25_auc | nr25_minus_full_auc | nr25_aupr | full_mcc | nr25_mcc | nr25_minus_full_mcc | p4_8_minus_p4_6_nr25_auc | sota_auc | nr25_gap_vs_sota |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SL329 | 176 | 2824 | 0.920545 | 0.918557 | -0.001988 | 0.919279 | 0.733181 | 0.733606 | 0.000425 | 0.001496 | 0.915000 | 0.003557 |
| MXD494 | 323 | 2677 | 0.850471 | 0.833695 | -0.016776 | 0.590812 | 0.501629 | 0.483823 | -0.017806 | -0.000394 | 0.842000 | -0.008305 |
| DISORDER723 | 424 | 2576 | 0.945409 | 0.938194 | -0.007215 | 0.651359 | 0.613151 | 0.602456 | -0.010695 | 0.001319 | 0.943000 | -0.004806 |

## P4.8 Hard-Case Focus

| dataset | stratum | reference_auc | candidate_auc | auc_delta | reference_aupr | candidate_aupr | aupr_delta | reference_mcc | candidate_mcc | mcc_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SL329 | Internal IDRs | 0.888676 | 0.891854 | 0.003178 | 0.733146 | 0.741248 | 0.008102 | 0.617910 | 0.614205 | -0.003705 |
| MXD494 | Internal IDRs | 0.796259 | 0.799026 | 0.002767 | 0.149714 | 0.150889 | 0.001175 | 0.228756 | 0.234204 | 0.005448 |
| DISORDER723 | Long disorder regions | 0.933028 | 0.933295 | 0.000267 | 0.498690 | 0.511333 | 0.012643 | 0.388454 | 0.393066 | 0.004612 |
| DISORDER723 | Internal IDRs | 0.887416 | 0.894658 | 0.007242 | 0.152541 | 0.171858 | 0.019317 | 0.210033 | 0.237379 | 0.027346 |
| DISORDER723 | Middle residues | 0.916962 | 0.918476 | 0.001514 | 0.443969 | 0.454825 | 0.010856 | 0.422864 | 0.429908 | 0.007044 |

## P4.8 Calibration and Uncertainty

| dataset | auc | aupr | mcc | raw_ece | platt_ece | ece_delta | raw_brier | platt_brier | brier_delta | top10_uncertain_error_enrichment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SL329 | 0.920545 | 0.920569 | 0.733181 | 0.108403 | 0.087562 | -0.020841 | 0.125130 | 0.108405 | -0.016725 | 2.730646 |
| MXD494 | 0.850471 | 0.602829 | 0.501629 | 0.241887 | 0.108556 | -0.133331 | 0.218987 | 0.152707 | -0.066280 | 2.524622 |
| DISORDER723 | 0.945409 | 0.693164 | 0.613151 | 0.159805 | 0.016327 | -0.143478 | 0.085060 | 0.032010 | -0.053050 | 7.028223 |

## Recommended Claim Boundary

The safest claim is that warm-start region-specialized adapters preserve full-benchmark AUC, provide modest but consistent internal-IDR improvements, and maintain target-specific NR25 robustness relative to P4.6. The strongest direct evidence is DISORDER723 internal IDR improvement: AUC +0.007242, AUPR +0.019317 and MCC +0.027346. The paper should not overstate aggregate performance deltas, because MXD494 full-benchmark AUC changes by only -0.000166 and paired permutation p-values do not indicate strong global dominance.
