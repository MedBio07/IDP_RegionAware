# P4.8 Warm-Start Adapter Replacement Decision

Date: 2026-08-04

## Scope

P4.8 evaluates whether the P4.7 warm-start `RegionAdapterMoETCN` ensemble should replace the P4.6 `RegionAwareTCN` ensemble as the manuscript main model. The candidate keeps the same sequence-only ESM2-t33/position/one-hot input setting, warm-starts from matched P4.6 seeds, freezes the shared backbone, and trains region adapters plus MoE gate/expert heads.

## Main Platt-Calibrated Benchmark

| dataset | auc | aupr | mcc | fmax | ece | brier | delta_auc_vs_p4_6_platt | delta_aupr_vs_p4_6_platt | delta_mcc_vs_p4_6_platt | delta_ece_vs_p4_6_platt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SL329 | 0.920545 | 0.920569 | 0.733181 | 0.841076 | 0.087562 | 0.108405 | 0.001218 | 0.001597 | -0.003300 | -0.012699 |
| MXD494 | 0.850471 | 0.602829 | 0.501629 | 0.624979 | 0.108556 | 0.152707 | -0.000166 | -0.002043 | 0.002190 | 0.007654 |
| DISORDER723 | 0.945409 | 0.693164 | 0.613151 | 0.642751 | 0.016327 | 0.032010 | 0.000798 | 0.002915 | 0.002689 | -0.001281 |

## Paired Protein-Level AUC Test

| dataset | reference_auc | candidate_auc | auc_delta | auc_delta_ci_low | auc_delta_ci_high | paired_bootstrap_p_delta_le_0 | paired_permutation_p_one_sided | n_bootstrap | n_permutation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SL329 | 0.919327 | 0.920545 | 0.001218 | 0.000298 | 0.002092 | 0.005994 | 0.253493 | 1000 | 500 |
| MXD494 | 0.850637 | 0.850471 | -0.000166 | -0.001384 | 0.001085 | 0.578422 | 0.497006 | 1000 | 500 |
| DISORDER723 | 0.944611 | 0.945409 | 0.000799 | 0.000015 | 0.001502 | 0.025974 | 0.321357 | 1000 | 500 |

## Platt Probability Quality

| dataset | brier | nll | ece | delta_brier_vs_p4_6_platt | delta_nll_vs_p4_6_platt | delta_ece_vs_p4_6_platt |
| --- | --- | --- | --- | --- | --- | --- |
| DM1229_Validation | 0.045983 | 0.164833 | 0.002883 | NA | NA | NA |
| SL329 | 0.108405 | 0.389415 | 0.087562 | -0.002480 | -0.012551 | -0.012699 |
| MXD494 | 0.152707 | 0.486333 | 0.108556 | 0.005773 | 0.028618 | 0.007654 |
| DISORDER723 | 0.032010 | 0.118956 | 0.016327 | -0.000469 | -0.001311 | -0.001281 |

## DISORDER723 Hard Cases

| display_stratum | reference_auc | candidate_auc | auc_delta | reference_aupr | candidate_aupr | aupr_delta | reference_mcc | candidate_mcc | mcc_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Internal IDRs | 0.887416 | 0.894658 | 0.007242 | 0.152541 | 0.171858 | 0.019317 | 0.210033 | 0.237379 | 0.027346 |
| Middle residues | 0.916962 | 0.918476 | 0.001514 | 0.443969 | 0.454825 | 0.010856 | 0.422864 | 0.429908 | 0.007044 |

## NR25 Check

| dataset | reference_auc | candidate_auc | auc_delta | reference_mcc | candidate_mcc | mcc_delta |
| --- | --- | --- | --- | --- | --- | --- |
| SL329 | 0.917061 | 0.918557 | 0.001496 | 0.733184 | 0.733606 | 0.000422 |
| MXD494 | 0.834089 | 0.833695 | -0.000394 | 0.483503 | 0.483823 | 0.000320 |
| DISORDER723 | 0.936875 | 0.938194 | 0.001319 | 0.601105 | 0.602456 | 0.001351 |

## Decision Criteria

| criterion | status | evidence |
| --- | --- | --- |
| external_auc_preservation | pass | Minimum Platt AUC delta vs P4.6 across external datasets is -0.000166. |
| external_mcc_balance | pass | MCC deltas vs P4.6 Platt are -0.003300, 0.002190, 0.002689. |
| platt_probability_quality | mixed | Mean ECE delta is -0.002109; mean Brier delta is 0.000941. |
| disorder723_internal_idr_gain | pass | DISORDER723 internal-IDR AUC delta is 0.007242; MCC delta is 0.027346. |
| nr25_low_homology_check | pass | Minimum NR25 AUC delta vs P4.6 NR25 is -0.000394. |

## Decision

Replace P4.6 with the P4.8 warm-start RegionAdapterMoETCN as the manuscript main method, while retaining P4.6 as the warm-start backbone and strongest ablation baseline.

## Manuscript Claim Boundary

The replacement, if used, should be framed conservatively: aggregate AUC gains are small and protein-level confidence intervals are expected to be wider than the point deltas, whereas the strongest direct mechanistic signal is improved DISORDER723 internal-IDR performance. The paper should describe P4.8 as a sequence-only region-adapter specialization over the P4.6 RegionAwareTCN backbone, not as a wholesale representation change.
