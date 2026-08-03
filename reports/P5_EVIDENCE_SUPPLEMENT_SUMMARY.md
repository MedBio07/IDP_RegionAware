# P5 Evidence Supplement Summary

Date: 2026-08-03

## Scope

This P5 evidence bundle adds the three manuscript-level checks requested after P4.6:

1. Protein-level bootstrap confidence intervals and paired permutation tests for the ESM2-t33 upgrade over the ESM2-t12 RegionAwareTCN ensemble.
2. t33 Platt-calibrated uncertainty-error enrichment and hard-case stratified figures.
3. A paper-level ablation table covering representation, architecture, calibration, and NR25 robustness.

## 1. Statistical Support for the Representation Upgrade

The statistical test is paired at the protein level for t33 versus t12 because both models predict the same benchmark proteins. External SOTA methods provide only reported aggregate metrics, so they can only be compared against the t33 bootstrap confidence interval, not by a paired test.

| dataset | reference_auc | candidate_auc | auc_delta | auc_delta_ci_low | auc_delta_ci_high | paired_permutation_p_one_sided | sota_auc | candidate_gap_vs_sota | candidate_auc_ci_low_gt_sota |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SL329 | 0.915272 | 0.919327 | 0.004055 | -0.002693 | 0.011198 | 0.169661 | 0.915000 | 0.004327 | 0 |
| MXD494 | 0.845856 | 0.850637 | 0.004781 | -0.003227 | 0.012471 | 0.177645 | 0.842000 | 0.008637 | 0 |
| DISORDER723 | 0.923132 | 0.944611 | 0.021479 | 0.013868 | 0.028520 | 0.001996 | 0.943000 | 0.001611 | 0 |

Interpretation:

- The ESM2-t33 upgrade has positive AUC point deltas on all three datasets.
- The protein-level statistical support is strong on DISORDER723 only; SL329 and MXD494 are positive but their protein-level bootstrap CIs cross zero.
- The DISORDER723 gain is the strongest mechanistic result because it directly closes the P4.5 failure gap.
- Against aggregate literature SOTA, the point estimates are above SOTA on all three datasets, but bootstrap lower bounds do not exceed the SOTA scalar; this should be phrased as AUC-level SOTA point performance, not statistical dominance over unavailable external predictions.

## 2. t33 Uncertainty and Hard-Case Evidence

Top 10% Platt uncertainty enrichment:

| dataset | overall_error_rate | top_uncertain_error_rate | error_enrichment | mean_uncertainty_top | mean_uncertainty_all |
| --- | --- | --- | --- | --- | --- |
| DM1229_Validation | 0.063254 | 0.354020 | 5.596795 | 0.859785 | 0.239919 |
| SL329 | 0.130642 | 0.335205 | 2.565826 | 0.941887 | 0.412931 |
| MXD494 | 0.203332 | 0.511170 | 2.513963 | 0.935010 | 0.373414 |
| DISORDER723 | 0.047173 | 0.328114 | 6.955547 | 0.836643 | 0.225927 |

Generated figures:

- `figures/p5/P5_T33_PLATT_UNCERTAINTY_ERROR_ENRICHMENT.pdf`
- `figures/p5/P5_T33_HARD_CASE_STRATIFIED_AUC.pdf`

## 3. Paper-Level Ablation Conclusions

Representation deltas:

| dataset | auc_delta | aupr_delta | mcc_delta | interpretation |
| --- | --- | --- | --- | --- |
| SL329 | 0.004055 | 0.002627 | 0.031063 | ESM2-t33 is the decisive ranking-performance upgrade, especially on DISORDER723. |
| MXD494 | 0.004781 | 0.004664 | 0.002013 | ESM2-t33 is the decisive ranking-performance upgrade, especially on DISORDER723. |
| DISORDER723 | 0.021479 | 0.064740 | 0.050966 | ESM2-t33 is the decisive ranking-performance upgrade, especially on DISORDER723. |

Calibration deltas:

| dataset | auc_delta | ece_delta | brier_delta | interpretation |
| --- | --- | --- | --- | --- |
| SL329 | 0.000000 | -0.009123 | -0.015659 | Platt preserves ranking metrics and improves probability quality. |
| MXD494 | 0.000000 | -0.156019 | -0.072682 | Platt preserves ranking metrics and improves probability quality. |
| DISORDER723 | 0.000000 | -0.170328 | -0.061230 | Platt preserves ranking metrics and improves probability quality. |

NR25 deltas:

| dataset | auc_delta | mcc_delta | interpretation |
| --- | --- | --- | --- |
| SL329 | -0.002266 | -0.003297 | NR25 remains competitive but is not uniformly SOTA. |
| MXD494 | -0.016548 | -0.015936 | NR25 remains competitive but is not uniformly SOTA. |
| DISORDER723 | -0.007736 | -0.009357 | NR25 remains competitive but is not uniformly SOTA. |

Generated tables:

- `results/p5/P5_T33_AUC_BOOTSTRAP_CI.tsv`
- `results/p5/P5_T33_PLATT_UNCERTAINTY_ERROR_ENRICHMENT.tsv`
- `results/p5/P5_T33_HARD_CASE_STRATIFIED_SUMMARY.tsv`
- `results/p5/P5_PAPER_LEVEL_ABLATION_TABLE.tsv`
- `results/p5/P5_PAPER_LEVEL_KEY_DELTAS.tsv`

Generated mechanism figure:

- `figures/p5/P5_EVIDENCE_CHAIN_MECHANISM.pdf`

## Decision

The added evidence strengthens P5. The safest manuscript framing is:

Sequence-only RegionAwareTCN with ESM2-t33 achieves AUC-level SOTA point performance in the full DM3000 benchmark, provides strong calibrated probability estimates, and remains competitive under NR25 low-homology evaluation. The manuscript should explicitly state that NR25 performance is not uniformly SOTA and that DISORDER723 MCC/internal IDR classification remain the main limitations.
