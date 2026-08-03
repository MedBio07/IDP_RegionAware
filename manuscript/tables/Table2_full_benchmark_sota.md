# Table 2. Full DM3000 benchmark performance

| dataset | method | sn | sp | bacc | mcc | auc | aupr | fmax | sota_method | sota_auc | auc_gap_vs_sota | t12_auc | t33_minus_t12_auc | delta_ci_95 | paired_permutation_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SL329 | RegionAwareTCN + ESM2-t33, 3-seed ensemble | 0.774150 | 0.942759 | 0.858455 | 0.736481 | 0.919327 | 0.918972 | 0.842101 | IDP-EDL | 0.915000 | 0.004327 | 0.915272 | 0.004055 | -0.002693 to 0.011198 | 0.169661 |
| MXD494 | RegionAwareTCN + ESM2-t33, 3-seed ensemble | 0.742350 | 0.812379 | 0.777365 | 0.499439 | 0.850637 | 0.604872 | 0.623957 | FusionEncoder | 0.842000 | 0.008637 | 0.845856 | 0.004781 | -0.003227 to 0.012471 | 0.177645 |
| DISORDER723 | RegionAwareTCN + ESM2-t33, 3-seed ensemble | 0.654000 | 0.972866 | 0.813433 | 0.610462 | 0.944611 | 0.690249 | 0.639095 | IDP-EDL | 0.943000 | 0.001611 | 0.923132 | 0.021479 | 0.013868 to 0.028520 | 0.001996 |

Note: protein-level paired statistics compare t33 against local t12 predictions, not against external aggregate SOTA methods.
