# Results Narrative Draft

## Full Benchmark Performance

The ESM2-t33 RegionAwareTCN three-seed ensemble reached AUC values of 0.919327, 0.850637 and 0.944611 on SL329, MXD494 and DISORDER723, respectively. These point estimates exceed the currently curated direct SOTA AUC values for all three external benchmarks. The performance claim should be framed as AUC-level SOTA point performance because the strongest external competitors do not provide paired residue-level predictions for direct statistical testing.

## Representation Upgrade

Replacing ESM2-t12 with ESM2-t33 increased AUC on all three external datasets. The gain was modest on SL329 (+0.004055) and MXD494 (+0.004781), but large on DISORDER723 (+0.021479). Protein-level paired resampling supports the DISORDER723 improvement strongly, with a 95% bootstrap CI of 0.013868 to 0.028520 and paired permutation p=0.001996.

## Low-Homology Evaluation

Target-specific NR25 training reduced performance on all three benchmarks relative to the full DM3000 setting. The drop was small on SL329 (-0.002266 AUC), larger on MXD494 (-0.016548), and moderate on DISORDER723 (-0.007736). Thus, the model remains competitive under low-homology evaluation but should not be described as uniformly low-homology SOTA.

## Hard-Case Stratification

The t33 ensemble maintains strong terminal IDR performance but internal IDRs remain difficult. On DISORDER723, terminal IDR AUC was 0.968186 whereas internal IDR AUC was 0.887416 and internal MCC was 0.210033. This supports the manuscript's hard-case narrative: representation upgrade alleviates but does not fully solve internal IDR prediction.

## Calibration and Uncertainty

Platt calibration preserved AUC, AUPR and MCC while reducing probability error. On DISORDER723, ECE dropped from 0.187936 to 0.017608 and Brier score from 0.093709 to 0.032479. Calibrated uncertainty tracked errors: the top 10% most uncertain residues were enriched for errors by 2.57x, 2.51x and 6.96x on SL329, MXD494 and DISORDER723.
