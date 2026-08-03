<!--
P5.4 hardened draft.
Still not submission-final until authors/affiliations, funding, conflict-of-interest,
repository DOI, license, exact hardware/runtime and formal references are inserted.
-->

# RegionAwareTCN: sequence-only, region-aware and uncertainty-calibrated intrinsic disorder prediction with protein language model representations

## Article Type

Original Paper

## Running Title

Calibrated sequence-only IDR prediction

## Authors

To be filled.

## Corresponding Author

To be filled.

## Abstract

Intrinsic disorder prediction has benefited from protein language models, but benchmark gains can be difficult to interpret because disorder labels are regionally heterogeneous, test sets differ in prevalence, and homologous training examples may inflate apparent performance. We developed RegionAwareTCN, a sequence-only residue-level predictor that combines frozen ESM2 representations with temporal convolutional sequence modeling and auxiliary supervision for short/long and terminal/internal disorder patterns. The final model uses ESM2-t33 embeddings, three-seed score averaging, validation-selected thresholds, and Platt calibration fitted only on the DM1229 validation set. Across SL329, MXD494 and DISORDER723, the ESM2-t33 RegionAwareTCN ensemble achieved AUC values of 0.919327, 0.850637 and 0.944611, respectively, exceeding the curated direct state-of-the-art point AUC values for all three benchmarks under the full DM3000 training protocol. Protein-level paired resampling showed that the ESM2-t33 upgrade produced the strongest statistically supported gain on DISORDER723, improving AUC by 0.021479 over the ESM2-t12 RegionAwareTCN ensemble (95% bootstrap CI 0.013868 to 0.028520; paired permutation p=0.001996). Target-specific NR25 evaluation showed competitive but not uniformly state-of-the-art low-homology performance. Platt calibration preserved ranking metrics while reducing DISORDER723 ECE from 0.187936 to 0.017608 and Brier score from 0.093709 to 0.032479. These results support a calibrated, leakage-aware and hard-case-stratified sequence-only framework for intrinsic disorder prediction.

## Keywords

intrinsic disorder prediction; intrinsically disordered regions; protein language model; ESM2; calibration; uncertainty; low-homology evaluation

## Introduction

Intrinsically disordered proteins and intrinsically disordered regions (IDRs) are central to regulation, signaling and molecular recognition, but their residue-level prediction remains difficult. Disorder is not a single homogeneous state. Disordered residues may occur in short or long segments, at protein termini or in internal regions, and in proteins with very different overall disorder content. Benchmark labels also vary in origin and certainty, including DisProt-like annotations, missing-residue-derived annotations, NoX-style labels and unknown residues. These factors make global benchmark metrics useful but incomplete.

Recent IDR predictors increasingly use protein language models (PLMs), feature fusion, ensembles and length-aware or region-aware predictors. IDP-EDL and FusionEncoder define strong direct comparators on SL329, MXD494 and DISORDER723, while CAID-style evaluations emphasize ranking metrics, threshold-dependent metrics, low-similarity subsets and functional extensions. However, three gaps remain important for a method paper. First, full-training benchmark gains can be misleading without explicit low-homology controls. Second, internal and terminal IDRs can have different error profiles that are hidden by aggregate AUC. Third, most predictors report raw scores rather than calibrated probabilities, limiting their practical use for experimental prioritization.

We present RegionAwareTCN, a sequence-only residue-level IDR prediction framework that combines frozen ESM2 representations, temporal convolutional sequence modeling and auxiliary supervision for SDR/LDR and terminal/internal disorder patterns. The selected final model uses ESM2-t33-650M residue embeddings, one-hot and relative-position features, a three-seed score-averaged ensemble and validation-fitted Platt calibration. We keep the main model sequence-only and avoid PDB coordinates, experimentally missing residues, multiple-sequence alignments, profile features, AlphaFold confidence features and function labels as primary inputs. This design reduces leakage risk and simplifies deployment.

The study makes five contributions. First, it evaluates a sequence-only PLM-based RegionAwareTCN under a unified train/validation/test protocol. Second, it separates full DM3000 benchmark performance from target-specific NR25 low-homology training. Third, it uses error-analysis-driven representation upgrading to show that ESM2-t33 closes the DISORDER723 ranking gap that smoothing and loss reweighting did not solve. Fourth, it reports hard-case stratification across SDR/LDR, terminal/internal IDRs, residue zones and protein-level strata. Fifth, it evaluates calibrated probability quality and uncertainty-error enrichment.

## Materials and Methods

### Datasets and label handling

The study used DM3000 as the training set and DM1229 as the validation set. Three independent external benchmarks were used for final testing: SL329, MXD494 and DISORDER723 (Table 1). Labels were treated at residue level with three states: ordered, disordered and unknown. Unknown residues were excluded from all metric calculations, which primarily affects SL329. Thresholds, calibration parameters and model choices were selected on DM1229 and fixed before test-set evaluation.

The three external benchmarks differ substantially in disorder prevalence. Among known residues, the disorder fraction is 0.435334 for SL329, 0.224360 for MXD494 and 0.062845 for DISORDER723. Because DISORDER723 is highly imbalanced, AUPR, MCC, Fmax and calibration metrics are reported alongside ROC-AUC.

### NR25 leakage-control evaluation

Target-specific NR25 training sets were generated by removing DM3000 training proteins with MMseqs2 percentage identity above 25% against each target benchmark. The resulting training sets retained 2,824 proteins for SL329, 2,677 proteins for MXD494 and 2,576 proteins for DISORDER723. Full DM3000 and NR25 results are reported separately to avoid mixing standard benchmark performance with low-homology generalization claims.

### Sequence representation and RegionAwareTCN

Residue-level ESM2 embeddings were extracted and cached. The representation-upgrade experiment replaced ESM2-t12-35M layer 12 with ESM2-t33-650M layer 33 while keeping the RegionAwareTCN head, thresholding protocol and benchmark evaluation fixed. RegionAwareTCN applies temporal convolutional sequence modeling to frozen residue representations, residue one-hot features and relative-position features. It also includes auxiliary supervision for short versus long disordered regions and terminal versus internal disorder patterns. These auxiliary tasks provide a region-aware training and analysis scaffold, rather than a claim of uniform aggregate-metric improvement.

The final predictor averages scores from three independently trained ESM2-t33 RegionAwareTCN models with seeds 1, 2 and 3. The ensemble score is used for ranking metrics, validation-selected thresholding and calibration.

### Training, calibration and evaluation

Models were trained on DM3000 or target-specific NR25 training sets and selected using validation performance on DM1229. The binary decision threshold was selected on DM1229 by maximizing Fmax for the corresponding score distribution. Test labels were never used for threshold tuning or calibration fitting.

Post-hoc calibration was fitted using DM1229 validation predictions. Raw scores, temperature scaling, Platt scaling and isotonic regression were evaluated. Platt scaling was selected for the main calibrated output because it preserved ranking metrics while improving ECE, Brier score and NLL with lower overfitting risk than isotonic regression. Uncertainty was computed from calibrated binary predictive entropy.

Residue-level performance was evaluated with sensitivity (Sn), specificity (Sp), balanced accuracy (BACC), Matthews correlation coefficient (MCC), ROC-AUC, AUPR and Fmax. Calibration quality was evaluated with expected calibration error (ECE), Brier score and negative log-likelihood. Stratified analyses were performed for SDR/LDR, terminal/internal IDRs, residue zones, protein length bins and protein disorder-content bins. ESM2-t33 and ESM2-t12 RegionAwareTCN ensembles were compared with protein-level paired bootstrap confidence intervals and paired permutation tests. External SOTA methods were available only as aggregate published metrics, so they were compared as point references rather than by paired statistical tests.

## Results

### RegionAwareTCN reaches AUC-level SOTA point performance in the full DM3000 setting

Under the full DM3000 training protocol, the ESM2-t33 RegionAwareTCN ensemble reached AUC values of 0.919327, 0.850637 and 0.944611 on SL329, MXD494 and DISORDER723, respectively (Table 2 and Figure 2). These point estimates exceed the curated direct SOTA AUC values for all three benchmarks: IDP-EDL on SL329, FusionEncoder on MXD494 and IDP-EDL on DISORDER723. The corresponding point gains were +0.004327, +0.008637 and +0.001611.

Threshold-dependent metrics give a more restrained interpretation. On SL329, the model achieved MCC 0.736481, exceeding the curated IDP-EDL MCC of 0.700. On MXD494, it achieved MCC 0.499439, slightly above the curated FusionEncoder MCC of 0.492. On DISORDER723, MCC was 0.610462, below the curated IDP-EDL MCC of 0.636. Therefore, the primary performance claim is AUC-level SOTA point performance, not comprehensive dominance across all metrics.

### ESM2-t33 is the main driver of the DISORDER723 gain

Replacing ESM2-t12 with ESM2-t33 improved AUC point estimates on all external datasets. The AUC delta was +0.004055 on SL329, +0.004781 on MXD494 and +0.021479 on DISORDER723 (Table 2 and Supplementary Table S3). Protein-level paired statistics showed strong support for the DISORDER723 improvement, with a 95% bootstrap CI of 0.013868 to 0.028520 and paired permutation p=0.001996. SL329 and MXD494 had positive point deltas, but their protein-level confidence intervals crossed zero and should be interpreted as favorable but not statistically secure under conservative protein-level resampling.

This result connects to the prior failure analysis. Score smoothing, position removal and focal/asymmetric losses did not close the DISORDER723 gap, whereas a stronger frozen PLM representation substantially improved ranking performance. The largest gain occurred on the most difficult benchmark, supporting representation quality as the main lever for this stage of the project.

### NR25 evaluation defines the low-homology boundary

Target-specific NR25 training reduced performance relative to full DM3000 training (Table 3). The AUC drop was small on SL329 (-0.002266), larger on MXD494 (-0.016548) and moderate on DISORDER723 (-0.007736). Under NR25 training, RegionAwareTCN remained above the curated SL329 SOTA AUC but fell below the curated MXD494 and DISORDER723 SOTA AUC values. The appropriate claim is therefore competitive low-homology robustness, not uniformly low-homology SOTA.

### Hard-case stratification identifies internal IDRs as the remaining weakness

Hard-case stratification confirmed that IDR prediction difficulty is region dependent (Figure 3 and Supplementary Table S4). In DISORDER723, terminal IDR AUC was 0.968186, whereas internal IDR AUC was 0.887416 and internal MCC was 0.210033. The same qualitative pattern appears in other datasets, where internal IDRs are consistently harder than terminal IDRs. ESM2-t33 alleviates this failure mode but does not fully solve it.

### Platt calibration improves probability quality and uncertainty tracks errors

Platt calibration preserved AUC, AUPR and MCC while improving probability quality (Table 4). On DISORDER723, ECE decreased from 0.187936 to 0.017608 and Brier score decreased from 0.093709 to 0.032479. On MXD494, ECE decreased from 0.256921 to 0.100902 and Brier score decreased from 0.219616 to 0.146934.

Calibrated uncertainty was informative (Figure 4 and Supplementary Table S5). The top 10% most uncertain residues were enriched for errors by 2.57x on SL329, 2.51x on MXD494 and 6.96x on DISORDER723. This supports uncertainty-aware interpretation, especially for low-prevalence or annotation-sensitive benchmarks.

## Discussion

RegionAwareTCN supports a sequence-only, region-aware and calibrated framing for intrinsic disorder prediction. The final ESM2-t33 ensemble reaches AUC-level SOTA point performance on SL329, MXD494 and DISORDER723 under the full DM3000 protocol. The strongest evidence is not merely the final point estimate but the experimental chain: low-cost post-processing and loss variants did not close the DISORDER723 gap, whereas upgrading the frozen PLM representation substantially improved DISORDER723 ranking and internal-IDR behavior.

The study also clarifies the boundary of the performance claim. Protein-level paired statistics support a strong t33-over-t12 gain on DISORDER723, but the corresponding gains on SL329 and MXD494 are smaller and not significant under conservative protein-level resampling. External SOTA methods do not provide residue-level predictions, so direct paired tests against them are not possible. For this reason, the manuscript claims AUC-level SOTA point performance rather than broad statistical dominance over all existing predictors.

The NR25 experiments are a strength because they separate standard benchmark performance from low-homology generalization. The t33 model remains competitive under NR25 training and remains above the SL329 SOTA AUC, but MXD494 and DISORDER723 fall below the curated SOTA values. This limitation should be stated explicitly. It improves credibility and positions the work as leakage-aware rather than benchmark-optimized.

Calibration is a practical contribution. Platt scaling markedly improves ECE and Brier score, especially on MXD494 and DISORDER723, without changing AUC, AUPR or MCC. Calibrated uncertainty also enriches prediction errors, supporting a caution signal for ambiguous residues, disorder boundaries and low-prevalence datasets.

Internal IDRs remain the major unsolved technical problem. Although ESM2-t33 improved DISORDER723 internal AUC, internal MCC remained low. This suggests that internal disorder may require additional signals, such as function-aware labels, binding-region annotations, MoRF labels or specialized hard-example training. Structure features should not be added casually because local audits found no controlled structure-feature coverage and several benchmarks contain PDB-chain-like identifiers, creating leakage risk.

Overall, the strongest manuscript position is not that a new neural network solves IDR prediction. The stronger contribution is a transparent and calibrated PLM-era benchmark framework that achieves strong full-training performance, exposes low-homology and internal-IDR limitations, and provides probability estimates suitable for uncertainty-aware use.

## Conclusion

RegionAwareTCN provides a sequence-only, region-aware and calibrated framework for residue-level intrinsic disorder prediction. The ESM2-t33 three-seed ensemble achieves AUC-level SOTA point performance on three external benchmarks under the full DM3000 protocol, with the strongest statistically supported representation-upgrade gain on DISORDER723. The model remains competitive but not uniformly SOTA under target-specific NR25 evaluation. Platt calibration substantially improves probability quality and calibrated uncertainty enriches prediction errors. The main remaining challenges are internal IDR classification, DISORDER723 MCC and low-homology generalization on MXD494 and DISORDER723.

## Data Availability

Code, configuration files, trained model weights, validation-fitted calibration parameters, residue-level predictions, benchmark result tables and figure-generation assets will be made available in a public repository and archived with a permanent DOI before publication. The repository URL and archival DOI are to be inserted before submission.

The local reproducibility package contains labeled FASTA datasets under `data/`, target-specific NR25 train-set construction artifacts under `data/nr25_by_test/`, cached ESM2 embeddings under `data/features/esm2_embeddings/`, trained RegionAwareTCN weights under `models/`, predictions under `predictions/`, result tables under `results/`, figures under `figures/`, and manuscript assets under `manuscript/`. The final public release should either include the cached ESM2 embeddings or provide exact commands for regenerating them from sequence FASTA files.

## Funding

To be filled.

## Conflict of Interest

To be filled.

## References

A formal journal-style reference list must be inserted before submission. The P5.4 citation audit file (`manuscript/P5_4_CORE_REFERENCE_AUDIT.md`) provides the core reference set and DOI/URL checks for IDP-EDL, FusionEncoder, IDP-Fusion, CAID, IDP-LM, DR-BERT, DisoFLAG, PUNCH2, PredIDR2, flDPnn3 and ESMDisPred. The final reference list should also include primary ESM2, MMseqs2, DisProt, MobiDB and calibration/statistical-method citations where cited in the text.

## Main Tables

### Table 1. Dataset and NR25 leakage-control summary

| dataset | role | proteins | known_residues | disorder_residues | unknown_residues | disorder_fraction_known | sdr_segments | ldr_segments | terminal_segments | internal_segments | nr25_removed_train_proteins | nr25_kept_train_proteins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DM3000 train | train | 3000 | 730804 | 74170 | 0 | 0.101491 | 5427 | 392 | 4183 | 1636 | NA | NA |
| DM1229 validation | validation | 1229 | 305830 | 29082 | 0 | 0.095092 | 2372 | 166 | 1751 | 787 | NA | NA |
| SL329 test | test | 329 | 90836 | 39544 | 89582 | 0.435334 | 464 | 274 | 328 | 410 | 176 | 2824 |
| MXD494 test | test | 494 | 196501 | 44087 | 0 | 0.224360 | 577 | 271 | 517 | 331 | 323 | 2677 |
| DISORDER723 test | test | 723 | 215229 | 13526 | 0 | 0.062845 | 1363 | 60 | 1017 | 406 | 424 | 2576 |
| NR25 train vs SL329 | nr25_train | 2824 | 686451 | 64364 | 0 | 0.093763 | 5144 | 325 | 3941 | 1528 | 176 | 2824 |
| NR25 train vs MXD494 | nr25_train | 2677 | 639506 | 54830 | 0 | 0.085738 | 4904 | 287 | 3729 | 1462 | 323 | 2677 |
| NR25 train vs DISORDER723 | nr25_train | 2576 | 604856 | 59658 | 0 | 0.098632 | 4584 | 292 | 3568 | 1308 | 424 | 2576 |

Note: SL329 contains unknown labels and all `-1` residues are masked during evaluation.

### Table 2. Full DM3000 benchmark performance

| dataset | method | sn | sp | bacc | mcc | auc | aupr | fmax | sota_method | sota_auc | auc_gap_vs_sota | t12_auc | t33_minus_t12_auc | delta_ci_95 | paired_permutation_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SL329 | RegionAwareTCN + ESM2-t33, 3-seed ensemble | 0.774150 | 0.942759 | 0.858455 | 0.736481 | 0.919327 | 0.918972 | 0.842101 | IDP-EDL | 0.915000 | 0.004327 | 0.915272 | 0.004055 | -0.002693 to 0.011198 | 0.169661 |
| MXD494 | RegionAwareTCN + ESM2-t33, 3-seed ensemble | 0.742350 | 0.812379 | 0.777365 | 0.499439 | 0.850637 | 0.604872 | 0.623957 | FusionEncoder | 0.842000 | 0.008637 | 0.845856 | 0.004781 | -0.003227 to 0.012471 | 0.177645 |
| DISORDER723 | RegionAwareTCN + ESM2-t33, 3-seed ensemble | 0.654000 | 0.972866 | 0.813433 | 0.610462 | 0.944611 | 0.690249 | 0.639095 | IDP-EDL | 0.943000 | 0.001611 | 0.923132 | 0.021479 | 0.013868 to 0.028520 | 0.001996 |

Note: protein-level paired statistics compare t33 against local t12 predictions, not against external aggregate SOTA methods.

### Table 3. Target-specific NR25 low-homology robustness

| dataset | removed_train_proteins | kept_train_proteins | full_auc | nr25_auc | auc_delta_nr25_minus_full | nr25_aupr | full_mcc | nr25_mcc | mcc_delta_nr25_minus_full | sota_auc | nr25_gap_vs_sota |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SL329 | 176 | 2824 | 0.919327 | 0.917061 | -0.002266 | 0.913435 | 0.736481 | 0.733184 | -0.003297 | 0.915000 | 0.002061 |
| MXD494 | 323 | 2677 | 0.850637 | 0.834089 | -0.016548 | 0.585903 | 0.499439 | 0.483503 | -0.015936 | 0.842000 | -0.007911 |
| DISORDER723 | 424 | 2576 | 0.944611 | 0.936875 | -0.007736 | 0.652169 | 0.610462 | 0.601105 | -0.009357 | 0.943000 | -0.006125 |

Note: NR25 results are reported as robustness evidence and are not uniformly SOTA.

### Table 4. Platt calibration and uncertainty-error enrichment

| dataset | auc | aupr | mcc | raw_ece | platt_ece | ece_delta | raw_brier | platt_brier | brier_delta | top10_uncertain_error_enrichment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SL329 | 0.919327 | 0.918972 | 0.736481 | 0.109384 | 0.100261 | -0.009123 | 0.126544 | 0.110885 | -0.015659 | 2.565826 |
| MXD494 | 0.850637 | 0.604872 | 0.499439 | 0.256921 | 0.100902 | -0.156019 | 0.219616 | 0.146934 | -0.072682 | 2.513963 |
| DISORDER723 | 0.944611 | 0.690249 | 0.610462 | 0.187936 | 0.017608 | -0.170328 | 0.093709 | 0.032479 | -0.061230 | 6.955547 |

Note: Platt calibration is fitted only on DM1229 validation predictions.

## Main Figure List

Figure 1. Evidence chain and model framing. Source: `figures/p5/P5_EVIDENCE_CHAIN_MECHANISM.pdf`

Figure 2. ROC and precision-recall curves for the representation upgrade. Source: `figures/p5/P5_T12_T33_ROC_PR_CURVES.pdf`

Figure 3. Hard-case stratified AUC. Source: `figures/p5/P5_T33_HARD_CASE_STRATIFIED_AUC.pdf`

Figure 4. Calibrated uncertainty tracks prediction errors. Source: `figures/p5/P5_T33_PLATT_UNCERTAINTY_ERROR_ENRICHMENT.pdf`
