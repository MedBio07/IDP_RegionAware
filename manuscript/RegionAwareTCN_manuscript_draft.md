# RegionAwareTCN: a sequence-only, region-aware and uncertainty-calibrated framework for intrinsic disorder prediction

## Manuscript Status

Draft stage: P5.2 integrated English draft.

Claim boundary: this draft intentionally uses conservative language. It claims AUC-level state-of-the-art point performance under the full DM3000 training protocol, strong protein-level support for the ESM2-t33 upgrade on DISORDER723, and competitive but not uniformly state-of-the-art NR25 low-homology performance.

## Authors

To be filled.

## Abstract

Intrinsic disorder prediction has benefited from protein language models, but benchmark-level gains can be difficult to interpret because disorder annotations are regionally heterogeneous, test sets differ in disorder prevalence, and homologous training examples may inflate apparent performance. We developed RegionAwareTCN, a sequence-only residue-level predictor that combines frozen ESM2 representations with temporal convolutional sequence modeling and auxiliary supervision for short/long and terminal/internal disorder patterns. The final model uses ESM2-t33 embeddings, three-seed score averaging, validation-selected thresholds, and Platt calibration fitted only on the DM1229 validation set.

Across SL329, MXD494 and DISORDER723, the ESM2-t33 RegionAwareTCN ensemble achieved AUC values of 0.919327, 0.850637 and 0.944611, respectively, exceeding the curated direct SOTA point AUC values for all three benchmarks under the full DM3000 training protocol. Protein-level paired resampling showed that the ESM2-t33 upgrade produced the strongest statistically supported gain on DISORDER723, improving AUC by 0.021479 over the ESM2-t12 RegionAwareTCN ensemble (95% bootstrap CI 0.013868 to 0.028520; paired permutation p=0.001996). Target-specific NR25 evaluation showed competitive but not uniformly SOTA low-homology performance, with AUC values of 0.917061, 0.834089 and 0.936875 on SL329, MXD494 and DISORDER723. Platt calibration preserved ranking metrics while improving probability quality, reducing DISORDER723 ECE from 0.187936 to 0.017608 and Brier score from 0.093709 to 0.032479. Calibrated uncertainty enriched errors among the top 10% most uncertain residues by 2.57x, 2.51x and 6.96x on SL329, MXD494 and DISORDER723.

These results support a calibrated, leakage-aware and hard-case-stratified sequence-only framework for intrinsic disorder prediction, while also identifying internal IDRs and low-homology generalization as the major remaining limitations.

## Keywords

intrinsic disorder prediction; intrinsically disordered regions; protein language model; ESM2; calibration; uncertainty; low-homology evaluation; RegionAwareTCN

## Introduction

Intrinsically disordered proteins and intrinsically disordered regions (IDRs) are central to many regulatory, signaling and interaction processes, yet they remain challenging for residue-level prediction. Unlike structured domains, disorder is not a single homogeneous state. Disordered residues may occur in short or long segments, at protein termini or in internal regions, and in proteins with very different overall disorder content. Benchmark labels also differ in origin and certainty, including DisProt-like annotations, missing-residue-derived annotations, NoX-style labels and unknown residues. These factors make global benchmark metrics informative but incomplete.

Recent IDR predictors increasingly rely on protein language models (PLMs), feature fusion, ensemble learning and region-aware training strategies. Methods such as IDP-EDL and FusionEncoder define strong direct competitors on the SL329, MXD494 and DISORDER723 benchmarks, while CAID-style evaluations emphasize that disorder prediction should be assessed with ranking metrics, threshold-dependent metrics, low-similarity subsets and functional/binding extensions. However, several issues remain underreported in many benchmark studies. First, full-training benchmark gains may be difficult to interpret without a low-homology leakage-control setting. Second, region-specific failure modes, especially terminal versus internal IDRs, can be hidden by aggregate AUC. Third, model scores are usually treated as ranking values rather than calibrated probabilities, limiting their practical use for experimental prioritization and manual review.

Here we present RegionAwareTCN, a sequence-only residue-level IDR prediction framework built around frozen ESM2 representations, temporal convolutional sequence modeling and auxiliary supervision for SDR/LDR and terminal/internal disorder patterns. The final model uses ESM2-t33-650M residue embeddings, one-hot and relative-position features, a three-seed score-averaged ensemble and validation-fitted Platt calibration. We deliberately keep the main model sequence-only and avoid PDB coordinates, experimentally missing residues, multiple-sequence alignments, profile features, AlphaFold confidence features and function labels as primary inputs. This design reduces potential leakage and simplifies deployment.

Our study makes five contributions. First, it evaluates a sequence-only PLM-based RegionAwareTCN under a unified train/validation/test protocol. Second, it separates full DM3000 benchmark results from target-specific NR25 low-homology training results. Third, it uses error-analysis-driven representation upgrading to show that ESM2-t33 closes the main DISORDER723 ranking gap that smoothing and loss reweighting did not solve. Fourth, it reports hard-case stratification across SDR/LDR, terminal/internal regions, residue zones, protein length bins and protein disorder-content bins. Fifth, it turns raw model scores into calibrated probabilities and evaluates ECE, Brier score, NLL and uncertainty-error enrichment.

## Materials and Methods

### Datasets and label handling

The study used DM3000 as the training set and DM1229 as the validation set. Three independent external benchmarks were used for final testing: SL329, MXD494 and DISORDER723 (Table 1). Labels were treated at residue level with three possible states: ordered, disordered and unknown. Unknown residues were excluded from all metric calculations. This primarily affects SL329, which contains 89,582 unknown residues. Thresholds and calibration parameters were selected only on DM1229 and were then fixed before test-set evaluation.

The external benchmarks differ substantially in disorder prevalence. Among known residues, SL329 has a disorder fraction of 0.435334, MXD494 has 0.224360 and DISORDER723 has 0.062845. DISORDER723 is therefore especially sensitive to class imbalance and requires AUPR, MCC and calibration metrics in addition to ROC-AUC.

### NR25 leakage-control evaluation

To evaluate low-homology robustness, target-specific NR25 training sets were generated by removing DM3000 training proteins with MMseqs2 percentage identity above 25% against each target benchmark. The resulting training sets retained 2,824 proteins for SL329, 2,677 proteins for MXD494 and 2,576 proteins for DISORDER723. Full DM3000 and NR25 results were reported separately to avoid mixing standard benchmark performance with low-homology generalization claims.

### Sequence representations

The main model is sequence-only. It does not use PDB coordinates, experimentally missing residues, multiple-sequence alignments, profile features, AlphaFold confidence features or function labels as primary inputs. Residue-level ESM2 embeddings were extracted and cached. The P4.6 representation-upgrade experiment replaced ESM2-t12-35M layer 12 with ESM2-t33-650M layer 33 while keeping the RegionAwareTCN head, thresholding protocol and benchmark evaluation fixed.

### RegionAwareTCN architecture

RegionAwareTCN applies temporal convolutional sequence modeling to frozen residue representations, residue one-hot features and relative-position features. The model includes auxiliary supervision for short versus long disordered regions and terminal versus internal disorder patterns. These auxiliary tasks are not presented as universally improving every aggregate metric. Instead, they provide a region-aware training and analysis scaffold that aligns model evaluation with known IDR heterogeneity.

The selected final predictor averages scores from three independently trained ESM2-t33 RegionAwareTCN models with seeds 1, 2 and 3. The ensemble score is used for ranking metrics, validation-selected thresholding and calibration.

### Training, model selection and thresholding

Models were trained on DM3000 or target-specific NR25 training sets and selected using validation performance on DM1229. The binary decision threshold was selected on DM1229 by maximizing Fmax for the corresponding raw or calibrated score distribution. Test labels were never used for threshold tuning, model selection or calibration fitting.

### Calibration and uncertainty

Post-hoc calibration was fitted using DM1229 validation predictions. Raw scores, temperature scaling, Platt scaling and isotonic regression were evaluated. Platt scaling was selected for the main calibrated output because it preserved ranking metrics while improving ECE, Brier score and NLL with lower overfitting risk than isotonic regression.

Uncertainty was computed from calibrated binary predictive entropy. To test whether uncertainty was informative, residues were ranked by uncertainty and the error rate among the most uncertain residues was compared with the overall error rate.

### Metrics and statistical testing

Residue-level performance was evaluated with sensitivity (Sn), specificity (Sp), balanced accuracy (BACC), Matthews correlation coefficient (MCC), ROC-AUC, AUPR and Fmax. Calibration quality was evaluated with expected calibration error (ECE), Brier score and negative log-likelihood (NLL). Stratified analyses were performed for SDR/LDR, terminal/internal IDRs, residue zones, protein length bins and protein disorder-content bins.

The ESM2-t33 and ESM2-t12 RegionAwareTCN ensembles were compared with protein-level paired bootstrap confidence intervals and paired permutation tests. Protein-level resampling was used because residues from the same protein are correlated. External SOTA methods were available only as aggregate published metrics, so they were compared as point references rather than by paired statistical tests.

## Results

The following sections summarize full benchmark performance, representation-upgrade effects, NR25 robustness, hard-case stratification, and calibration/uncertainty analyses.

### Full benchmark comparison

Under the full DM3000 training protocol, the ESM2-t33 RegionAwareTCN ensemble reached AUC values of 0.919327, 0.850637 and 0.944611 on SL329, MXD494 and DISORDER723, respectively (Table 2 and Figure 2). These point estimates exceed the curated direct SOTA AUC values for all three benchmarks: IDP-EDL on SL329, FusionEncoder on MXD494 and IDP-EDL on DISORDER723. The gains over the direct SOTA point AUC values were +0.004327, +0.008637 and +0.001611, respectively.

The threshold-dependent metrics provide a more nuanced view. On SL329, the model achieved MCC 0.736481, exceeding the curated IDP-EDL MCC of 0.700. On MXD494, it achieved MCC 0.499439, slightly above the curated FusionEncoder MCC of 0.492. On DISORDER723, however, MCC was 0.610462, below the curated IDP-EDL MCC of 0.636. Therefore, the most defensible performance claim is AUC-level SOTA point performance, not comprehensive dominance across all metrics.

### Representation upgrade from ESM2-t12 to ESM2-t33

The representation upgrade improved AUC point estimates on all external datasets. The t33-minus-t12 AUC delta was +0.004055 on SL329, +0.004781 on MXD494 and +0.021479 on DISORDER723 (Table 2 and Table 4). Protein-level paired statistics showed that the DISORDER723 improvement was strongly supported, with a 95% bootstrap CI of 0.013868 to 0.028520 and paired permutation p=0.001996. The SL329 and MXD494 deltas were positive but their protein-level CIs crossed zero, so these gains should be interpreted as favorable point improvements rather than statistically secure improvements under conservative protein-level resampling.

This result connects directly to the preceding error analysis. Low-cost post-processing and loss variants did not close the DISORDER723 gap, whereas a stronger frozen PLM representation substantially improved ranking performance. The largest gain was also observed on the most difficult benchmark, supporting the interpretation that representation quality is the main driver of the remaining full-benchmark performance improvement.

### Target-specific NR25 evaluation

NR25 training reduced performance relative to the full DM3000 setting (Table 3). The AUC drop was small on SL329 (-0.002266), larger on MXD494 (-0.016548) and moderate on DISORDER723 (-0.007736). Under NR25 training, the model remained above the curated SL329 SOTA AUC but fell below the curated MXD494 and DISORDER723 SOTA AUC values. This supports a transparent robustness claim: RegionAwareTCN remains competitive under low-homology evaluation, but it is not uniformly low-homology SOTA.

### Hard-case stratification

Hard-case stratification confirmed that IDR prediction difficulty is region dependent (Figure 3). In DISORDER723, terminal IDR AUC was 0.968186, whereas internal IDR AUC was 0.887416 and internal MCC was 0.210033. This gap indicates that ESM2-t33 alleviates but does not solve internal IDR classification. The same qualitative pattern appears in other datasets, where internal IDRs are consistently harder than terminal IDRs. These results justify keeping terminal/internal analysis as a central part of the manuscript rather than treating it as a supplemental diagnostic only.

### Calibration and uncertainty

Platt calibration preserved AUC, AUPR and MCC while improving probability quality (Table 5). The largest ECE improvements were observed on MXD494 and DISORDER723. On DISORDER723, ECE decreased from 0.187936 to 0.017608 and Brier score decreased from 0.093709 to 0.032479. On MXD494, ECE decreased from 0.256921 to 0.100902 and Brier score decreased from 0.219616 to 0.146934.

Calibrated uncertainty was also informative (Figure 4). The top 10% most uncertain residues were enriched for errors by 2.57x on SL329, 2.51x on MXD494 and 6.96x on DISORDER723. This supports uncertainty-aware use of the predictor, especially for low-prevalence or annotation-sensitive benchmarks where a single binary call may be misleading.

## Discussion

This project supports a sequence-only, region-aware and calibrated framing for intrinsic disorder prediction. The final ESM2-t33 RegionAwareTCN ensemble reaches AUC-level SOTA point performance on SL329, MXD494 and DISORDER723 under the full DM3000 training protocol. Importantly, the strongest evidence is not merely the final point estimate but the experimental chain leading to it: P4.5 showed that smoothing, position removal and loss reweighting did not close the DISORDER723 gap, whereas P4.6 showed that upgrading the frozen PLM representation substantially improved DISORDER723 ranking and internal-IDR behavior.

The study also clarifies the boundary of the performance claim. Protein-level paired statistics support a strong t33-over-t12 gain on DISORDER723, but the corresponding gains on SL329 and MXD494 are smaller and not significant under conservative protein-level resampling. In addition, the external SOTA methods do not provide residue-level predictions, so direct paired tests against them are not possible. For this reason, the manuscript should claim AUC-level SOTA point performance rather than broad statistical dominance over all existing predictors.

The NR25 experiments are an important strength because they separate standard benchmark performance from low-homology generalization. The t33 model remains competitive under NR25 training and remains above the SL329 SOTA AUC, but MXD494 and DISORDER723 fall below the curated SOTA values. This limitation should be stated explicitly. It improves the credibility of the paper and helps position the work as leakage-aware rather than benchmark-optimized.

Calibration is a practical contribution. Platt scaling markedly improves ECE and Brier score, especially on MXD494 and DISORDER723, without changing AUC, AUPR or MCC. The uncertainty analysis further shows that calibrated uncertainty is biologically useful as a caution signal: residues with the highest uncertainty are much more likely to be prediction errors. This is particularly relevant for disorder boundaries, ambiguous annotations and low-prevalence datasets.

Internal IDRs remain the major unsolved technical problem. Although ESM2-t33 improved DISORDER723 internal AUC, internal MCC remained low. This suggests that internal disorder may require additional signals, such as function-aware labels, binding-region annotations, MoRF labels, or specialized hard-example training. Structure features should not be added casually because local audits found no controlled structure-feature coverage and several benchmarks contain PDB-chain-like identifiers, creating leakage risk.

Overall, the best manuscript position is not a generic claim that a new neural network solves disorder prediction. The stronger contribution is a transparent and calibrated PLM-era benchmark framework that achieves strong full-training performance, exposes low-homology and internal-IDR limitations, and provides probability estimates suitable for uncertainty-aware use.

## Conclusion

RegionAwareTCN provides a sequence-only, region-aware and calibrated framework for residue-level intrinsic disorder prediction. The ESM2-t33 three-seed ensemble achieves AUC-level SOTA point performance on three external benchmarks under the full DM3000 protocol, with the strongest statistically supported representation-upgrade gain on DISORDER723. The model remains competitive but not uniformly SOTA under target-specific NR25 evaluation. Platt calibration substantially improves probability quality and calibrated uncertainty enriches prediction errors, making the framework more useful for practical interpretation than a raw ranking score alone. The main remaining challenges are internal IDR classification, DISORDER723 MCC and low-homology generalization on MXD494 and DISORDER723.

## Data and Code Availability

To be finalized before submission. Current project assets include labeled FASTA files under `data/`, prediction files under `predictions/`, result tables under `results/`, figures under `figures/`, and manuscript assets under `manuscript/`. The final public release should include trained model weights, embedding-generation instructions or cached-feature instructions, environment files, evaluation scripts and calibration scripts.

## Author Contributions

To be filled.

## Competing Interests

To be filled.

## Acknowledgements

To be filled.

## References Draft

- IDP-EDL. Briefings in Bioinformatics, 2025. https://academic.oup.com/bib/article/26/2/bbaf182/8116687
- FusionEncoder. Bioinformatics, 2025. https://academic.oup.com/bioinformatics/article/41/7/btaf362/8169326
- Critical Assessment of Protein Intrinsic Disorder Round 3 (CAID3). PubMed: https://pubmed.ncbi.nlm.nih.gov/40859602/
- CAID3 full text. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/
- Modern resources for intrinsic disorder predictions. 2026 review: https://link.springer.com/article/10.1007/s00018-026-06087-3
- Additional local references to formalize from `references/`: IDP-Fusion, SPOT-Disorder2, IDP-LM, DR-BERT, DisoFLAG, DisorderUnetLM, PUNCH2, PredIDR2, ESMDisPred and flDPnn3.

## Tables

# Table 1. Dataset and NR25 leakage-control summary

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

# Table 2. Full DM3000 benchmark performance

| dataset | method | sn | sp | bacc | mcc | auc | aupr | fmax | sota_method | sota_auc | auc_gap_vs_sota | t12_auc | t33_minus_t12_auc | delta_ci_95 | paired_permutation_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SL329 | RegionAwareTCN + ESM2-t33, 3-seed ensemble | 0.774150 | 0.942759 | 0.858455 | 0.736481 | 0.919327 | 0.918972 | 0.842101 | IDP-EDL | 0.915000 | 0.004327 | 0.915272 | 0.004055 | -0.002693 to 0.011198 | 0.169661 |
| MXD494 | RegionAwareTCN + ESM2-t33, 3-seed ensemble | 0.742350 | 0.812379 | 0.777365 | 0.499439 | 0.850637 | 0.604872 | 0.623957 | FusionEncoder | 0.842000 | 0.008637 | 0.845856 | 0.004781 | -0.003227 to 0.012471 | 0.177645 |
| DISORDER723 | RegionAwareTCN + ESM2-t33, 3-seed ensemble | 0.654000 | 0.972866 | 0.813433 | 0.610462 | 0.944611 | 0.690249 | 0.639095 | IDP-EDL | 0.943000 | 0.001611 | 0.923132 | 0.021479 | 0.013868 to 0.028520 | 0.001996 |

Note: protein-level paired statistics compare t33 against local t12 predictions, not against external aggregate SOTA methods.

# Table 3. Target-specific NR25 low-homology robustness

| dataset | removed_train_proteins | kept_train_proteins | full_auc | nr25_auc | auc_delta_nr25_minus_full | nr25_aupr | full_mcc | nr25_mcc | mcc_delta_nr25_minus_full | sota_auc | nr25_gap_vs_sota |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SL329 | 176 | 2824 | 0.919327 | 0.917061 | -0.002266 | 0.913435 | 0.736481 | 0.733184 | -0.003297 | 0.915000 | 0.002061 |
| MXD494 | 323 | 2677 | 0.850637 | 0.834089 | -0.016548 | 0.585903 | 0.499439 | 0.483503 | -0.015936 | 0.842000 | -0.007911 |
| DISORDER723 | 424 | 2576 | 0.944611 | 0.936875 | -0.007736 | 0.652169 | 0.610462 | 0.601105 | -0.009357 | 0.943000 | -0.006125 |

Note: NR25 results are reported as robustness evidence and are not uniformly SOTA.

# Table 4. Paper-level key ablation deltas

| axis | dataset | comparison | auc_delta | aupr_delta | mcc_delta | ece_delta | brier_delta | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture | SL329 | RegionAwareTCN_aux_t12_3seed minus GenericTCN_t12_3seed | -0.001532 | -0.003458 | -0.012859 | NA | NA | Region-aware supervision helps MXD494/DISORDER723 but is not uniformly beneficial for SL329. |
| representation | SL329 | ESM2-t33 RegionAwareTCN minus ESM2-t12 RegionAwareTCN | 0.004055 | 0.002627 | 0.031063 | NA | NA | ESM2-t33 is the decisive ranking-performance upgrade, especially on DISORDER723. |
| calibration | SL329 | Platt-calibrated t33 minus raw t33 | 0.000000 | 0.000000 | 0.000000 | -0.009123 | -0.015659 | Platt preserves ranking metrics and improves probability quality. |
| homology | SL329 | NR25-vs-SL329 t33 minus full-train t33 | -0.002266 | -0.005537 | -0.003297 | NA | NA | NR25 remains competitive but is not uniformly SOTA. |
| architecture | MXD494 | RegionAwareTCN_aux_t12_3seed minus GenericTCN_t12_3seed | 0.006270 | 0.052391 | 0.010291 | NA | NA | Region-aware supervision helps MXD494/DISORDER723 but is not uniformly beneficial for SL329. |
| representation | MXD494 | ESM2-t33 RegionAwareTCN minus ESM2-t12 RegionAwareTCN | 0.004781 | 0.004664 | 0.002013 | NA | NA | ESM2-t33 is the decisive ranking-performance upgrade, especially on DISORDER723. |
| calibration | MXD494 | Platt-calibrated t33 minus raw t33 | 0.000000 | 0.000000 | 0.000000 | -0.156019 | -0.072682 | Platt preserves ranking metrics and improves probability quality. |
| homology | MXD494 | NR25-vs-MXD494 t33 minus full-train t33 | -0.016548 | -0.018969 | -0.015936 | NA | NA | NR25 remains competitive but is not uniformly SOTA. |
| architecture | DISORDER723 | RegionAwareTCN_aux_t12_3seed minus GenericTCN_t12_3seed | 0.001908 | 0.006036 | 0.002509 | NA | NA | Region-aware supervision helps MXD494/DISORDER723 but is not uniformly beneficial for SL329. |
| representation | DISORDER723 | ESM2-t33 RegionAwareTCN minus ESM2-t12 RegionAwareTCN | 0.021479 | 0.064740 | 0.050966 | NA | NA | ESM2-t33 is the decisive ranking-performance upgrade, especially on DISORDER723. |
| calibration | DISORDER723 | Platt-calibrated t33 minus raw t33 | 0.000000 | 0.000000 | 0.000000 | -0.170328 | -0.061230 | Platt preserves ranking metrics and improves probability quality. |
| homology | DISORDER723 | NR25-vs-DISORDER723 t33 minus full-train t33 | -0.007736 | -0.038080 | -0.009357 | NA | NA | NR25 remains competitive but is not uniformly SOTA. |

# Table 5. Platt calibration and uncertainty-error enrichment

| dataset | auc | aupr | mcc | raw_ece | platt_ece | ece_delta | raw_brier | platt_brier | brier_delta | top10_uncertain_error_enrichment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SL329 | 0.919327 | 0.918972 | 0.736481 | 0.109384 | 0.100261 | -0.009123 | 0.126544 | 0.110885 | -0.015659 | 2.565826 |
| MXD494 | 0.850637 | 0.604872 | 0.499439 | 0.256921 | 0.100902 | -0.156019 | 0.219616 | 0.146934 | -0.072682 | 2.513963 |
| DISORDER723 | 0.944611 | 0.690249 | 0.610462 | 0.187936 | 0.017608 | -0.170328 | 0.093709 | 0.032479 | -0.061230 | 6.955547 |

Note: Platt calibration is fitted only on DM1229 validation predictions.

## Figure Captions

## Figure 1. Evidence chain and model framing

Schematic overview of the manuscript evidence chain. P4.5 error analysis identified DISORDER723 internal IDRs as the major remaining failure mode; P4.6 upgraded frozen sequence representations from ESM2-t12 to ESM2-t33 while keeping the sequence-only RegionAwareTCN head fixed; P5 evaluates the resulting model with full DM3000 training, target-specific NR25 training, hard-case stratification, and validation-fitted Platt calibration.

Source asset: `figures/p5/P5_EVIDENCE_CHAIN_MECHANISM.pdf`

## Figure 2. ROC and precision-recall curves for the representation upgrade

ROC and precision-recall curves comparing the ESM2-t12 and ESM2-t33 RegionAwareTCN 3-seed ensembles on SL329, MXD494, and DISORDER723. The t33 representation improves ranking metrics on all three datasets, with the largest AUC and AUPR gain on DISORDER723.

Source asset: `figures/p5/P5_T12_T33_ROC_PR_CURVES.pdf`

## Figure 3. Hard-case stratified performance

AUC of the ESM2-t33 RegionAwareTCN 3-seed ensemble across overall, SDR, LDR, terminal IDR, internal IDR, middle-residue, and low-disorder-content strata. The plot highlights that internal IDRs remain substantially harder than terminal IDRs, especially in DISORDER723.

Source asset: `figures/p5/P5_T33_HARD_CASE_STRATIFIED_AUC.pdf`

## Figure 4. Calibrated uncertainty tracks prediction errors

Error enrichment among the most uncertain residues after validation-fitted Platt calibration. Top-10% uncertainty residues are enriched for errors by 2.57x on SL329, 2.51x on MXD494, and 6.96x on DISORDER723, supporting uncertainty-aware use of the predictor.

Source asset: `figures/p5/P5_T33_PLATT_UNCERTAINTY_ERROR_ENRICHMENT.pdf`

## Supplementary Figure. Reliability diagrams

Reliability diagrams for raw, temperature-scaled, Platt-calibrated, and isotonic-calibrated t33 ensemble predictions on DM1229 validation and the three external test sets.

Source assets: `figures/p4_6/calibration/reliability_p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_*.pdf`

## Latest Literature Check

Date checked: 2026-08-03

## Sources Checked

- IDP-EDL, Briefings in Bioinformatics 2025: https://academic.oup.com/bib/article/26/2/bbaf182/8116687
- FusionEncoder, Bioinformatics 2025: https://academic.oup.com/bioinformatics/article/41/7/btaf362/8169326
- CAID3, Proteins 2026 / PubMed: https://pubmed.ncbi.nlm.nih.gov/40859602/
- CAID3 full text / PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/
- 2026 review, Modern resources for intrinsic disorder predictions: https://link.springer.com/article/10.1007/s00018-026-06087-3

## Current Direct Benchmark Competitors

- IDP-EDL remains the direct local comparator for SL329 and DISORDER723 in the curated project tables.
- FusionEncoder remains the direct local comparator for MXD494 in the curated project tables.
- FusionEncoder also reports DISORDER723 and MXD494 benchmark results and CAID3 results, reinforcing that PLM-based semantic features and fusion remain current in 2025 methods.

## 2026 Context

- A 2026 review emphasizes that intrinsic disorder prediction is now shaped by protein language models, deep learning, meta-servers, and curated disorder/function databases.
- CAID3 confirms that current community evaluation increasingly emphasizes PLM-era predictors, challenge-style assessment, AUC/Fmax/average precision, and statistical comparisons such as DeLong tests.
- CAID3 also introduced a binding-region-in-IDR subchallenge, which supports treating disorder function prediction as an important future extension rather than a current main claim for this project.

## Manuscript Impact on This Project

The manuscript should position itself against PLM-era methods rather than older profile-only predictors. The direct benchmark claim should remain tied to SL329/MXD494/DISORDER723 point estimates from the curated local comparison table, while CAID3 and recent reviews should be used to motivate:

1. sequence-only and PLM-based representation,
2. transparent low-homology evaluation,
3. AUPR/Fmax reporting under class imbalance,
4. probability calibration and uncertainty as practical reliability features.

## Required Manual Citation Audit Before Submission

Before journal submission, manually verify whether any 2026 method has reported direct SL329, MXD494, or DISORDER723 results exceeding the current local SOTA table. If none is found, keep IDP-EDL and FusionEncoder as the direct benchmark comparators.
