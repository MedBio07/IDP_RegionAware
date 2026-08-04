<!--
P5.7 P4.8 replacement draft.
Still not submission-final until authors/affiliations, funding, conflict-of-interest,
repository DOI, license, exact hardware/runtime and formal references are inserted.
-->

# RegionAdapterMoETCN: warm-start region-specialized adapters for calibrated sequence-only intrinsic disorder prediction

## Article Type

Original Paper

## Running Title

Warm-start region adapters for IDR prediction

## Authors

To be filled.

## Abstract

Intrinsic disorder prediction has benefited from protein language models, but benchmark gains can be difficult to interpret when disorder labels are regionally heterogeneous, test sets differ in prevalence, and homologous training examples may inflate apparent performance. We developed RegionAdapterMoETCN, a sequence-only residue-level predictor that warm-starts from an ESM2-t33 RegionAwareTCN backbone and adds low-rank region-specialized adapters with a residue-level mixture-of-experts gate for short/long and terminal/internal disorder patterns. The final model uses three-seed score averaging, validation-selected thresholding and Platt calibration fitted only on the DM1229 validation set. Across SL329, MXD494 and DISORDER723, the P4.8 warm-start RegionAdapterMoETCN ensemble achieved AUC values of 0.920545, 0.850471 and 0.945409, respectively, exceeding curated direct state-of-the-art point AUC values under the full DM3000 training protocol. Relative to the P4.6 ESM2-t33 RegionAwareTCN ensemble, the largest practical gain was observed on DISORDER723 internal IDRs, where AUC, AUPR and MCC improved by 0.007242, 0.019317 and 0.027346. Protein-level paired bootstrap intervals supported small positive AUC deltas on SL329 and DISORDER723, while MXD494 was effectively unchanged. Target-specific NR25 evaluation showed that warm-start adapters preserved low-homology robustness relative to P4.6 NR25 models. Platt calibration preserved ranking metrics while reducing DISORDER723 ECE from 0.159805 to 0.016327 and Brier score from 0.085060 to 0.032010. These results support a calibrated, leakage-aware and hard-case-stratified sequence-only framework, with the strongest new evidence centered on internal-IDR generalization rather than broad aggregate dominance.

## Keywords

intrinsic disorder prediction; intrinsically disordered regions; protein language model; ESM2; adapter; mixture of experts; calibration; uncertainty; low-homology evaluation

## Introduction

Intrinsically disordered proteins and intrinsically disordered regions (IDRs) are central to regulation, signaling and molecular recognition, but their residue-level prediction remains difficult. Disorder is not a single homogeneous state: disordered residues may occur in short or long segments, at protein termini or in internal regions, and in proteins with very different overall disorder content. Benchmark labels also vary in origin and certainty, including DisProt-like annotations, missing-residue-derived annotations, NoX-style labels and unknown residues. These factors make global benchmark metrics useful but incomplete.

Recent IDR predictors increasingly use protein language models (PLMs), feature fusion, ensembles and length-aware or region-aware predictors. IDP-EDL and FusionEncoder define strong direct comparators on SL329, MXD494 and DISORDER723, while CAID-style evaluations emphasize ranking metrics, threshold-dependent metrics, low-similarity subsets and functional extensions. However, three gaps remain important for a method paper. First, full-training benchmark gains can be misleading without explicit low-homology controls. Second, internal and terminal IDRs can have different error profiles that are hidden by aggregate AUC. Third, most predictors report raw scores rather than calibrated probabilities, limiting their practical use for experimental prioritization.

We present RegionAdapterMoETCN, a sequence-only residue-level IDR predictor that upgrades the earlier RegionAwareTCN framework with warm-start region-specialized low-rank adapters. The model keeps frozen ESM2-t33-650M residue embeddings, one-hot residue features and relative-position features fixed, initializes from matched RegionAwareTCN seeds, freezes the shared temporal-convolutional backbone, and trains adapter, expert-head and gate parameters for SDR, LDR, terminal-IDR and internal-IDR specialization. This design separates the strong representation/backbone effect from the region-adapter contribution and avoids uncontrolled structure, profile or function-label inputs.

The study makes five contributions. First, it introduces a warm-start region-adapter MoE specialization over a strong sequence-only ESM2-t33 backbone. Second, it separates full DM3000 benchmark performance from target-specific NR25 low-homology training. Third, it reports hard-case stratification across SDR/LDR, terminal/internal IDRs, residue zones and protein-level strata. Fourth, it fits Platt calibration on validation predictions and evaluates probability quality and uncertainty-error enrichment. Fifth, it includes exploratory gate analysis to test whether region routing is interpretable, while keeping the primary claim anchored to empirical benchmark, NR25 and hard-case evidence.

## Materials and Methods

### Datasets and label handling

The study used DM3000 as the training set and DM1229 as the validation set. Three independent external benchmarks were used for final testing: SL329, MXD494 and DISORDER723. Labels were treated at residue level with three states: ordered, disordered and unknown. Unknown residues were excluded from all metric calculations, which primarily affects SL329. Thresholds, calibration parameters and model choices were selected on DM1229 and fixed before test-set evaluation.

The three external benchmarks differ substantially in disorder prevalence. Among known residues, the disorder fraction is 0.435334 for SL329, 0.224360 for MXD494 and 0.062845 for DISORDER723. Because DISORDER723 is highly imbalanced, AUPR, MCC, Fmax and calibration metrics are reported alongside ROC-AUC.

### NR25 leakage-control evaluation

Target-specific NR25 training sets were generated by removing DM3000 training proteins with MMseqs2 percentage identity above 25% against each target benchmark. The resulting training sets retained 2,824 proteins for SL329, 2,677 proteins for MXD494 and 2,576 proteins for DISORDER723. Full DM3000 and NR25 results are reported separately to avoid mixing standard benchmark performance with low-homology generalization claims.

### RegionAdapterMoETCN architecture

The P4.8 model is sequence-only. It uses frozen ESM2-t33 residue embeddings, relative-position features and residue one-hot features. A shared temporal-convolutional backbone is initialized from a P4.6 RegionAwareTCN checkpoint. RegionAdapterMoETCN then adds four low-rank residual adapters and four expert heads corresponding to SDR, LDR, terminal-IDR and internal-IDR patterns. A residue-level softmax gate mixes expert outputs and adds the resulting expert delta to a generic disorder logit.

For the selected warm-start configuration, each seed is initialized from its matched P4.6 RegionAwareTCN seed. The shared backbone is frozen, and only region adapters, expert heads and the gate head are trained. The final predictor averages scores from three independently warm-started seeds. This makes P4.6 both the warm-start source and the strongest backbone baseline, not a discarded comparator.

### Calibration, uncertainty and gate analysis

Post-hoc calibration was fitted using DM1229 validation predictions. Raw scores, temperature scaling, Platt scaling and isotonic regression were evaluated. Platt scaling was selected for the main calibrated output because it preserved ranking metrics while improving expected calibration error (ECE), Brier score and negative log-likelihood with lower overfitting risk than isotonic regression. Uncertainty was computed from calibrated binary predictive entropy.

The mixture-of-experts gate was analyzed descriptively by averaging gate weights across the three warm-start seeds and summarizing them by region strata. This analysis was used only as auxiliary mechanism evidence. The gate is not treated as a direct biological classifier because learned routing can reflect optimization and score-correction behavior rather than one-hot region labels.

### Metrics and statistical testing

Residue-level performance was evaluated with sensitivity (Sn), specificity (Sp), balanced accuracy (BACC), Matthews correlation coefficient (MCC), ROC-AUC, AUPR and Fmax. Calibration quality was evaluated with ECE, Brier score and negative log-likelihood. Stratified analyses were performed for SDR/LDR, terminal/internal IDRs, residue zones, protein length bins and protein disorder-content bins.

P4.8 and P4.6 predictions were compared with protein-level paired bootstrap confidence intervals and paired permutation tests. Protein-level resampling was used because residues from the same protein are correlated. External SOTA methods were available only as aggregate published metrics, so they were compared as point references rather than by paired statistical tests.

## Results

### Warm-start RegionAdapterMoETCN preserves full-benchmark performance and improves hard cases

Under the full DM3000 training protocol, the P4.8 warm-start RegionAdapterMoETCN ensemble reached AUC values of 0.920545, 0.850471 and 0.945409 on SL329, MXD494 and DISORDER723, respectively (Table 2). These point estimates exceed the curated direct SOTA AUC values for all three benchmarks: IDP-EDL on SL329, FusionEncoder on MXD494 and IDP-EDL on DISORDER723. The corresponding point gaps versus SOTA were +0.005545, +0.008471 and +0.002409.

Compared with the P4.6 ESM2-t33 RegionAwareTCN ensemble, P4.8 changed full-benchmark AUC by +0.001218 on SL329, -0.000166 on MXD494 and +0.000798 on DISORDER723. Protein-level bootstrap confidence intervals for the AUC delta were 0.000298 to 0.002092 on SL329, -0.001384 to 0.001085 on MXD494 and 0.000015 to 0.001502 on DISORDER723. Thus, the aggregate performance story is preservation with small favorable point changes, not large global dominance.

### Internal-IDR gains provide the strongest performance evidence

The main practical improvement occurs in internal-IDR hard cases (Table 4). On DISORDER723 internal IDRs, P4.8 improved AUC from 0.887416 to 0.894658, AUPR from 0.152541 to 0.171858 and MCC from 0.210033 to 0.237379. DISORDER723 middle residues also improved, with AUC +0.001514, AUPR +0.010856 and MCC +0.007044. MXD494 internal-IDR MCC improved by 0.005448, although SL329 internal-IDR MCC decreased slightly. These results justify making internal-IDR behavior a central result while keeping dataset-specific caveats explicit.

### NR25 evaluation confirms that warm-start adapters do not break low-homology robustness

Target-specific NR25 training reduced performance relative to full DM3000 training, but P4.8 preserved or slightly improved NR25 AUC relative to P4.6 NR25 baselines (Table 3). The P4.8-minus-P4.6 NR25 AUC deltas were +0.001496 on SL329, -0.000394 on MXD494 and +0.001319 on DISORDER723. Relative to external SOTA point AUC values, P4.8 NR25 remained above SL329 SOTA but below MXD494 and DISORDER723 SOTA. The appropriate low-homology claim is therefore robustness relative to the internal backbone baseline, not uniformly low-homology SOTA.

### Platt calibration improves probability quality and uncertainty tracks errors

Platt calibration preserved AUC, AUPR and MCC while improving probability quality (Table 5). For the P4.8 model, ECE decreased from 0.108403 to 0.087562 on SL329, from 0.241887 to 0.108556 on MXD494 and from 0.159805 to 0.016327 on DISORDER723. Brier score decreased by 0.016725, 0.066280 and 0.053050, respectively.

Calibrated uncertainty was informative. The top 10% most uncertain residues were enriched for errors by 2.73-fold on SL329, 2.52-fold on MXD494 and 7.03-fold on DISORDER723. This supports uncertainty-aware interpretation, especially for low-prevalence or annotation-sensitive benchmarks.

### Gate analysis supports only a cautious routing interpretation

Exploratory gate analysis showed partial but not one-hot region specialization. On DM1229 validation, SDR residues had mean SDR-gate weight 0.410476 and terminal-IDR residues had mean terminal-IDR-gate weight 0.433842. However, ordered validation residues had the highest mean internal-IDR gate component, and external internal-IDR routing was heterogeneous: internal target gate mass was 0.221655 on SL329, 0.239106 on MXD494 and 0.295357 on DISORDER723. The gate therefore should not be interpreted as a direct biological region classifier. It is better viewed as a learned routing mechanism that helps region-specific score correction after warm-starting.

## Discussion

P4.8 changes the manuscript emphasis from a representation-upgraded RegionAwareTCN to a warm-start region-adapter specialization over a strong ESM2-t33 RegionAwareTCN backbone. This matters for novelty: the main methodological claim is not simply that a larger PLM improves disorder prediction, but that a frozen strong backbone can be locally specialized with low-rank region adapters and a residue-level MoE gate while preserving calibration and low-homology robustness.

The claim boundary remains conservative. Full-benchmark AUC deltas over P4.6 are small, and paired permutation tests do not support broad global dominance. The strongest direct evidence is the hard-case improvement on DISORDER723 internal IDRs, where AUC, AUPR and MCC all improve meaningfully. This gives the method a clearer biological and evaluation-centered rationale than aggregate AUC alone.

The NR25 experiments are important because they separate standard benchmark performance from low-homology generalization. P4.8 does not collapse under target-specific NR25 training and remains close to or slightly better than P4.6 NR25 baselines. However, MXD494 and DISORDER723 NR25 AUC values remain below curated external SOTA point values, so the paper should avoid claiming uniform low-homology SOTA.

Calibration remains a practical contribution. Platt scaling substantially improves ECE and Brier score, especially on MXD494 and DISORDER723, without changing ranking metrics. Calibrated uncertainty also enriches prediction errors, making the predictor more useful as a prioritization and review tool than a raw ranking score alone.

The gate analysis is useful but limited. It shows routing shifts for validation SDR and terminal-IDR residues, yet it does not provide a clean biological decoder for every region type. This is not a failure of the model, but it constrains the interpretation: the gate should be framed as an optimization mechanism for region-specific adaptation, not as a stand-alone biological annotation model.

## Conclusion

RegionAdapterMoETCN provides a sequence-only, warm-start region-adapter framework for calibrated residue-level intrinsic disorder prediction. The P4.8 three-seed ensemble achieves AUC-level SOTA point performance on three external benchmarks under the full DM3000 protocol, preserves target-specific NR25 robustness relative to P4.6, and provides its clearest practical improvement on DISORDER723 internal IDRs. Platt calibration substantially improves probability quality and calibrated uncertainty enriches prediction errors. The remaining limitations are small aggregate gains over the strong P4.6 backbone, heterogeneous gate interpretability, DISORDER723 MCC below external SOTA and non-uniform low-homology SOTA performance.

## Data Availability

Code, configuration files, reproducibility scripts, figures and manuscript assets are available at https://github.com/MedBio07/IDP_RegionAware. Trained model weights, validation-fitted calibration parameters, residue-level predictions, benchmark result tables and large derived artifacts should be archived with a permanent DOI before submission.

## Main Tables

See:

- `manuscript/tables/P5_7_Table2_p4_8_full_benchmark.md`
- `manuscript/tables/P5_7_Table3_p4_8_nr25_robustness.md`
- `manuscript/tables/P5_7_Table4_p4_8_hard_case_gain.md`
- `manuscript/tables/P5_7_Table5_p4_8_calibration_uncertainty.md`

## Main Figure Updates

Figure 5 should be added as an exploratory mechanism/supplementary figure:

- `figures/p5_7/P5_7_REGION_ADAPTER_MOE_GATE_SPECIALIZATION.pdf`

