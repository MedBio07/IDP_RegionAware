# Manuscript Blueprint Draft

## Working Title

RegionAwareTCN: sequence-only, region-aware and uncertainty-calibrated intrinsic disorder prediction with protein language model representations

## Central Claim

RegionAwareTCN combines frozen ESM2-t33 residue representations with region-aware sequence modeling and validation-fitted Platt calibration. In the full DM3000 benchmark setting, it achieves AUC-level SOTA point performance on SL329, MXD494 and DISORDER723. Under target-specific NR25 training it remains competitive but not uniformly SOTA, defining a transparent low-homology generalization boundary.

## Contributions

1. A sequence-only IDR predictor that avoids PDB coordinate, missing-residue, MSA/profile and function-label inputs in the main model.
2. Region-aware supervision and hard-case evaluation for SDR/LDR and terminal/internal disorder patterns.
3. Error-analysis-driven representation upgrade showing that ESM2-t33, rather than smoothing or loss reweighting, closes the DISORDER723 ranking gap.
4. Leakage-aware evaluation with both full DM3000 training and target-specific NR25 training.
5. Calibrated probability output with Platt scaling, reliability metrics, and uncertainty-error enrichment.

## Recommended Main Text Structure

### Introduction

- IDRs are functionally important and difficult to model because disorder is regionally and annotation-wise heterogeneous.
- Modern methods increasingly use PLMs, feature fusion, length-aware predictors, and CAID-style evaluations.
- Current gaps: low-homology leakage control, internal/terminal hard-case analysis, and calibrated probability quality are usually underreported.
- This work focuses on a sequence-only, region-aware and uncertainty-calibrated framework.

### Materials and Methods

- Datasets: DM3000 train, DM1229 validation, SL329, MXD494, DISORDER723.
- Label handling: mask `-1` residues in SL329; evaluate only known residues.
- NR25: remove DM3000 training proteins with MMseqs2 pident >25% against each target test set.
- Model: frozen ESM2-t33 embeddings, one-hot and relative position features, RegionAwareTCN head, SDR/LDR and terminal/internal auxiliary supervision, three-seed score averaging.
- Calibration: fit Platt scaling only on DM1229 validation predictions.
- Statistics: protein-level bootstrap confidence intervals and paired permutation tests for t33 versus t12.

### Results

1. The t33 RegionAwareTCN ensemble reaches AUC-level SOTA point performance in the full DM3000 setting.
2. Representation upgrade is the main driver, especially on DISORDER723.
3. NR25 evaluation reveals a realistic low-homology generalization gap.
4. Hard-case stratification shows internal IDRs remain difficult despite t33 improvement.
5. Platt calibration improves ECE/Brier/NLL without changing ranking metrics.
6. Calibrated uncertainty enriches prediction errors and flags residues needing caution.

### Discussion

- The project should avoid claiming broad statistical dominance over external SOTA because external per-residue predictions are unavailable.
- The strongest mechanistic claim is DISORDER723: t33 improves AUC by 0.021479 with protein-level CI entirely above zero.
- SL329/MXD494 full benchmark AUC point estimates are above SOTA, but t33-vs-t12 deltas are modest and not protein-level significant.
- NR25 results are valuable precisely because they expose the low-homology boundary.
- Future work: function-aware labels, CAID/blind validation, and internal IDR-specialized modeling.

## Claim Wording

Use:

> RegionAwareTCN achieves AUC-level state-of-the-art point performance under the full DM3000 training protocol and provides calibrated, uncertainty-aware probabilities, while remaining competitive under target-specific NR25 evaluation.

Avoid:

> RegionAwareTCN significantly outperforms all existing predictors across all datasets and low-homology settings.
