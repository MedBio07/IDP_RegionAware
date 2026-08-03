# Cover Letter Draft

Dear Editors,

We are pleased to submit our manuscript, "RegionAwareTCN: sequence-only, region-aware and uncertainty-calibrated intrinsic disorder prediction with protein language model representations", for consideration as an Original Paper in Bioinformatics.

Intrinsic disorder prediction has rapidly moved into the protein-language-model era, but performance claims remain difficult to interpret when benchmarks differ in disorder prevalence, annotation type, unknown labels and sequence similarity to training data. In this manuscript, we present RegionAwareTCN, a sequence-only residue-level predictor that combines frozen ESM2-t33 representations, temporal convolutional sequence modeling, region-aware auxiliary supervision and validation-fitted Platt calibration.

The manuscript contributes a conservative but useful performance and reliability evidence chain. Under the full DM3000 training protocol, RegionAwareTCN achieves AUC-level state-of-the-art point performance on SL329, MXD494 and DISORDER723. The strongest statistically supported representation-upgrade effect is observed on DISORDER723, where ESM2-t33 improves AUC by 0.021479 over the ESM2-t12 RegionAwareTCN ensemble under protein-level paired resampling. We also report target-specific NR25 low-homology evaluation, hard-case stratification and calibrated uncertainty analyses. These experiments expose the remaining limitations of internal IDR prediction and low-homology generalization while providing calibrated probabilities suitable for uncertainty-aware use.

We believe this work will interest Bioinformatics readers because it combines a practical sequence-only predictor with a transparent evaluation design that addresses common concerns in PLM-based benchmark studies: leakage control, hard-case behavior, class imbalance and probability calibration.

The manuscript has not been published elsewhere and is not under consideration by another journal. All authors have approved the submission. Data and code availability details will be finalized before submission.

Sincerely,

To be filled.
