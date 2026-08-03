# Figure Captions Draft

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
