# P4.5 SOTA Gap-Closing Summary

Date: 2026-07-31

## Scope

P4.5 tested low-cost strategies intended to close the remaining SOTA gap, especially on DISORDER723:

- DISORDER723 residue/segment/protein error analysis.
- Validation-selected local score smoothing.
- Removing relative position from RegionAwareTCN.
- Equal ensembles combining the main model with no-position or focal variants.
- Focal BCE and asymmetric BCE re-training.

## Generated Assets

- `scripts/analyze_disorder_errors.py`
- `scripts/evaluate_score_postprocessing.py`
- `scripts/summarize_p4_5_gap_closing.py`
- `results/error_analysis/`
- `results/postprocess/`
- `results/p4_5/P4_5_GAP_CLOSING_COMPARISON.tsv`
- `results/p4_5/P4_5_DISORDER723_INTERNAL_COMPARISON.tsv`

## Main Result

Best DISORDER723 AUC in this P4.5 round:

- Variant: `validation_selected_w3_smoothing`
- AUC: 0.923186
- AUC gap vs IDP-EDL 0.943: -0.019814

This does not materially improve over the current main model AUC 0.923132 and remains far from IDP-EDL's reported 0.943.

## DISORDER723 Failure Mode

The dominant failure mode is internal/middle disorder:

- Main RegionAware internal AUC: 0.838827
- Main RegionAware internal AUPR: 0.105671
- Main RegionAware internal MCC: 0.158647

Best internal AUC in this P4.5 round:

- Variant: `region_aware_focal_g2_seed1`
- Internal AUC: 0.848233
- Internal AUPR: 0.119204
- Internal MCC: 0.184759

Some variants improve internal-region metrics, but they reduce overall AUC and do not close the SOTA gap.

## Interpretation

1. Local smoothing is not the answer. Validation-selected smoothing changes DISORDER723 AUC by only about +0.00005.
2. Removing position confirms a terminal-position bias component: internal metrics improve slightly, but overall performance drops.
3. Focal/asymmetric losses improve specificity or selected internal metrics, but they lower overall AUC.
4. Simple equal ensembles do not preserve the main model's AUC advantage.

## Decision

Do not enter final P5 as a performance-SOTA paper yet.

The current evidence supports a reliability/generalization/calibration manuscript, but not a full SOTA-performance claim. To pursue a higher-impact performance claim, the next technical step should be stronger representation, not more local post-processing:

1. Extract and test a larger PLM, preferably ESM2-t33-650M or ProtT5 if available.
2. Keep the current RegionAware + Platt pipeline fixed as the comparison scaffold.
3. Specifically monitor DISORDER723 internal IDR, 501-1000 aa proteins, LDR, and middle-zone residues.
4. Only after a representation upgrade improves DISORDER723 AUC should multi-seed and NR25 repeats be run.

## Go/No-Go

Current P4.5 first-round result: No-Go for P5 performance-SOTA framing.

Allowed next directions:

- P4.6 representation upgrade experiment.
- Or P5 reliability-focused manuscript framing without claiming broad SOTA.
