# P4.7 Region Adapter MoE Optimization Plan

Date: 2026-08-04

## Rationale

The current P4.6 manuscript has a credible evidence chain but limited method novelty. P4.7 upgrades the model from a region-aware auxiliary-head TCN to a region-specialized adapter mixture-of-experts model while keeping the sequence-only, frozen-ESM2-t33 setting fixed.

The main unresolved failure mode is internal IDR prediction. In P4.6, DISORDER723 terminal IDR AUC is 0.968186, but internal IDR AUC is 0.887416 and internal MCC is 0.210033. This makes internal IDR performance the primary optimization target.

## Model Change

P4.6:

- Frozen ESM2-t33 embeddings.
- Shared TCN backbone.
- Generic residue head plus simple expert logits and gate.
- Auxiliary SDR/LDR and terminal/internal heads.

P4.7:

- Frozen ESM2-t33 embeddings.
- Shared TCN backbone.
- Four low-rank residual adapters specialized for SDR, LDR, terminal IDR and internal IDR.
- Learned residue-level MoE gate over the four adapters.
- Generic residue head plus region-expert logits.
- Auxiliary heads remain for region annotation.
- Optional gate supervision uses normalized SDR/LDR/terminal/internal labels on disordered residues only.

## Primary Hypothesis

Region-specialized low-rank adapters should improve internal-IDR and hard-case metrics beyond the P4.6 RegionAwareTCN head without relying on structural features or MSA/profile inputs.

## Go/No-Go Criteria

- Primary: improve DISORDER723 internal IDR MCC over 0.210033.
- Strong positive: improve DISORDER723 internal IDR AUPR over 0.152541 and internal AUC over 0.887416.
- Safety: keep full DISORDER723 AUC within 0.002 of 0.944611.
- Secondary: improve MXD494 AUPR/MCC or keep them within 0.002 of P4.6 while improving internal IDR metrics.
- Claim boundary: if aggregate AUC does not improve, frame P4.7 as a hard-case/internal-IDR specialization experiment, not a new overall SOTA result.

## Seed Plan

1. Train seed 1 against P4.6 seed 1 and the P4.6 three-seed ensemble.
2. If seed 1 improves hard-case metrics without aggregate collapse, train seeds 2 and 3.
3. Rebuild ensemble, validation threshold, Platt calibration, NR25 variants and stratified tables.

## Expected Manuscript Impact

If successful, the method contribution becomes stronger: RegionAwareTCN is no longer only a TCN head over frozen PLM embeddings, but a sequence-only, region-specialized adapter MoE framework with direct evidence on internal-IDR failure modes.
