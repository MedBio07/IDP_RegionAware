# P5.7 RegionAdapterMoETCN Gate Mechanism Analysis

Date: 2026-08-04

## Scope

This analysis uses the P4.7/P4.8 warm-start three-seed `RegionAdapterMoETCN` ensemble and summarizes the mean residue-level MoE gate weights over DM1229 validation, SL329, MXD494 and DISORDER723. Gate experts correspond to SDR, LDR, terminal IDR and internal IDR adapters. Results are descriptive mechanism evidence, not a causal intervention.

## Key Observations

- The learned gate is not a clean one-hot biological region classifier. Ordered validation residues have the highest mean internal-IDR gate component (`mean_gate_internal_idr=0.319726`), so the internal expert cannot be interpreted directly as an internal-IDR detector.
- The clearest positive alignment is on DM1229 validation SDR and terminal-IDR residues: SDR residues have `mean_gate_sdr=0.410476` and terminal-IDR residues have `mean_gate_terminal_idr=0.433842`.
- Validation internal-IDR residues still have above-random target gate mass (`mean_target_gate_weight=0.296802` versus a four-expert random baseline of 0.25), but their largest mean gate component is SDR rather than internal.
- External internal-IDR routing is heterogeneous: target gate mass is `0.221655` on SL329, `0.239106` on MXD494 and `0.295357` on DISORDER723.
- DISORDER723 LDR residues increase the LDR expert contribution (`mean_gate_ldr=0.244537`), but the gate remains distributed rather than sharply specialized.

## Manuscript Interpretation

The gate analysis supports only a cautious auxiliary mechanism claim: warm-start adapters learn region-shifted routing preferences, especially for validation SDR and terminal-IDR residues, but the gate should not be described as a direct biological region classifier. The primary evidence for P4.8 should remain empirical: P4.8 preserves aggregate benchmark/NR25 performance and gives its largest practical gain on DISORDER723 internal IDRs.

## Generated Files

- `results/p5_7/P5_7_REGION_ADAPTER_MOE_GATE_SUMMARY.tsv`
- `results/p5_7/P5_7_REGION_ADAPTER_MOE_GATE_FOCUS.tsv`
- `figures/p5_7/P5_7_REGION_ADAPTER_MOE_GATE_SPECIALIZATION.pdf`
- `figures/p5_7/P5_7_REGION_ADAPTER_MOE_GATE_SPECIALIZATION.png`
