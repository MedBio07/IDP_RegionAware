# P4.6 Representation Upgrade Summary

Date: 2026-08-03

## Scope

P4.6 tested whether upgrading the frozen protein-language-model representation closes the remaining SOTA gap while keeping the sequence-only RegionAwareTCN framework fixed.

Main change:

- Previous representation: ESM2-t12-35M layer 12.
- New representation: ESM2-t33-650M layer 33.
- Same model head: RegionAwareTCN + SDR/LDR + terminal/internal auxiliary heads.
- Same protocol: train on DM3000, tune threshold only on DM1229 validation, evaluate SL329/MXD494/DISORDER723.

## Generated Assets

- `configs/p4_6_region_aware_tcn_esm2_t33.yaml`
- `data/features/esm2_embeddings/esm2_t33_650M_UR50D_layer33_fp16/`
- `models/p4_6_region_aware_tcn_esm2_t33_position_onehot_seed*.pt`
- `predictions/p4_6/`
- `results/p4_6/P4_6_REPRESENTATION_UPGRADE_COMPARISON.tsv`
- `results/p4_6/P4_6_T33_MULTISEED_RESULTS.tsv`
- `results/p4_6/P4_6_T33_MULTISEED_SUMMARY.tsv`
- `results/p4_6/P4_6_NR25_COMPARISON.tsv`
- `results/p4_6/P4_6_SELECTED_PLATT_CALIBRATION.tsv`
- `figures/p4_6/calibration/`

## Main Full-Train Result

| Dataset | t12 AUC | t33 3-seed AUC | AUC delta | SOTA AUC | Gap vs SOTA | t33 MCC |
|---|---:|---:|---:|---:|---:|---:|
| SL329 | 0.915272 | 0.919327 | 0.004055 | 0.915000 | 0.004327 | 0.736481 |
| MXD494 | 0.845856 | 0.850637 | 0.004781 | 0.842000 | 0.008637 | 0.499439 |
| DISORDER723 | 0.923132 | 0.944611 | 0.021479 | 0.943000 | 0.001611 | 0.610462 |

The t33 3-seed ensemble exceeds the collected SOTA AUC on all three target datasets. The largest gain is on DISORDER723: AUC improves from 0.923132 to 0.944611.

## Remaining Caveat

The DISORDER723 AUC claim is now strong, but MCC is still below IDP-EDL's reported 0.636:

- t33 3-seed DISORDER723 MCC: 0.610462
- IDP-EDL DISORDER723 MCC: 0.636

This means the manuscript can claim AUC-level SOTA only if the comparison table is explicit about MCC/Fmax.

## Internal IDR Improvement

DISORDER723 internal IDR improved but remains the hard case:

- t12 internal AUC: 0.838827
- t33 3-seed internal AUC: 0.887416
- t33 3-seed internal MCC: 0.210033

This is an important mechanistic result: the representation upgrade addresses the failure mode identified in P4.5, but does not fully solve internal IDR classification.

## NR25 Sanity Check

NR25 t33 seed1 results:

| Dataset | NR25 AUC | SOTA AUC | Gap vs SOTA | NR25 MCC |
|---|---:|---:|---:|---:|
| SL329 | 0.917061 | 0.915000 | 0.002061 | 0.733184 |
| MXD494 | 0.834089 | 0.842000 | -0.007911 | 0.483503 |
| DISORDER723 | 0.936875 | 0.943000 | -0.006125 | 0.601105 |

Interpretation: full-train t33 gives a performance-SOTA signal; NR25 shows the low-homology setting is harder and should be reported as a separate robustness result, not hidden.

## Calibration

Platt calibration preserves AUC/AUPR/MCC and substantially improves ECE/Brier/NLL, especially on DISORDER723:

- DISORDER723 raw ECE: 0.187936
- DISORDER723 Platt ECE: 0.017608
- DISORDER723 raw Brier: 0.093709
- DISORDER723 Platt Brier: 0.032479

## Decision

P4.6 is successful.

The project can now move toward P5, but the P5 framing must be precise:

1. Main performance claim: sequence-only RegionAwareTCN with ESM2-t33 reaches AUC-level SOTA on SL329, MXD494, and DISORDER723 in the full DM3000 setting.
2. Reliability claim: Platt calibration provides much better probability quality without degrading ranking metrics.
3. Robustness claim: NR25 results remain competitive but do not uniformly beat SOTA, so low-homology generalization should be reported transparently.
4. Remaining limitation: DISORDER723 MCC and internal IDR classification are still below the strongest reported benchmark.
