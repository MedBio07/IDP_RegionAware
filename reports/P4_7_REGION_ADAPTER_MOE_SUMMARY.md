# P4.7 Region Adapter MoE Summary

Date: 2026-08-04

## Scope

P4.7 implemented route 1: region-specialized low-rank adapters plus a learned residue-level MoE gate over SDR, LDR, terminal IDR and internal IDR experts.

The model keeps the P4.6 sequence-only input setting fixed:

- Frozen ESM2-t33-650M embeddings.
- Relative-position and one-hot residue features.
- DM3000 training, DM1229 validation threshold selection.
- SL329, MXD494 and DISORDER723 external evaluation.

## Code Changes

- Added `RegionAdapterMoETCN` in `models/sequence_models.py`.
- Added low-rank residual region adapters initialized as identity-preserving residual updates.
- Added optional gate supervision in `scripts/train_sequence_disorder_model.py`.
- Added checkpoint loading support in `scripts/predict_sequence_disorder_model.py`.
- Added P4.7 configuration: `configs/p4_7_region_adapter_moe_tcn_esm2_t33.yaml`.

## Variants Tested

All variants used seed 1 unless otherwise stated.

| variant | gate loss | validation AUC | validation MCC | best epoch |
| --- | ---: | ---: | ---: | ---: |
| P4.7 gate010 | 0.10 | 0.924166 | 0.607667 | 4 |
| P4.7 gate002 | 0.02 | 0.924074 | 0.611245 | 2 |
| P4.7 nogate | 0.00 | 0.922451 | 0.610240 | 2 |
| P4.7 gate002 3-seed ensemble | 0.02 | 0.926659 | 0.617321 | NA |

## Main External Result

Against the P4.6 three-seed ensemble, the initial from-scratch P4.7 gate002 3-seed ensemble:

| dataset | delta AUC | delta MCC | delta internal AUC | delta internal AUPR | delta internal MCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| SL329 | +0.000235 | -0.002067 | -0.000384 | -0.004348 | -0.003347 |
| MXD494 | -0.005326 | -0.001278 | +0.000947 | +0.001225 | -0.004080 |
| DISORDER723 | -0.001915 | +0.002007 | -0.000392 | +0.004279 | -0.006338 |

Against P4.6 seed1, P4.7 gate002 seed1 showed clearer hard-case gains:

| dataset | delta AUC | delta MCC | delta internal AUC | delta internal AUPR | delta internal MCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| SL329 | +0.005477 | +0.013733 | +0.007672 | +0.028105 | +0.014819 |
| MXD494 | -0.005369 | +0.004715 | +0.009031 | +0.014040 | +0.010875 |
| DISORDER723 | -0.002888 | +0.000438 | +0.001490 | +0.039688 | +0.015301 |

## Interpretation

P4.7 demonstrates that region-specialized adapters are technically feasible and can improve internal-IDR AUPR/MCC against the same-seed P4.6 baseline, especially on DISORDER723. However, the three-seed ensemble does not beat the P4.6 three-seed ensemble robustly. The current from-scratch adapter-MoE training improves DISORDER723 MCC slightly but loses AUC and does not improve internal MCC against the ensemble baseline.

This is not yet strong enough to replace the current P4.6 manuscript main model.

## Warm-Start Adapter Result

The more faithful adapter/LoRA-style experiment warm-started `RegionAdapterMoETCN` from each matched P4.6 `RegionAwareTCN` seed, copied the shared backbone and heads, kept low-rank adapters identity-initialized, froze the shared backbone, and trained only adapters plus gate/expert heads with learning rate 0.0002.

Warm-start P4.7 gate002 3-seed ensemble versus P4.6 three-seed ensemble:

| dataset | delta AUC | delta AUPR | delta MCC | delta Fmax | delta internal AUC | delta internal AUPR | delta internal MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SL329 | +0.001218 | +0.001597 | -0.003300 | -0.001025 | +0.003178 | +0.008102 | -0.003705 |
| MXD494 | -0.000166 | -0.002043 | +0.002190 | +0.001022 | +0.002767 | +0.001175 | +0.005448 |
| DISORDER723 | +0.000798 | +0.002915 | +0.002689 | +0.003656 | +0.007242 | +0.019317 | +0.027346 |

This is the strongest P4.7 result. It provides a real method-upgrade signal on DISORDER723 internal IDRs while preserving or nearly preserving aggregate AUC on the three external datasets.

## Decision

P4.7 route 1 is successful as a method-upgrade candidate only in the warm-start adapter setting, not in the from-scratch adapter-MoE setting.

The current recommended next step is to run calibration, paired bootstrap/permutation and NR25 checks for the warm-start P4.7 three-seed ensemble before replacing the P4.6 manuscript main model.
