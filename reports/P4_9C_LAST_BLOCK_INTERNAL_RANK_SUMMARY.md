# P4.9-C Last-Block Internal-Ranking Decision

## Decision

P4.9-C replaces P4.8 as the selected full-training model. The locked candidate
improves ROC-AUC on SL329, MXD494, and DISORDER723, and improves Internal-IDR
AUC, AUPR, and MCC on all three datasets. The mean full-benchmark AUC gain is
0.000385.

This is a point-performance promotion, not a claim of uniform statistical
dominance. Protein-level paired resampling supports the SL329 AUC gain; the
smaller MXD494 and DISORDER723 gains are positive but not significant.

## Selected Method

The model remains the sequence-only P4.8 `RegionAdapterMoETCN` initialized from
the matching P4.8 seed checkpoint. P4.9-C changes the continuation objective
and trainable representation boundary:

1. Internal-IDR residues are ranked above ordered residues from the same
   protein with a segment-balanced softplus pairwise loss.
2. A frozen P4.8 teacher protects ordered and Terminal-IDR logits with Huber
   loss; Internal-IDR positives are intentionally unprotected.
3. Region adapters, expert heads, and the gate are trained at `1e-4`.
4. Only the last TCN block and final normalization are additionally unfrozen,
   at `2e-5`; earlier shared layers remain frozen.

The training loss is

```text
L = L_BCE + 0.02 L_gate + 0.05 L_internal_rank + 0.10 L_teacher
```

where each Internal-IDR segment contributes equally to `L_internal_rank`, with
at most 16 Internal positives and 16 ordered negatives sampled per segment.

The three seed scores are combined with validation-locked weights
`0.4/0.2/0.4`. The weights and raw threshold (`0.813982`) were selected only on
DM1229. Platt scaling was also fitted only on DM1229; its operating threshold is
`0.321806`.

## Full-Benchmark Results

All values below use the raw ranking scores. Threshold metrics use the locked
DM1229 threshold.

| Dataset | P4.8 AUC | P4.9-C AUC | Delta | AUPR delta | MCC delta | BACC delta |
|---|---:|---:|---:|---:|---:|---:|
| SL329 | 0.920545 | 0.921516 | +0.000971 | +0.001129 | +0.002039 | +0.001108 |
| MXD494 | 0.850471 | 0.850593 | +0.000122 | +0.003678 | +0.001530 | +0.001192 |
| DISORDER723 | 0.945409 | 0.945471 | +0.000062 | -0.001894 | -0.000945 | +0.001614 |

DISORDER723 AUPR and MCC remain negative secondary deltas and must not be
omitted from the paper. Fmax changes are -0.000534, -0.000719, and -0.000091 on
SL329, MXD494, and DISORDER723, respectively.

## Internal-IDR Evidence

| Dataset | P4.8 Internal AUC | P4.9-C Internal AUC | AUC delta | AUPR delta | MCC delta |
|---|---:|---:|---:|---:|---:|
| SL329 | 0.891854 | 0.893864 | +0.002010 | +0.004394 | +0.007433 |
| MXD494 | 0.799026 | 0.800510 | +0.001484 | +0.002153 | +0.005798 |
| DISORDER723 | 0.894658 | 0.895972 | +0.001314 | +0.002438 | +0.003716 |

The consistency across all three datasets directly supports the project
motivation: Internal IDRs are the targeted weakness, and an Internal-directed
objective improves that stratum without reducing full-dataset AUC.

## Protein-Level Paired Statistics

The unit of resampling is the protein. Statistics use 2,000 bootstrap samples
and 1,000 paired permutations.

| Dataset | AUC delta | 95% bootstrap CI | One-sided permutation p |
|---|---:|---:|---:|
| SL329 | +0.000971 | [0.000074, 0.001898] | 0.009990 |
| MXD494 | +0.000122 | [-0.000641, 0.000889] | 0.470529 |
| DISORDER723 | +0.000062 | [-0.000286, 0.000409] | 0.377622 |

## Calibration

Platt calibration preserves ranking metrics and substantially improves the
candidate's probability quality relative to its raw scores.

| Dataset | Platt Brier | Platt NLL | Platt ECE |
|---|---:|---:|---:|
| DM1229 Validation | 0.045973 | 0.164566 | 0.002021 |
| SL329 | 0.108029 | 0.386534 | 0.085518 |
| MXD494 | 0.154332 | 0.498001 | 0.109981 |
| DISORDER723 | 0.032093 | 0.119060 | 0.016081 |

Calibration relative to P4.8 is mixed on MXD494 and nearly unchanged on
DISORDER723. The defensible claim is that Platt makes P4.9-C scores usable as
calibrated probabilities, not that every calibration metric beats P4.8.

## Target-Specific NR25

Each NR25 model starts from its corresponding target-specific P4.8 NR25
checkpoint and retains the initial checkpoint unless a trained epoch improves
DM1229 validation AUC.

| Dataset | P4.8 NR25 AUC | P4.9-C NR25 AUC | AUC delta | AUPR delta | MCC delta |
|---|---:|---:|---:|---:|---:|
| SL329 | 0.918557 | 0.918715 | +0.000158 | +0.000531 | +0.000287 |
| MXD494 | 0.833695 | 0.833628 | -0.000067 | +0.002709 | -0.000451 |
| DISORDER723 | 0.938194 | 0.938697 | +0.000503 | +0.004177 | +0.004145 |

The small MXD494 NR25 AUC regression is within the predeclared 0.001 tolerance.
NR25 Internal-IDR AUC improves by +0.001578, +0.000997, and +0.002265 on
SL329, MXD494, and DISORDER723, respectively; Internal AUPR and MCC also improve
on all three datasets.

## Rejected Routes

- Factorized length/location gating improved Internal metrics but reduced mean
  full-benchmark AUC and was not promoted.
- Rank-only and adapter-only rank-plus-teacher continuation showed a monotonic
  Internal-versus-overall tradeoff and were not promoted.
- Stronger teacher scaling and no-main-loss continuation did not remove that
  tradeoff without the last-block representation update.

## Reproduction Skeleton

Train one full-data seed (replace `{seed}` and output paths):

```bash
python scripts/train_sequence_disorder_model.py \
  --train data/DM3000_Train.fasta \
  --validation data/DM1229_Validation.fasta \
  --experiment-id p4_9c_lastblock_rank005_protect010_seed{seed} \
  --features esm,position,onehot \
  --model-type RegionAdapterMoETCN \
  --embedding-dir data/features/esm2_embeddings/esm2_t33_650M_UR50D_layer33_fp16 \
  --init-from-checkpoint models/p4_7_region_adapter_moe_tcn_esm2_t33_gate002_warm_seed{seed}.pt \
  --freeze-shared-backbone --unfreeze-last-block \
  --main-loss-weight 1 --aux-loss-weight 0 --gate-loss-weight 0.02 \
  --pairwise-rank-weight 0.05 --teacher-protect-weight 0.10 \
  --pairwise-margin 0.2 --pairwise-max-per-segment 16 \
  --learning-rate 0.0001 --last-block-learning-rate 0.00002 \
  --weight-decay 0.0001 --epochs 4 --dropout 0 --seed {seed} \
  --model-out models/p4_9c_seed{seed}.pt \
  --metrics-out results/p4_9c_seed{seed}.tsv \
  --epoch-log-out results/p4_9c_seed{seed}_epochs.tsv \
  --validation-predictions-out predictions/p4_9c_seed{seed}_DM1229.tsv
```

Build the locked ensemble:

```bash
python scripts/ensemble_disorder_predictions.py \
  --inputs predictions/p4_9c_seed1_DM1229.tsv \
           predictions/p4_9c_seed2_DM1229.tsv \
           predictions/p4_9c_seed3_DM1229.tsv \
  --weights 0.4 0.2 0.4 \
  --out predictions/p4_9c_w40204_DM1229.tsv
```

Use `scripts/calibrate_disorder_predictions.py` for DM1229-only Platt fitting
and `scripts/compare_paired_auc.py` for protein-level paired statistics.

## Release Scope

Source code, tests, configuration, and this aggregate report are suitable for
Git. Raw datasets, ESM embeddings, checkpoints, general result TSVs, and
residue-level predictions remain local under the existing `.gitignore` policy.
