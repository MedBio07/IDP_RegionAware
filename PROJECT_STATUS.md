# IDP Disorder Prediction Project Status

Last local review: 2026-07-31.

## Available Data

All main FASTA files use a labeled three-line record format:

1. FASTA header
2. amino-acid sequence
3. Python-list labels, where `1` is disordered, `0` is ordered, and `-1` is unknown or ignored

| Split | File | Proteins | Residues | Disordered | Ordered | Unknown | Disorder rate on known residues |
|---|---|---:|---:|---:|---:|---:|---:|
| Train | `data/DM3000_Train.fasta` | 3000 | 730804 | 74170 | 656634 | 0 | 0.1015 |
| Validation | `data/DM1229_Validation.fasta` | 1229 | 305830 | 29082 | 276748 | 0 | 0.0951 |
| Test | `data/SL329_test.fasta` | 329 | 180418 | 39544 | 51292 | 89582 | 0.4353 |
| Test | `data/MXD494_test.fasta` | 494 | 196501 | 44087 | 152414 | 0 | 0.2244 |
| Test | `data/DISORDER723_test.fasta` | 723 | 215229 | 13526 | 201703 | 0 | 0.0628 |

Notes:

- SL329 contains many `-1` labels. These residues should be masked during evaluation.
- The datasets are strongly class-imbalanced, especially DISORDER723.
- Label parsing should use a structured parser such as `ast.literal_eval`; simple delimiter splitting can miss the first and last labels because labels are stored with list brackets.

## NR25 Training Sets

`scripts/build_nr25_train_sets.py` generated three test-specific DM3000 training sets by searching each test set against DM3000 with MMseqs2 and removing any DM3000 sequence with `pident > 25%`.

| Target test set | Output train FASTA | Removed train proteins | Kept train proteins | Kept residues | Kept disordered | Kept ordered | Disorder rate |
|---|---|---:|---:|---:|---:|---:|---:|
| SL329 | `data/nr25_by_test/DM3000_Train_nr25_vs_SL329.fasta` | 176 | 2824 | 686451 | 64364 | 622087 | 0.0938 |
| MXD494 | `data/nr25_by_test/DM3000_Train_nr25_vs_MXD494.fasta` | 323 | 2677 | 639506 | 54830 | 584676 | 0.0857 |
| DISORDER723 | `data/nr25_by_test/DM3000_Train_nr25_vs_DISORDER723.fasta` | 424 | 2576 | 604856 | 59658 | 545198 | 0.0986 |

Supporting files:

- `data/nr25_by_test/summary.tsv`
- `data/nr25_by_test/mmseqs_hits/*.m8`
- `data/nr25_by_test/removed_ids/*.tsv`

## Collected Literature Performance

The literature results are organized under `results/literature_test_results/`.

Important files:

- `current_methods_on_SL329_MXD494_DISORDER723.tsv`
- `current_methods_with_year_results.tsv`
- `three_test_sets_total_with_latest_methods.tsv`
- `recent_2023_2026_methods_scope.tsv`

Best collected AUC by target dataset:

| Dataset | Best method | Year | Sn | Sp | BACC | MCC | AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| MXD494 | FusionEncoder | 2025 | 0.742 | 0.806 | 0.774 | 0.492 | 0.842 |
| SL329 | IDP-EDL | 2025 | 0.690 | 0.970 | 0.828 | 0.700 | 0.915 |
| DISORDER723 | IDP-EDL | 2025 | 0.603 | 0.984 | 0.793 | 0.636 | 0.943 |

Coverage of recent 2023-2026 methods in the local review:

| Category | Count |
|---|---:|
| Recent methods checked | 12 |
| Direct results on target datasets | 2 |
| Partial or non-comparable target result | 1 |
| No direct target-dataset result found locally | 9 |

The only recent methods merged into the main direct-comparison table are IDP-EDL and FusionEncoder. FusionEncoder has no direct SL329 result in the collected files.

## Recommended Next Step

Use the current files to run a controlled model-development loop:

1. Train on `data/DM3000_Train.fasta` and tune on `data/DM1229_Validation.fasta`.
2. Report final metrics on SL329, MXD494, and DISORDER723 using the same residue-level metrics: Sn, Sp, BACC, MCC, and AUC.
3. For leakage-controlled reporting, repeat training or fine-tuning with the three NR25 train files and evaluate only on the corresponding target test set.
4. Treat the table above as the minimum comparison target. A useful new result should be compared against FusionEncoder on MXD494 and IDP-EDL on SL329 and DISORDER723.
5. Use `scripts/evaluate_disorder_predictions.py` to mask `-1` labels, compute all five metrics, and write a single TSV row compatible with the collected benchmark tables.

Example evaluation command:

```bash
python3 scripts/evaluate_disorder_predictions.py \
  --labels data/SL329_test.fasta \
  --predictions path/to/SL329_predictions.tsv \
  --dataset SL329 \
  --out results/SL329_my_method_metrics.tsv
```
