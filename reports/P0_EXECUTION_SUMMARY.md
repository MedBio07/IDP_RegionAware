# P0 Execution Summary

Date: 2026-07-31

## Completed

P0 has been executed for the current DM3000 IDR project workspace.

Created:

- `configs/data.yaml`
- `scripts/annotate_disorder_regions.py`
- `scripts/summarize_disorder_splits.py`
- `scripts/evaluate_stratified_predictions.py`
- `results/experiment_registry.tsv`
- `results/dataset_region_summary.tsv`
- `results/testset_region_summary.tsv`
- `results/region_annotations/protein_region_annotations.tsv`
- `results/region_annotations/disorder_segments.tsv`

Created supporting directories for later phases:

- `predictions/`
- `results/stratified/`
- `results/calibration/`
- `results/final_tables/`
- `figures/`
- `models/`
- `reports/`

## Region Definitions

The P0 scripts use the following fixed definitions:

- Known residue: label is `0` or `1`.
- Ignored residue: label is `-1`.
- Disorder segment: consecutive label `1` residues.
- SDR: disorder segment length `<30`.
- LDR: disorder segment length `>=30`.
- Terminal segment: segment starts within the N-terminal cutoff or ends within the C-terminal cutoff.
- Terminal cutoff: `max(10 residues, ceil(10% of sequence length))`.
- Protein disorder content bins: `0-5`, `5-20`, `20-80`, `80-100`.
- Protein length bins: `<=200`, `201-500`, `501-1000`, `>1000`.

## Key Test-Set Region Statistics

| Dataset | Proteins | Known residues | Disorder residues | Disorder rate | SDR segments | LDR segments | Terminal IDR residues | Internal IDR residues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SL329 | 329 | 90836 | 39544 | 0.435334 | 464 | 274 | 27550 | 11994 |
| MXD494 | 494 | 196501 | 44087 | 0.224360 | 577 | 271 | 35049 | 9038 |
| DISORDER723 | 723 | 215229 | 13526 | 0.062845 | 1363 | 60 | 9578 | 3948 |

## Verification

- `python3 -m py_compile` passed for all new scripts.
- `scripts/summarize_disorder_splits.py` generated all summary and annotation TSV files.
- `scripts/evaluate_stratified_predictions.py` passed an oracle-prediction smoke test on SL329.
- The generated dataset-level protein/residue counts match the previously recorded project status.

## Ready For P1

P1 can now start with frozen PLM baseline experiments. The immediate next implementation targets are:

1. `scripts/extract_plm_embeddings.py`
2. `models/dataset.py`
3. `models/baseline_heads.py`
4. `scripts/train_disorder_model.py`
5. `scripts/predict_disorder_model.py`
