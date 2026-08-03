# IDP RegionAware

Region-aware intrinsic disorder prediction experiments for DM3000 training and
external IDP benchmarks.

## Repository Contents

- `scripts/`: training, prediction, evaluation, calibration, ablation, and
  manuscript-asset assembly scripts.
- `models/`: model architecture and feature code. Trained model weights are
  intentionally excluded from Git.
- `configs/`: experiment and data configuration files.
- `reports/`: project execution and experiment summaries.
- `manuscript/`: manuscript drafts, tables, and LaTeX source.
- `figures/`: generated manuscript figures that are small enough for Git.

Large local artifacts are excluded by `.gitignore`, including raw data,
embeddings, trained weights, prediction TSVs, generated result tables, local
Conda environments, and cached files.

## Environment

Install the provisional runtime requirements with:

```bash
python3 -m pip install -r requirements_p5_4.txt
```

For CUDA-enabled PyTorch, install the wheel or channel appropriate for the
target workstation. The local environment used `torch==2.13.0+cu130`.

## Data

The main labeled FASTA files are expected under `data/`:

- `data/DM3000_Train.fasta`
- `data/DM1229_Validation.fasta`
- `data/SL329_test.fasta`
- `data/MXD494_test.fasta`
- `data/DISORDER723_test.fasta`

These data files and derived embeddings are not tracked in Git because of size
and redistribution concerns. See `PROJECT_STATUS.md` for the local data summary.

## Example Evaluation

```bash
python3 scripts/evaluate_disorder_predictions.py \
  --labels data/SL329_test.fasta \
  --predictions path/to/SL329_predictions.tsv \
  --dataset SL329 \
  --out results/SL329_my_method_metrics.tsv
```

## Project Status

See `PROJECT_STATUS.md`, `reports/`, and `manuscript/` for the current
benchmarking, calibration, ablation, and manuscript status.
