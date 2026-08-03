# P5.4 Availability And Reproducibility Statement

## Availability And Implementation Draft

Code, configuration files, evaluation scripts, trained RegionAwareTCN weights, validation-fitted calibration parameters, residue-level prediction files, generated result tables and figure-generation assets should be deposited in a public repository before submission. The repository URL and archival DOI are not yet assigned and must be inserted before journal submission.

The current local package already contains the following reproducibility assets:

- Labeled FASTA datasets: `data/DM3000_Train.fasta`, `data/DM1229_Validation.fasta`, `data/SL329_test.fasta`, `data/MXD494_test.fasta`, and `data/DISORDER723_test.fasta`.
- Target-specific NR25 training sets and MMseqs2 hit artifacts: `data/nr25_by_test/`.
- Final ESM2-t33 RegionAwareTCN weights for seeds 1-3: `models/p4_6_region_aware_tcn_esm2_t33_position_onehot_seed*.pt`.
- Final raw and Platt-calibrated predictions: `predictions/p4_6/` and `predictions/p4_6/calibration/`.
- Main result tables and paper-level evidence bundles: `results/p4_6/`, `results/p4_6/calibration/`, and `results/p5/`.
- P5.4 reproducibility smoke-test outputs: `results/p5_4/`.

## Minimal Reproduction Path

1. Install the runtime described in `requirements_p5_4.txt`.
2. Set `LD_LIBRARY_PATH=/opt/anaconda3/lib:/usr/local/cuda/lib64` or use an equivalent Conda environment that resolves `sklearn` and `pyarrow` against the same C++ runtime.
3. Use the cached ESM2 embeddings in `data/features/esm2_embeddings/`, or regenerate them with `scripts/extract_plm_embeddings.py`.
4. Regenerate final predictions with `scripts/predict_sequence_disorder_model.py` and ensemble them with `scripts/ensemble_disorder_predictions.py`.
5. Recompute validation-threshold metrics with `scripts/evaluate_disorder_predictions.py`.
6. Recompute calibration and uncertainty summaries with `scripts/calibrate_disorder_predictions.py`.
7. Rebuild manuscript tables and figures with `scripts/compile_p5_evidence_bundle.py`, `scripts/assemble_p5_manuscript_assets.py`, and `scripts/assemble_p5_3_submission_package.py`.

## Submission-Time Missing Items

- Public repository URL and archival DOI.
- License for code and model weights.
- Confirmation that the 5.1 GB ESM2 embedding cache can be redistributed; otherwise provide regeneration commands only.
- Exact GPU model, driver version, CUDA runtime and wall-clock training times.
- Author-approved statement for third-party dataset redistribution rights.
