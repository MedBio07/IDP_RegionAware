# P5.4 Environment Snapshot

- Python executable: `/opt/anaconda3/bin/python`
- Python version: `3.13.9 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 19:16:10) [GCC 11.2.0]`
- Platform: `Linux-6.8.0-124-generic-x86_64-with-glibc2.35`
- Current LD_LIBRARY_PATH: `/opt/anaconda3/lib:/usr/local/cuda/lib64`
- Required local library setting observed during audit: `LD_LIBRARY_PATH=/opt/anaconda3/lib:/usr/local/cuda/lib64`

## Python Packages

| module | version/status | check |
| --- | --- | --- |
| numpy | 2.3.5 | PASS |
| pandas | 2.3.3 | PASS |
| sklearn | 1.7.2 | PASS |
| scipy | 1.16.3 | PASS |
| torch | 2.13.0+cu130 | PASS |
| matplotlib | 3.10.6 | PASS |
| seaborn | 0.13.2 | PASS |
| esm | 2.0.0 | PASS |
| yaml | 6.0.3 | PASS |

## Storage Footprint

- ESM2 embedding cache: approximately 5.1 GB in `data/features/esm2_embeddings/`.
- Prediction files: approximately 504 MB in `predictions/`.
- Trained model weights: approximately 171 MB in `models/`.

## Environment Risk

The project does not yet include a full `environment.yml`, `requirements.txt` or container image. P5.4 generates `requirements_p5_4.txt` as a provisional snapshot, but a final public release should use a clean locked Conda environment or Docker/Singularity container.
