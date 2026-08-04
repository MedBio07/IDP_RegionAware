# P5.4 Submission Hardening Report

Date: 2026-08-03

## Overall Status

P5.4 status: **CONDITIONALLY_READY_FOR_AUTHOR_COMPLETION**.

The scientific evidence chain is internally consistent, and the main benchmark metrics are reproducible from the final prediction files. The package is not yet submission-final because author-specific metadata, archival DOI, formal references, licensing and exact hardware/runtime details remain unresolved.

## Compliance Snapshot

| item | status |
| --- | --- |
| Bioinformatics article type | Original Paper |
| Main draft word count | 2788 words |
| Supplementary draft word count | 507 words |
| Main tables | 4 in P5.3 main draft |
| Main figures | 4 in P5.3 main draft |
| File integrity failures | 0 |
| Figure file failures | 0 |
| Reproducibility smoke-test rows | 48 PASS / 0 FAIL |
| Placeholder/manual-action rows | 8 |
| P0 submission blockers | 3 |

## Reproducibility Result

The P5.4 smoke test recomputed validation-threshold binary metrics from:

- `data/*_test.fasta`
- `predictions/p4_6/p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_*.tsv`
- validation threshold `0.833731` from `results/p4_6/p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_DM1229_Validation_metrics.tsv`

The recomputed Sn, Sp, BACC, MCC and AUC match both `*_metrics_val_threshold.tsv` files and manuscript Table 2. AUPR and Fmax recomputed from predictions match Table 2 and the per-test Fmax metric files.

Important clarification: the per-test `*_metrics.tsv` files store test-set Fmax-threshold binary metrics, while the manuscript Table 2 uses the DM1229 validation-selected threshold for Sn/Sp/BACC/MCC. This distinction should be explicit in Methods or table notes.

## SOTA Claim Hardening

The SOTA claim remains conservative:

- Keep: AUC-level SOTA point performance under full DM3000 training against curated direct target-set aggregate metrics.
- Keep: strongest protein-level statistical support is t33 over t12 on DISORDER723.
- Do not claim: statistically significant superiority over external SOTA, uniformly SOTA NR25 performance, or comprehensive superiority across MCC/AUPR/Fmax.

The D2MOE 2026 arXiv risk was checked and does not report the same SL329/MXD494/DISORDER723 target-set results. CAID3, flDPnn3 and ESMDisPred remain important context but do not replace the direct comparator rows.

## Generated P5.4 Files

- `manuscript/P5_4_Bioinformatics_submission_hardened_draft.md`
- `manuscript/P5_4_AVAILABILITY_AND_REPRODUCIBILITY_STATEMENT.md`
- `manuscript/P5_4_CORE_REFERENCE_AUDIT.md`
- `manuscript/P5_4_FINAL_SOTA_AUDIT.md`
- `manuscript/P5_4_ENVIRONMENT_SNAPSHOT.md`
- `manuscript/P5_4_SUBMISSION_BLOCKERS.tsv`
- `results/p5_4/P5_4_REPRODUCIBILITY_SMOKE_TEST.tsv`
- `results/p5_4/P5_4_FILE_INTEGRITY.tsv`
- `results/p5_4/P5_4_FIGURE_AUDIT.tsv`
- `results/p5_4/P5_4_PLACEHOLDER_AUDIT.tsv`
- `requirements_p5_4.txt`

## Next Required Actions

1. Fill authors, affiliations, corresponding author, funding, conflicts and final cover-letter signature.
2. Create archival DOI for weights, predictions, large result tables and derived artifacts.
3. Convert the reference audit into journal-ready references.
4. Add exact software/hardware/runtime details.
5. Re-run final direct SOTA search immediately before submission upload.
6. Convert Markdown tables and figures into the journal submission format.
