# IDP-EDL full-benchmark reproduction predictions

This directory publishes residue-level scores from the supplied IDP-EDL
reproduction artifact for SL329, MXD494, and DISORDER723. The attachment was
validated against the local benchmark FASTA files before packaging. These are
recomputed reproduction results, not values copied from the paper tables.

This v2 release supersedes the earlier public-artifact package, which contained
only 322 SL329 proteins and truncated proteins at 1,023 residues. The old
prediction tables, summaries, manifest, and conclusions are not part of the
current release.

## Public files

- `sl329_residue_predictions.tsv.gz`
- `mxd494_residue_predictions.tsv.gz`
- `disorder723_residue_predictions.tsv.gz`
- `summary.tsv` and `summary.json`: final-score residue metrics
- `component_summary.tsv`: generic, LDR, SDR, and final-score metrics
- `paper_comparison.tsv`: reproduction minus paper-reported values
- `manifest.json`: source archive and public-content provenance

The compressed prediction tables contain only:

```text
dataset  protein_id  position  idp_edl_g_score  idp_edl_l_score  idp_edl_s_score  idp_edl_score
```

Sequences, amino-acid identities, FASTA headers, reference labels, and
evaluability flags are deliberately excluded. Aggregate counts in the summary
files retain enough information to audit the reported metrics.

## Source audit

Attachment SHA256:

```text
5ae756164c918499ac6a853b38d14d67ec39fcf8745c267b0991f2bbe30b532d
```

The source CSV files were checked residue by residue against the local test
sets. Protein IDs are matched case-insensitively, while public output uses the
canonical local FASTA identifiers.

| Dataset | Proteins | All residues | Evaluated | Unknown | Alignment errors |
|---|---:|---:|---:|---:|---:|
| SL329 | 329 | 180,418 | 90,836 | 89,582 | 0 |
| MXD494 | 494 | 196,501 | 196,501 | 0 | 0 |
| DISORDER723 | 723 | 215,229 | 215,229 | 0 | 0 |

No protein, position, amino-acid, label, or evaluability mismatch was found.
All four score columns are finite and lie in `[0, 1]`.

## Recomputed final-score results

Metrics use only residues marked evaluable in the source, with class prediction
defined as `idp_edl_score > 0.5`.

| Dataset | Sn | Sp | BACC | MCC | AUC | AUPR | Fmax |
|---|---:|---:|---:|---:|---:|---:|---:|
| SL329 | 0.688499 | 0.973173 | 0.830836 | 0.706311 | 0.919071 | 0.916129 | 0.836602 |
| MXD494 | 0.652528 | 0.864225 | 0.758377 | 0.496705 | 0.854314 | 0.604996 | 0.634646 |
| DISORDER723 | 0.478856 | 0.989579 | 0.734217 | 0.581104 | 0.937725 | 0.659822 | 0.623983 |

SL329 closely matches or exceeds the reported point metrics. MXD494 has a
different sensitivity-specificity operating point but similar BACC and MCC.
DISORDER723 remains below the paper's reported BACC, MCC, and AUC. Therefore,
the attachment does not justify a claim of exact numerical reproduction on all
three benchmarks. See `paper_comparison.tsv` and `component_summary.tsv` for
the complete audit.
