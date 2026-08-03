# P4 Structure/Function Extension Audit

Date: 2026-07-31

## Scope

P4 was executed as a local feasibility and leakage-risk audit before any structure-aware or function-aware model training.

Inputs:

- `data/DM3000_Train.fasta`
- `data/DM1229_Validation.fasta`
- `data/SL329_test.fasta`
- `data/MXD494_test.fasta`
- `data/DISORDER723_test.fasta`
- local `references/` text/XML files for literature support only

Generated assets:

- `scripts/audit_structure_function_resources.py`
- `results/structure/P4_DATASET_ID_TYPE_SUMMARY.tsv`
- `results/structure/P4_POTENTIAL_PDB_MAPPING.tsv`
- `results/structure/P4_STRUCTURE_COVERAGE_BY_DATASET.tsv`
- `results/structure/P4_STRUCTURE_RESOURCE_AUDIT.tsv`
- `results/structure/P4_FUNCTION_RESOURCE_AUDIT.tsv`
- `results/structure/P4_LITERATURE_TERM_COUNTS.tsv`
- `results/structure/P4_DECISION_MATRIX.tsv`

## Dataset ID Audit

Overall dataset sizes:

| dataset | split_role | proteins | residues | known_residues | disorder_residues | unknown_residues | disorder_fraction_known |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DISORDER723 | test | 723 | 215229 | 215229 | 13526 | 0 | 0.062845 |
| DM1229_Validation | validation | 1229 | 305830 | 305830 | 29082 | 0 | 0.095092 |
| DM3000_Train | train | 3000 | 730804 | 730804 | 74170 | 0 | 0.101491 |
| MXD494 | test | 494 | 196501 | 196501 | 44087 | 0 | 0.224360 |
| SL329 | test | 329 | 180418 | 90836 | 39544 | 89582 | 0.435334 |

ID-type composition:

| dataset | id_type | proteins | unique_pdb_ids | residues | unknown_residues | disorder_fraction_known |
| --- | --- | --- | --- | --- | --- | --- |
| DISORDER723 | pdb_chain | 723 | 704 | 215229 | 0 | 0.062845 |
| DM1229_Validation | custom_validation | 1229 | 0 | 305830 | 0 | 0.095092 |
| DM3000_Train | disprot | 52 | 0 | 12395 | 0 | 1.000000 |
| DM3000_Train | pdb_chain | 2120 | 2096 | 485118 | 0 | 0.074846 |
| DM3000_Train | pdb_variant | 828 | 778 | 233291 | 0 | 0.109160 |
| MXD494 | disprot | 289 | 0 | 142635 | 0 | 0.276952 |
| MXD494 | pdb_chain | 205 | 204 | 53866 | 0 | 0.085100 |
| SL329 | disprot | 329 | 0 | 180418 | 89582 | 0.435334 |

Structure mapping/leakage signals:

| dataset | proteins | pdb_mappable_proteins | pdb_mappable_fraction | unique_pdb_ids | local_structure_matched_proteins | local_structure_coverage | high_leakage_risk_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DISORDER723 | 723 | 723 | 1.000000 | 704 | 0 | 0.000000 | 1.000000 |
| DM1229_Validation | 1229 | 0 | 0.000000 | 0 | 0 | 0.000000 | 0.000000 |
| DM3000_Train | 3000 | 2948 | 0.982667 | 2874 | 0 | 0.000000 | 0.982667 |
| MXD494 | 494 | 205 | 0.414980 | 204 | 0 | 0.000000 | 0.414980 |
| SL329 | 329 | 0 | 0.000000 | 0 | 0 | 0.000000 | 0.000000 |

Interpretation:

- DM3000, MXD494, and DISORDER723 contain many PDB-chain or PDB-fragment style IDs. These are useful for tracing benchmark provenance, but they make experimentally solved structure features risky.
- SL329 uses DisProt-style IDs, so structure use would require a separate accession/mapping step and a dated source audit.
- DM1229 validation uses project-local IDs and has no direct external mapping in the FASTA headers.

## Local Structure Resource Audit

Candidate local structure resources detected: 0.

Decision from local scan: no local PDB/mmCIF/AlphaFold/ESMFold/pLDDT/PAE resource files were detected.

Because local structure coverage is currently zero, P4 should not proceed directly to structure-enhanced training. A structure-aware model would first need a controlled resource acquisition step.

## Local Function Resource Audit

Candidate local function-label resources detected outside `references/`, `reports/`, and `scripts/`: 0.

Decision from local scan: no local DisProt/CAID/MoRF/binding/linker label table was detected.

This means function auxiliary heads are not ready for training from local files alone. The literature supports the biological relevance of disordered function prediction, but labels must be collected and split-audited first.

## Literature Support Signals

Term totals across local reference text/XML files:

| AlphaFold | ESMFold | pLDDT | PAE | DSSP | SASA | function | binding | MoRF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 49 | 3 | 28 | 2 | 0 | 13 | 607 | 412 | 45 |

Top reference files by P4-relevant term hits:

| source_file | AlphaFold | ESMFold | pLDDT | PAE | DSSP | SASA | function | binding | MoRF | total_hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| references/DisoFLAG_2024_BMC_Biology.epmc.xml | 0 | 0 | 0 | 0 | 0 | 0 | 148 | 111 | 12 | 271 |
| references/pdf_texts/DisoFLAG_2024_BMC_Biology.txt | 0 | 0 | 0 | 0 | 0 | 0 | 132 | 99 | 7 | 238 |
| references/pdf_texts/IDP-LM_2023_PLOS_Computational_Biology.txt | 7 | 0 | 7 | 0 | 0 | 0 | 98 | 35 | 1 | 148 |
| references/pdf_texts/CAID1_2021_Nature_Methods.txt | 0 | 0 | 0 | 0 | 0 | 0 | 17 | 69 | 2 | 88 |
| references/pdf_texts/flDPnn2.txt | 9 | 0 | 1 | 0 | 0 | 0 | 13 | 21 | 3 | 47 |
| references/pdf_texts/ESMDisPred.txt | 8 | 3 | 2 | 2 | 0 | 0 | 14 | 10 | 2 | 41 |
| references/pdf_texts/SPOT-Disorder2.txt | 0 | 0 | 0 | 0 | 0 | 1 | 16 | 7 | 14 | 38 |
| references/pdf_texts/DR-BERT.txt | 3 | 0 | 0 | 0 | 0 | 0 | 14 | 17 | 0 | 34 |

Interpretation:

- The local references strongly support function-aware IDR prediction, especially binding, MoRF, and flexible-linker directions.
- The current local literature set contains much weaker support for direct AlphaFold/pLDDT/PAE feature use in this project than for PLM plus function-aware modeling.

## P4 Decision

| extension | current_status | coverage_signal | leakage_signal | decision | recommended_action |
| --- | --- | --- | --- | --- | --- |
| structure_features | no local structure feature files | max local structure coverage=0.000000 | max high-risk PDB-derived ID fraction=1.000000 | no_go_for_training | collect UniProt-mapped AlphaFold/ESMFold confidence features first; avoid using solved PDB missing residues as direct inputs |
| function_auxiliary_heads | no local function-label tables | candidate local function resources=0 | function labels require source/date/split audit before multi-task training | pilot_only_after_label_collection | collect DisProt/CAID binding-linker/MoRF labels and keep them out of test-tuned threshold/model selection |

Main decision:

1. Do not train a structure-enhanced main model yet, because local structure feature coverage is zero.
2. Do not use solved PDB missing-residue or coordinate-derived features as ordinary inputs for DM3000/MXD494/DISORDER723 without a leakage-control design.
3. Treat AlphaFold/ESMFold confidence features as a future optional branch, preferably through UniProt mapping and sequence-alignment verification.
4. Treat function prediction as a more publication-relevant P4 extension, but only after collecting explicit DisProt/CAID/MoRF/binding/linker labels.

## Recommended Next P4 Actions

P4.1 Lock the current main model as the sequence-only calibrated model: `RegionAwareTCN + aux 3-seed ensemble + Platt`.

P4.2 Build an accession-mapping table for all records:

- PDB-chain IDs: map PDB code/chain to UniProt where possible.
- DisProt IDs: map DisProt entry to UniProt accession and evidence date.
- Project-local validation IDs: recover source accession if available from upstream metadata.

P4.3 If structure is still desired, collect AlphaFold/ESMFold pLDDT only for UniProt-mappable proteins, then report coverage and run a sequence-only versus sequence+pLDDT ablation on the same covered subset.

P4.4 If function extension is prioritized, collect residue-level labels for protein-binding IDRs, DNA/RNA-binding IDRs, MoRFs, and flexible linkers, then train auxiliary heads without changing the disorder-threshold selection protocol.

P4.5 For a high-level journal manuscript, keep structure/function as optional extension evidence unless coverage becomes high and leakage-controlled. The current strongest manuscript core remains: low-leakage NR25 evaluation, region-aware modeling, calibration, and uncertainty/error enrichment.
