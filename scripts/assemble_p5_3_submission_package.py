#!/usr/bin/env python3
"""Create a Bioinformatics-oriented P5.3 submission package."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
TABLES = MANUSCRIPT / "tables"
SUPP = MANUSCRIPT / "supplementary"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def read_table_without_title(path: Path) -> str:
    text = read(path)
    return re.sub(r"^# .+?\n\n", "", text, count=1, flags=re.DOTALL).strip()


def strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_>|#]", " ", text)
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", strip_markdown(text)))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_submission_draft() -> str:
    table1 = read_table_without_title(TABLES / "Table1_dataset_and_nr25_summary.md")
    table2 = read_table_without_title(TABLES / "Table2_full_benchmark_sota.md")
    table3 = read_table_without_title(TABLES / "Table3_nr25_robustness.md")
    table5 = read_table_without_title(TABLES / "Table5_calibration_uncertainty.md")

    return f"""# RegionAwareTCN: sequence-only, region-aware and uncertainty-calibrated intrinsic disorder prediction with protein language model representations

## Article Type

Original Paper

## Running Title

Calibrated sequence-only IDR prediction

## Authors

To be filled.

## Corresponding Author

To be filled.

## Abstract

Intrinsic disorder prediction has benefited from protein language models, but benchmark gains can be difficult to interpret because disorder labels are regionally heterogeneous, test sets differ in prevalence, and homologous training examples may inflate apparent performance. We developed RegionAwareTCN, a sequence-only residue-level predictor that combines frozen ESM2 representations with temporal convolutional sequence modeling and auxiliary supervision for short/long and terminal/internal disorder patterns. The final model uses ESM2-t33 embeddings, three-seed score averaging, validation-selected thresholds, and Platt calibration fitted only on the DM1229 validation set. Across SL329, MXD494 and DISORDER723, the ESM2-t33 RegionAwareTCN ensemble achieved AUC values of 0.919327, 0.850637 and 0.944611, respectively, exceeding the curated direct state-of-the-art point AUC values for all three benchmarks under the full DM3000 training protocol. Protein-level paired resampling showed that the ESM2-t33 upgrade produced the strongest statistically supported gain on DISORDER723, improving AUC by 0.021479 over the ESM2-t12 RegionAwareTCN ensemble (95% bootstrap CI 0.013868 to 0.028520; paired permutation p=0.001996). Target-specific NR25 evaluation showed competitive but not uniformly state-of-the-art low-homology performance. Platt calibration preserved ranking metrics while reducing DISORDER723 ECE from 0.187936 to 0.017608 and Brier score from 0.093709 to 0.032479. These results support a calibrated, leakage-aware and hard-case-stratified sequence-only framework for intrinsic disorder prediction.

## Keywords

intrinsic disorder prediction; intrinsically disordered regions; protein language model; ESM2; calibration; uncertainty; low-homology evaluation

## Introduction

Intrinsically disordered proteins and intrinsically disordered regions (IDRs) are central to regulation, signaling and molecular recognition, but their residue-level prediction remains difficult. Disorder is not a single homogeneous state. Disordered residues may occur in short or long segments, at protein termini or in internal regions, and in proteins with very different overall disorder content. Benchmark labels also vary in origin and certainty, including DisProt-like annotations, missing-residue-derived annotations, NoX-style labels and unknown residues. These factors make global benchmark metrics useful but incomplete.

Recent IDR predictors increasingly use protein language models (PLMs), feature fusion, ensembles and length-aware or region-aware predictors. IDP-EDL and FusionEncoder define strong direct comparators on SL329, MXD494 and DISORDER723, while CAID-style evaluations emphasize ranking metrics, threshold-dependent metrics, low-similarity subsets and functional extensions. However, three gaps remain important for a method paper. First, full-training benchmark gains can be misleading without explicit low-homology controls. Second, internal and terminal IDRs can have different error profiles that are hidden by aggregate AUC. Third, most predictors report raw scores rather than calibrated probabilities, limiting their practical use for experimental prioritization.

We present RegionAwareTCN, a sequence-only residue-level IDR prediction framework that combines frozen ESM2 representations, temporal convolutional sequence modeling and auxiliary supervision for SDR/LDR and terminal/internal disorder patterns. The selected final model uses ESM2-t33-650M residue embeddings, one-hot and relative-position features, a three-seed score-averaged ensemble and validation-fitted Platt calibration. We keep the main model sequence-only and avoid PDB coordinates, experimentally missing residues, multiple-sequence alignments, profile features, AlphaFold confidence features and function labels as primary inputs. This design reduces leakage risk and simplifies deployment.

The study makes five contributions. First, it evaluates a sequence-only PLM-based RegionAwareTCN under a unified train/validation/test protocol. Second, it separates full DM3000 benchmark performance from target-specific NR25 low-homology training. Third, it uses error-analysis-driven representation upgrading to show that ESM2-t33 closes the DISORDER723 ranking gap that smoothing and loss reweighting did not solve. Fourth, it reports hard-case stratification across SDR/LDR, terminal/internal IDRs, residue zones and protein-level strata. Fifth, it evaluates calibrated probability quality and uncertainty-error enrichment.

## Materials and Methods

### Datasets and label handling

The study used DM3000 as the training set and DM1229 as the validation set. Three independent external benchmarks were used for final testing: SL329, MXD494 and DISORDER723 (Table 1). Labels were treated at residue level with three states: ordered, disordered and unknown. Unknown residues were excluded from all metric calculations, which primarily affects SL329. Thresholds, calibration parameters and model choices were selected on DM1229 and fixed before test-set evaluation.

The three external benchmarks differ substantially in disorder prevalence. Among known residues, the disorder fraction is 0.435334 for SL329, 0.224360 for MXD494 and 0.062845 for DISORDER723. Because DISORDER723 is highly imbalanced, AUPR, MCC, Fmax and calibration metrics are reported alongside ROC-AUC.

### NR25 leakage-control evaluation

Target-specific NR25 training sets were generated by removing DM3000 training proteins with MMseqs2 percentage identity above 25% against each target benchmark. The resulting training sets retained 2,824 proteins for SL329, 2,677 proteins for MXD494 and 2,576 proteins for DISORDER723. Full DM3000 and NR25 results are reported separately to avoid mixing standard benchmark performance with low-homology generalization claims.

### Sequence representation and RegionAwareTCN

Residue-level ESM2 embeddings were extracted and cached. The representation-upgrade experiment replaced ESM2-t12-35M layer 12 with ESM2-t33-650M layer 33 while keeping the RegionAwareTCN head, thresholding protocol and benchmark evaluation fixed. RegionAwareTCN applies temporal convolutional sequence modeling to frozen residue representations, residue one-hot features and relative-position features. It also includes auxiliary supervision for short versus long disordered regions and terminal versus internal disorder patterns. These auxiliary tasks provide a region-aware training and analysis scaffold, rather than a claim of uniform aggregate-metric improvement.

The final predictor averages scores from three independently trained ESM2-t33 RegionAwareTCN models with seeds 1, 2 and 3. The ensemble score is used for ranking metrics, validation-selected thresholding and calibration.

### Training, calibration and evaluation

Models were trained on DM3000 or target-specific NR25 training sets and selected using validation performance on DM1229. The binary decision threshold was selected on DM1229 by maximizing Fmax for the corresponding score distribution. Test labels were never used for threshold tuning or calibration fitting.

Post-hoc calibration was fitted using DM1229 validation predictions. Raw scores, temperature scaling, Platt scaling and isotonic regression were evaluated. Platt scaling was selected for the main calibrated output because it preserved ranking metrics while improving ECE, Brier score and NLL with lower overfitting risk than isotonic regression. Uncertainty was computed from calibrated binary predictive entropy.

Residue-level performance was evaluated with sensitivity (Sn), specificity (Sp), balanced accuracy (BACC), Matthews correlation coefficient (MCC), ROC-AUC, AUPR and Fmax. Calibration quality was evaluated with expected calibration error (ECE), Brier score and negative log-likelihood. Stratified analyses were performed for SDR/LDR, terminal/internal IDRs, residue zones, protein length bins and protein disorder-content bins. ESM2-t33 and ESM2-t12 RegionAwareTCN ensembles were compared with protein-level paired bootstrap confidence intervals and paired permutation tests. External SOTA methods were available only as aggregate published metrics, so they were compared as point references rather than by paired statistical tests.

## Results

### RegionAwareTCN reaches AUC-level SOTA point performance in the full DM3000 setting

Under the full DM3000 training protocol, the ESM2-t33 RegionAwareTCN ensemble reached AUC values of 0.919327, 0.850637 and 0.944611 on SL329, MXD494 and DISORDER723, respectively (Table 2 and Figure 2). These point estimates exceed the curated direct SOTA AUC values for all three benchmarks: IDP-EDL on SL329, FusionEncoder on MXD494 and IDP-EDL on DISORDER723. The corresponding point gains were +0.004327, +0.008637 and +0.001611.

Threshold-dependent metrics give a more restrained interpretation. On SL329, the model achieved MCC 0.736481, exceeding the curated IDP-EDL MCC of 0.700. On MXD494, it achieved MCC 0.499439, slightly above the curated FusionEncoder MCC of 0.492. On DISORDER723, MCC was 0.610462, below the curated IDP-EDL MCC of 0.636. Therefore, the primary performance claim is AUC-level SOTA point performance, not comprehensive dominance across all metrics.

### ESM2-t33 is the main driver of the DISORDER723 gain

Replacing ESM2-t12 with ESM2-t33 improved AUC point estimates on all external datasets. The AUC delta was +0.004055 on SL329, +0.004781 on MXD494 and +0.021479 on DISORDER723 (Table 2 and Supplementary Table S3). Protein-level paired statistics showed strong support for the DISORDER723 improvement, with a 95% bootstrap CI of 0.013868 to 0.028520 and paired permutation p=0.001996. SL329 and MXD494 had positive point deltas, but their protein-level confidence intervals crossed zero and should be interpreted as favorable but not statistically secure under conservative protein-level resampling.

This result connects to the prior failure analysis. Score smoothing, position removal and focal/asymmetric losses did not close the DISORDER723 gap, whereas a stronger frozen PLM representation substantially improved ranking performance. The largest gain occurred on the most difficult benchmark, supporting representation quality as the main lever for this stage of the project.

### NR25 evaluation defines the low-homology boundary

Target-specific NR25 training reduced performance relative to full DM3000 training (Table 3). The AUC drop was small on SL329 (-0.002266), larger on MXD494 (-0.016548) and moderate on DISORDER723 (-0.007736). Under NR25 training, RegionAwareTCN remained above the curated SL329 SOTA AUC but fell below the curated MXD494 and DISORDER723 SOTA AUC values. The appropriate claim is therefore competitive low-homology robustness, not uniformly low-homology SOTA.

### Hard-case stratification identifies internal IDRs as the remaining weakness

Hard-case stratification confirmed that IDR prediction difficulty is region dependent (Figure 3 and Supplementary Table S4). In DISORDER723, terminal IDR AUC was 0.968186, whereas internal IDR AUC was 0.887416 and internal MCC was 0.210033. The same qualitative pattern appears in other datasets, where internal IDRs are consistently harder than terminal IDRs. ESM2-t33 alleviates this failure mode but does not fully solve it.

### Platt calibration improves probability quality and uncertainty tracks errors

Platt calibration preserved AUC, AUPR and MCC while improving probability quality (Table 4). On DISORDER723, ECE decreased from 0.187936 to 0.017608 and Brier score decreased from 0.093709 to 0.032479. On MXD494, ECE decreased from 0.256921 to 0.100902 and Brier score decreased from 0.219616 to 0.146934.

Calibrated uncertainty was informative (Figure 4 and Supplementary Table S5). The top 10% most uncertain residues were enriched for errors by 2.57x on SL329, 2.51x on MXD494 and 6.96x on DISORDER723. This supports uncertainty-aware interpretation, especially for low-prevalence or annotation-sensitive benchmarks.

## Discussion

RegionAwareTCN supports a sequence-only, region-aware and calibrated framing for intrinsic disorder prediction. The final ESM2-t33 ensemble reaches AUC-level SOTA point performance on SL329, MXD494 and DISORDER723 under the full DM3000 protocol. The strongest evidence is not merely the final point estimate but the experimental chain: low-cost post-processing and loss variants did not close the DISORDER723 gap, whereas upgrading the frozen PLM representation substantially improved DISORDER723 ranking and internal-IDR behavior.

The study also clarifies the boundary of the performance claim. Protein-level paired statistics support a strong t33-over-t12 gain on DISORDER723, but the corresponding gains on SL329 and MXD494 are smaller and not significant under conservative protein-level resampling. External SOTA methods do not provide residue-level predictions, so direct paired tests against them are not possible. For this reason, the manuscript claims AUC-level SOTA point performance rather than broad statistical dominance over all existing predictors.

The NR25 experiments are a strength because they separate standard benchmark performance from low-homology generalization. The t33 model remains competitive under NR25 training and remains above the SL329 SOTA AUC, but MXD494 and DISORDER723 fall below the curated SOTA values. This limitation should be stated explicitly. It improves credibility and positions the work as leakage-aware rather than benchmark-optimized.

Calibration is a practical contribution. Platt scaling markedly improves ECE and Brier score, especially on MXD494 and DISORDER723, without changing AUC, AUPR or MCC. Calibrated uncertainty also enriches prediction errors, supporting a caution signal for ambiguous residues, disorder boundaries and low-prevalence datasets.

Internal IDRs remain the major unsolved technical problem. Although ESM2-t33 improved DISORDER723 internal AUC, internal MCC remained low. This suggests that internal disorder may require additional signals, such as function-aware labels, binding-region annotations, MoRF labels or specialized hard-example training. Structure features should not be added casually because local audits found no controlled structure-feature coverage and several benchmarks contain PDB-chain-like identifiers, creating leakage risk.

Overall, the strongest manuscript position is not that a new neural network solves IDR prediction. The stronger contribution is a transparent and calibrated PLM-era benchmark framework that achieves strong full-training performance, exposes low-homology and internal-IDR limitations, and provides probability estimates suitable for uncertainty-aware use.

## Conclusion

RegionAwareTCN provides a sequence-only, region-aware and calibrated framework for residue-level intrinsic disorder prediction. The ESM2-t33 three-seed ensemble achieves AUC-level SOTA point performance on three external benchmarks under the full DM3000 protocol, with the strongest statistically supported representation-upgrade gain on DISORDER723. The model remains competitive but not uniformly SOTA under target-specific NR25 evaluation. Platt calibration substantially improves probability quality and calibrated uncertainty enriches prediction errors. The main remaining challenges are internal IDR classification, DISORDER723 MCC and low-homology generalization on MXD494 and DISORDER723.

## Data Availability

Data and trained-model release details must be finalized before submission. The current project contains labeled FASTA files under `data/`, prediction files under `predictions/`, result tables under `results/`, figures under `figures/`, and manuscript assets under `manuscript/`. The public release should include trained model weights, environment files, prediction scripts, evaluation scripts, calibration scripts and clear instructions for regenerating or downloading ESM2 embeddings.

## Funding

To be filled.

## Conflict of Interest

To be filled.

## References

References are currently placeholders and must be converted to journal style before submission. Key sources include IDP-EDL (Briefings in Bioinformatics, 2025), FusionEncoder (Bioinformatics, 2025), CAID3 (Proteins, 2026), a 2026 review on modern intrinsic disorder prediction resources, IDP-Fusion, SPOT-Disorder2, IDP-LM, DR-BERT, DisoFLAG, DisorderUnetLM, PUNCH2, PredIDR2, ESMDisPred and flDPnn3.

## Main Tables

### Table 1. Dataset and NR25 leakage-control summary

{table1}

### Table 2. Full DM3000 benchmark performance

{table2}

### Table 3. Target-specific NR25 low-homology robustness

{table3}

### Table 4. Platt calibration and uncertainty-error enrichment

{table5}

## Main Figure List

Figure 1. Evidence chain and model framing. Source: `figures/p5/P5_EVIDENCE_CHAIN_MECHANISM.pdf`

Figure 2. ROC and precision-recall curves for the representation upgrade. Source: `figures/p5/P5_T12_T33_ROC_PR_CURVES.pdf`

Figure 3. Hard-case stratified AUC. Source: `figures/p5/P5_T33_HARD_CASE_STRATIFIED_AUC.pdf`

Figure 4. Calibrated uncertainty tracks prediction errors. Source: `figures/p5/P5_T33_PLATT_UNCERTAINTY_ERROR_ENRICHMENT.pdf`
"""


def build_supplement() -> str:
    table4 = read(TABLES / "Table4_key_ablation_deltas.md")
    hard_case = read(ROOT / "results/p5/P5_T33_HARD_CASE_STRATIFIED_SUMMARY.tsv")
    bootstrap = read(ROOT / "results/p5/P5_T33_AUC_BOOTSTRAP_CI.tsv")
    uncertainty = read(ROOT / "results/p5/P5_T33_PLATT_UNCERTAINTY_ERROR_ENRICHMENT.tsv")
    selected_cal = read(ROOT / "results/p4_6/P4_6_SELECTED_PLATT_CALIBRATION.tsv")

    return f"""# Supplementary Material Draft

## Supplementary Methods

### Region annotation

Short disordered regions were defined as continuous disordered segments shorter than 30 residues; long disordered regions were defined as segments of at least 30 residues. Terminal IDRs were segments near the N- or C-terminus using the project annotation rule, while the remaining disordered segments were treated as internal IDRs. Stratified analyses also included residue zones, protein length bins and disorder-content bins.

### Protein-level bootstrap and paired permutation

Protein-level resampling was used to avoid treating residues from the same protein as independent. Each bootstrap replicate sampled proteins with replacement and recomputed AUC for both the ESM2-t12 and ESM2-t33 RegionAwareTCN ensembles. Paired permutation randomly swapped t12 and t33 predictions within proteins under the null hypothesis of no paired model difference.

## Supplementary Table S1. Key paper-level ablation deltas

{table4}

## Supplementary Table S2. Protein-level AUC bootstrap and paired permutation

```tsv
{bootstrap}
```

## Supplementary Table S3. Hard-case stratified t33 performance

```tsv
{hard_case}
```

## Supplementary Table S4. Full Platt calibration metrics

```tsv
{selected_cal}
```

## Supplementary Table S5. Platt uncertainty-error enrichment

```tsv
{uncertainty}
```

## Supplementary Figures

Supplementary Figure S1. Reliability diagrams for raw, temperature-scaled, Platt-calibrated and isotonic-calibrated t33 ensemble predictions. Source files: `figures/p4_6/calibration/reliability_p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_*.pdf`.
"""


def build_target_memo() -> str:
    return """# P5.3 Target Journal Decision Memo

## Primary target

Bioinformatics, Original Paper.

Rationale:

- The current manuscript is an original computational method and benchmark study.
- Bioinformatics explicitly publishes original papers in computational biology and bioinformatics.
- The current evidence chain is method-focused, includes benchmark comparisons, reusable scripts and a potential software release.
- The conservative claim boundary fits a specialist methods journal better than a broad biology journal.

## Backup target

NAR Genomics and Bioinformatics or Briefings in Bioinformatics after reframing.

Briefings in Bioinformatics is not the primary target for the current draft because the official journal scope is more review/case/protocol oriented. It may become appropriate only if the manuscript is reframed as a broad benchmark/resource review with substantial narrative synthesis.

## Formatting assumptions used

- Main text targeted around a concise original-paper structure.
- Main manuscript keeps four main tables and four main figures.
- Full ablations, bootstrap details and hard-case TSVs are moved to supplementary material.
- References are currently placeholders and must be formatted in the selected journal style.

## Official links checked

- Bioinformatics author guidelines: https://academic.oup.com/bioinformatics/pages/author-guidelines
- Briefings in Bioinformatics instructions: https://academic.oup.com/bib/pages/General_Instructions
"""


def build_cover_letter() -> str:
    return """# Cover Letter Draft

Dear Editors,

We are pleased to submit our manuscript, "RegionAwareTCN: sequence-only, region-aware and uncertainty-calibrated intrinsic disorder prediction with protein language model representations", for consideration as an Original Paper in Bioinformatics.

Intrinsic disorder prediction has rapidly moved into the protein-language-model era, but performance claims remain difficult to interpret when benchmarks differ in disorder prevalence, annotation type, unknown labels and sequence similarity to training data. In this manuscript, we present RegionAwareTCN, a sequence-only residue-level predictor that combines frozen ESM2-t33 representations, temporal convolutional sequence modeling, region-aware auxiliary supervision and validation-fitted Platt calibration.

The manuscript contributes a conservative but useful performance and reliability evidence chain. Under the full DM3000 training protocol, RegionAwareTCN achieves AUC-level state-of-the-art point performance on SL329, MXD494 and DISORDER723. The strongest statistically supported representation-upgrade effect is observed on DISORDER723, where ESM2-t33 improves AUC by 0.021479 over the ESM2-t12 RegionAwareTCN ensemble under protein-level paired resampling. We also report target-specific NR25 low-homology evaluation, hard-case stratification and calibrated uncertainty analyses. These experiments expose the remaining limitations of internal IDR prediction and low-homology generalization while providing calibrated probabilities suitable for uncertainty-aware use.

We believe this work will interest Bioinformatics readers because it combines a practical sequence-only predictor with a transparent evaluation design that addresses common concerns in PLM-based benchmark studies: leakage control, hard-case behavior, class imbalance and probability calibration.

The manuscript has not been published elsewhere and is not under consideration by another journal. All authors have approved the submission. Data and code availability details will be finalized before submission.

Sincerely,

To be filled.
"""


def build_prebuttal() -> str:
    return """# Reviewer Risk Prebuttal

## Risk 1: The method is not architecturally novel enough

Response strategy: frame the contribution as a calibrated, leakage-aware and hard-case-stratified PLM-era benchmark framework, not as a radically new neural architecture. Emphasize the sequence-only input design, NR25 split, protein-level statistics, hard-case analysis and calibration.

## Risk 2: AUC gains over SOTA are small

Response strategy: acknowledge this explicitly. The manuscript claims AUC-level SOTA point performance, not broad statistical dominance over external predictors. The strongest statistical evidence is t33 versus t12 on DISORDER723.

## Risk 3: NR25 is not uniformly SOTA

Response strategy: present NR25 as robustness evidence and a leakage-control boundary. State that the model remains competitive but does not uniformly beat SOTA under NR25.

## Risk 4: DISORDER723 MCC remains below IDP-EDL

Response strategy: keep MCC in the main table and state the limitation. Do not hide it. Use AUPR, Fmax, calibration and hard-case analysis to show additional value beyond a single MCC number.

## Risk 5: External SOTA predictions are unavailable for paired testing

Response strategy: explain that paired tests are only possible for local t12 versus t33 predictions. External methods are compared by curated aggregate metrics. Avoid claiming statistical superiority over external SOTA.

## Risk 6: Structure/function features were not used

Response strategy: explain the local audit. Structure features were not used because local coverage was zero and PDB-chain-like benchmark identifiers introduce leakage risk. Function heads are future work pending curated residue-level labels.

## Risk 7: Data and code availability

Response strategy: before submission, package scripts, environment, model weights, prediction files, evaluation scripts, calibration scripts and instructions for ESM2 embedding extraction or download.
"""


def build_checklist(main_text: str, supplement: str) -> str:
    return f"""# P5.3 Submission Checklist

## Current package

- Main submission draft: `manuscript/P5_3_Bioinformatics_submission_draft.md`
- Supplementary draft: `manuscript/supplementary/P5_3_SUPPLEMENTARY_MATERIAL_DRAFT.md`
- Cover letter: `manuscript/P5_3_COVER_LETTER_DRAFT.md`
- Target journal memo: `manuscript/P5_3_TARGET_JOURNAL_DECISION.md`
- Reviewer risk prebuttal: `manuscript/P5_3_REVIEWER_RISK_PREBUTTAL.md`
- Word count table: `manuscript/P5_3_WORD_COUNT.tsv`

## Approximate word counts

- Main draft words: {word_count(main_text)}
- Supplementary draft words: {word_count(supplement)}

## Must finish before submission

- Fill authors, affiliations and corresponding author.
- Convert placeholder references into formal journal style.
- Finalize data/code/model availability and repository URL.
- Decide whether model weights and ESM2 embedding cache can be publicly hosted.
- Add exact software versions, hardware and training runtime.
- Verify the latest direct SL329/MXD494/DISORDER723 SOTA one final time.
- Convert Markdown tables to journal-ready DOCX or LaTeX tables.
- Confirm figure resolution and journal file format requirements.
- Run final reproducibility smoke test from raw predictions to main tables.

## Claim-control checklist

- Do not claim uniformly significant superiority over external SOTA.
- Do not claim NR25 SOTA on MXD494 or DISORDER723.
- Do not claim structure-aware or function-aware modeling as a main contribution.
- Keep DISORDER723 MCC limitation in the main manuscript.
- Keep internal IDR limitation in Discussion.
"""


def write_word_counts(rows: list[tuple[str, str]]) -> None:
    path = MANUSCRIPT / "P5_3_WORD_COUNT.tsv"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["file", "words"])
        for name, text in rows:
            writer.writerow([name, word_count(text)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    main_text = build_submission_draft()
    supplement = build_supplement()
    write(MANUSCRIPT / "P5_3_Bioinformatics_submission_draft.md", main_text)
    write(SUPP / "P5_3_SUPPLEMENTARY_MATERIAL_DRAFT.md", supplement)
    write(MANUSCRIPT / "P5_3_TARGET_JOURNAL_DECISION.md", build_target_memo())
    write(MANUSCRIPT / "P5_3_COVER_LETTER_DRAFT.md", build_cover_letter())
    write(MANUSCRIPT / "P5_3_REVIEWER_RISK_PREBUTTAL.md", build_prebuttal())
    write(MANUSCRIPT / "P5_3_SUBMISSION_CHECKLIST.md", build_checklist(main_text, supplement))
    write_word_counts(
        [
            ("P5_3_Bioinformatics_submission_draft.md", main_text),
            ("supplementary/P5_3_SUPPLEMENTARY_MATERIAL_DRAFT.md", supplement),
            ("P5_3_COVER_LETTER_DRAFT.md", build_cover_letter()),
            ("P5_3_REVIEWER_RISK_PREBUTTAL.md", build_prebuttal()),
        ]
    )


if __name__ == "__main__":
    main()
