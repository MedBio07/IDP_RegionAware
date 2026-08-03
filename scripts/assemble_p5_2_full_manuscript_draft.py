#!/usr/bin/env python3
"""Build a single integrated English manuscript draft for P5.2."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIR = ROOT / "manuscript"
TABLE_DIR = MANUSCRIPT_DIR / "tables"


def read_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    return re.sub(r"^# .+?\n\n", "", text, count=1, flags=re.DOTALL).strip()


def read_table(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_draft() -> str:
    abstract = read_body(MANUSCRIPT_DIR / "ABSTRACT_DRAFT.md")
    discussion = read_body(MANUSCRIPT_DIR / "DISCUSSION_DRAFT.md")
    latest_lit = read_body(MANUSCRIPT_DIR / "LATEST_LITERATURE_CHECK.md")

    table1 = read_table(TABLE_DIR / "Table1_dataset_and_nr25_summary.md")
    table2 = read_table(TABLE_DIR / "Table2_full_benchmark_sota.md")
    table3 = read_table(TABLE_DIR / "Table3_nr25_robustness.md")
    table4 = read_table(TABLE_DIR / "Table4_key_ablation_deltas.md")
    table5 = read_table(TABLE_DIR / "Table5_calibration_uncertainty.md")
    figure_captions = read_body(MANUSCRIPT_DIR / "FIGURE_CAPTIONS.md")

    return f"""# RegionAwareTCN: a sequence-only, region-aware and uncertainty-calibrated framework for intrinsic disorder prediction

## Manuscript Status

Draft stage: P5.2 integrated English draft.

Claim boundary: this draft intentionally uses conservative language. It claims AUC-level state-of-the-art point performance under the full DM3000 training protocol, strong protein-level support for the ESM2-t33 upgrade on DISORDER723, and competitive but not uniformly state-of-the-art NR25 low-homology performance.

## Authors

To be filled.

## Abstract

{abstract}

## Keywords

intrinsic disorder prediction; intrinsically disordered regions; protein language model; ESM2; calibration; uncertainty; low-homology evaluation; RegionAwareTCN

## Introduction

Intrinsically disordered proteins and intrinsically disordered regions (IDRs) are central to many regulatory, signaling and interaction processes, yet they remain challenging for residue-level prediction. Unlike structured domains, disorder is not a single homogeneous state. Disordered residues may occur in short or long segments, at protein termini or in internal regions, and in proteins with very different overall disorder content. Benchmark labels also differ in origin and certainty, including DisProt-like annotations, missing-residue-derived annotations, NoX-style labels and unknown residues. These factors make global benchmark metrics informative but incomplete.

Recent IDR predictors increasingly rely on protein language models (PLMs), feature fusion, ensemble learning and region-aware training strategies. Methods such as IDP-EDL and FusionEncoder define strong direct competitors on the SL329, MXD494 and DISORDER723 benchmarks, while CAID-style evaluations emphasize that disorder prediction should be assessed with ranking metrics, threshold-dependent metrics, low-similarity subsets and functional/binding extensions. However, several issues remain underreported in many benchmark studies. First, full-training benchmark gains may be difficult to interpret without a low-homology leakage-control setting. Second, region-specific failure modes, especially terminal versus internal IDRs, can be hidden by aggregate AUC. Third, model scores are usually treated as ranking values rather than calibrated probabilities, limiting their practical use for experimental prioritization and manual review.

Here we present RegionAwareTCN, a sequence-only residue-level IDR prediction framework built around frozen ESM2 representations, temporal convolutional sequence modeling and auxiliary supervision for SDR/LDR and terminal/internal disorder patterns. The final model uses ESM2-t33-650M residue embeddings, one-hot and relative-position features, a three-seed score-averaged ensemble and validation-fitted Platt calibration. We deliberately keep the main model sequence-only and avoid PDB coordinates, experimentally missing residues, multiple-sequence alignments, profile features, AlphaFold confidence features and function labels as primary inputs. This design reduces potential leakage and simplifies deployment.

Our study makes five contributions. First, it evaluates a sequence-only PLM-based RegionAwareTCN under a unified train/validation/test protocol. Second, it separates full DM3000 benchmark results from target-specific NR25 low-homology training results. Third, it uses error-analysis-driven representation upgrading to show that ESM2-t33 closes the main DISORDER723 ranking gap that smoothing and loss reweighting did not solve. Fourth, it reports hard-case stratification across SDR/LDR, terminal/internal regions, residue zones, protein length bins and protein disorder-content bins. Fifth, it turns raw model scores into calibrated probabilities and evaluates ECE, Brier score, NLL and uncertainty-error enrichment.

## Materials and Methods

### Datasets and label handling

The study used DM3000 as the training set and DM1229 as the validation set. Three independent external benchmarks were used for final testing: SL329, MXD494 and DISORDER723 (Table 1). Labels were treated at residue level with three possible states: ordered, disordered and unknown. Unknown residues were excluded from all metric calculations. This primarily affects SL329, which contains 89,582 unknown residues. Thresholds and calibration parameters were selected only on DM1229 and were then fixed before test-set evaluation.

The external benchmarks differ substantially in disorder prevalence. Among known residues, SL329 has a disorder fraction of 0.435334, MXD494 has 0.224360 and DISORDER723 has 0.062845. DISORDER723 is therefore especially sensitive to class imbalance and requires AUPR, MCC and calibration metrics in addition to ROC-AUC.

### NR25 leakage-control evaluation

To evaluate low-homology robustness, target-specific NR25 training sets were generated by removing DM3000 training proteins with MMseqs2 percentage identity above 25% against each target benchmark. The resulting training sets retained 2,824 proteins for SL329, 2,677 proteins for MXD494 and 2,576 proteins for DISORDER723. Full DM3000 and NR25 results were reported separately to avoid mixing standard benchmark performance with low-homology generalization claims.

### Sequence representations

The main model is sequence-only. It does not use PDB coordinates, experimentally missing residues, multiple-sequence alignments, profile features, AlphaFold confidence features or function labels as primary inputs. Residue-level ESM2 embeddings were extracted and cached. The P4.6 representation-upgrade experiment replaced ESM2-t12-35M layer 12 with ESM2-t33-650M layer 33 while keeping the RegionAwareTCN head, thresholding protocol and benchmark evaluation fixed.

### RegionAwareTCN architecture

RegionAwareTCN applies temporal convolutional sequence modeling to frozen residue representations, residue one-hot features and relative-position features. The model includes auxiliary supervision for short versus long disordered regions and terminal versus internal disorder patterns. These auxiliary tasks are not presented as universally improving every aggregate metric. Instead, they provide a region-aware training and analysis scaffold that aligns model evaluation with known IDR heterogeneity.

The selected final predictor averages scores from three independently trained ESM2-t33 RegionAwareTCN models with seeds 1, 2 and 3. The ensemble score is used for ranking metrics, validation-selected thresholding and calibration.

### Training, model selection and thresholding

Models were trained on DM3000 or target-specific NR25 training sets and selected using validation performance on DM1229. The binary decision threshold was selected on DM1229 by maximizing Fmax for the corresponding raw or calibrated score distribution. Test labels were never used for threshold tuning, model selection or calibration fitting.

### Calibration and uncertainty

Post-hoc calibration was fitted using DM1229 validation predictions. Raw scores, temperature scaling, Platt scaling and isotonic regression were evaluated. Platt scaling was selected for the main calibrated output because it preserved ranking metrics while improving ECE, Brier score and NLL with lower overfitting risk than isotonic regression.

Uncertainty was computed from calibrated binary predictive entropy. To test whether uncertainty was informative, residues were ranked by uncertainty and the error rate among the most uncertain residues was compared with the overall error rate.

### Metrics and statistical testing

Residue-level performance was evaluated with sensitivity (Sn), specificity (Sp), balanced accuracy (BACC), Matthews correlation coefficient (MCC), ROC-AUC, AUPR and Fmax. Calibration quality was evaluated with expected calibration error (ECE), Brier score and negative log-likelihood (NLL). Stratified analyses were performed for SDR/LDR, terminal/internal IDRs, residue zones, protein length bins and protein disorder-content bins.

The ESM2-t33 and ESM2-t12 RegionAwareTCN ensembles were compared with protein-level paired bootstrap confidence intervals and paired permutation tests. Protein-level resampling was used because residues from the same protein are correlated. External SOTA methods were available only as aggregate published metrics, so they were compared as point references rather than by paired statistical tests.

## Results

The following sections summarize full benchmark performance, representation-upgrade effects, NR25 robustness, hard-case stratification, and calibration/uncertainty analyses.

### Full benchmark comparison

Under the full DM3000 training protocol, the ESM2-t33 RegionAwareTCN ensemble reached AUC values of 0.919327, 0.850637 and 0.944611 on SL329, MXD494 and DISORDER723, respectively (Table 2 and Figure 2). These point estimates exceed the curated direct SOTA AUC values for all three benchmarks: IDP-EDL on SL329, FusionEncoder on MXD494 and IDP-EDL on DISORDER723. The gains over the direct SOTA point AUC values were +0.004327, +0.008637 and +0.001611, respectively.

The threshold-dependent metrics provide a more nuanced view. On SL329, the model achieved MCC 0.736481, exceeding the curated IDP-EDL MCC of 0.700. On MXD494, it achieved MCC 0.499439, slightly above the curated FusionEncoder MCC of 0.492. On DISORDER723, however, MCC was 0.610462, below the curated IDP-EDL MCC of 0.636. Therefore, the most defensible performance claim is AUC-level SOTA point performance, not comprehensive dominance across all metrics.

### Representation upgrade from ESM2-t12 to ESM2-t33

The representation upgrade improved AUC point estimates on all external datasets. The t33-minus-t12 AUC delta was +0.004055 on SL329, +0.004781 on MXD494 and +0.021479 on DISORDER723 (Table 2 and Table 4). Protein-level paired statistics showed that the DISORDER723 improvement was strongly supported, with a 95% bootstrap CI of 0.013868 to 0.028520 and paired permutation p=0.001996. The SL329 and MXD494 deltas were positive but their protein-level CIs crossed zero, so these gains should be interpreted as favorable point improvements rather than statistically secure improvements under conservative protein-level resampling.

This result connects directly to the preceding error analysis. Low-cost post-processing and loss variants did not close the DISORDER723 gap, whereas a stronger frozen PLM representation substantially improved ranking performance. The largest gain was also observed on the most difficult benchmark, supporting the interpretation that representation quality is the main driver of the remaining full-benchmark performance improvement.

### Target-specific NR25 evaluation

NR25 training reduced performance relative to the full DM3000 setting (Table 3). The AUC drop was small on SL329 (-0.002266), larger on MXD494 (-0.016548) and moderate on DISORDER723 (-0.007736). Under NR25 training, the model remained above the curated SL329 SOTA AUC but fell below the curated MXD494 and DISORDER723 SOTA AUC values. This supports a transparent robustness claim: RegionAwareTCN remains competitive under low-homology evaluation, but it is not uniformly low-homology SOTA.

### Hard-case stratification

Hard-case stratification confirmed that IDR prediction difficulty is region dependent (Figure 3). In DISORDER723, terminal IDR AUC was 0.968186, whereas internal IDR AUC was 0.887416 and internal MCC was 0.210033. This gap indicates that ESM2-t33 alleviates but does not solve internal IDR classification. The same qualitative pattern appears in other datasets, where internal IDRs are consistently harder than terminal IDRs. These results justify keeping terminal/internal analysis as a central part of the manuscript rather than treating it as a supplemental diagnostic only.

### Calibration and uncertainty

Platt calibration preserved AUC, AUPR and MCC while improving probability quality (Table 5). The largest ECE improvements were observed on MXD494 and DISORDER723. On DISORDER723, ECE decreased from 0.187936 to 0.017608 and Brier score decreased from 0.093709 to 0.032479. On MXD494, ECE decreased from 0.256921 to 0.100902 and Brier score decreased from 0.219616 to 0.146934.

Calibrated uncertainty was also informative (Figure 4). The top 10% most uncertain residues were enriched for errors by 2.57x on SL329, 2.51x on MXD494 and 6.96x on DISORDER723. This supports uncertainty-aware use of the predictor, especially for low-prevalence or annotation-sensitive benchmarks where a single binary call may be misleading.

## Discussion

{discussion}

## Conclusion

RegionAwareTCN provides a sequence-only, region-aware and calibrated framework for residue-level intrinsic disorder prediction. The ESM2-t33 three-seed ensemble achieves AUC-level SOTA point performance on three external benchmarks under the full DM3000 protocol, with the strongest statistically supported representation-upgrade gain on DISORDER723. The model remains competitive but not uniformly SOTA under target-specific NR25 evaluation. Platt calibration substantially improves probability quality and calibrated uncertainty enriches prediction errors, making the framework more useful for practical interpretation than a raw ranking score alone. The main remaining challenges are internal IDR classification, DISORDER723 MCC and low-homology generalization on MXD494 and DISORDER723.

## Data and Code Availability

To be finalized before submission. Current project assets include labeled FASTA files under `data/`, prediction files under `predictions/`, result tables under `results/`, figures under `figures/`, and manuscript assets under `manuscript/`. The final public release should include trained model weights, embedding-generation instructions or cached-feature instructions, environment files, evaluation scripts and calibration scripts.

## Author Contributions

To be filled.

## Competing Interests

To be filled.

## Acknowledgements

To be filled.

## References Draft

- IDP-EDL. Briefings in Bioinformatics, 2025. https://academic.oup.com/bib/article/26/2/bbaf182/8116687
- FusionEncoder. Bioinformatics, 2025. https://academic.oup.com/bioinformatics/article/41/7/btaf362/8169326
- Critical Assessment of Protein Intrinsic Disorder Round 3 (CAID3). PubMed: https://pubmed.ncbi.nlm.nih.gov/40859602/
- CAID3 full text. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/
- Modern resources for intrinsic disorder predictions. 2026 review: https://link.springer.com/article/10.1007/s00018-026-06087-3
- Additional local references to formalize from `references/`: IDP-Fusion, SPOT-Disorder2, IDP-LM, DR-BERT, DisoFLAG, DisorderUnetLM, PUNCH2, PredIDR2, ESMDisPred and flDPnn3.

## Tables

{table1}

{table2}

{table3}

{table4}

{table5}

## Figure Captions

{figure_captions}

## Latest Literature Check

{latest_lit}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=MANUSCRIPT_DIR / "RegionAwareTCN_manuscript_draft.md",
        help="Output Markdown manuscript path.",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_draft(), encoding="utf-8")


if __name__ == "__main__":
    main()
