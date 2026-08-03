#!/usr/bin/env python3
"""Assemble manuscript-ready tables, captions, and narrative drafts for P5."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_disorder_predictions import average_precision, collect_known_labels_scores
from evaluate_disorder_predictions import parse_labeled_fasta, read_prediction_tsv


DATASETS = ("SL329", "MXD494", "DISORDER723")
LABELS = {
    "SL329": ROOT / "data/SL329_test.fasta",
    "MXD494": ROOT / "data/MXD494_test.fasta",
    "DISORDER723": ROOT / "data/DISORDER723_test.fasta",
}
T12_PREDICTIONS = {
    dataset: ROOT / f"predictions/fusion/p2_region_aware_tcn_3seed_ensemble_{dataset}.tsv"
    for dataset in DATASETS
}
T33_PREDICTIONS = {
    dataset: ROOT / f"predictions/p4_6/p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_{dataset}.tsv"
    for dataset in DATASETS
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_float(value: object) -> float:
    if value in (None, "", "NA"):
        return math.nan
    return float(value)


def fmt(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.6f}"
    return str(value)


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def markdown_table(rows: list[dict[str, object]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def write_md_table(path: Path, title: str, rows: list[dict[str, object]], fields: list[str], note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f"# {title}\n\n{markdown_table(rows, fields)}\n"
    if note:
        body += f"\n{note}\n"
    path.write_text(body, encoding="utf-8")


def row_by(rows: list[dict[str, str]], key: str, value: str, dataset: str | None = None) -> dict[str, str]:
    matches = [row for row in rows if row.get(key) == value]
    if dataset is not None:
        matches = [row for row in matches if row.get("dataset") == dataset]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {key}={value}, dataset={dataset}; found {len(matches)}")
    return matches[0]


def table1_dataset(rows: list[dict[str, str]], nr25_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    removed_by_dataset = {row["test_set"]: int(row["removed_train_records"]) for row in nr25_rows}
    kept_by_dataset = {row["test_set"]: int(row["kept_train_records"]) for row in nr25_rows}
    output: list[dict[str, object]] = []
    selected = [
        "DM3000_Train",
        "DM1229_Validation",
        "SL329",
        "MXD494",
        "DISORDER723",
        "DM3000_Train_nr25_vs_SL329",
        "DM3000_Train_nr25_vs_MXD494",
        "DM3000_Train_nr25_vs_DISORDER723",
    ]
    display_names = {
        "DM3000_Train": "DM3000 train",
        "DM1229_Validation": "DM1229 validation",
        "SL329": "SL329 test",
        "MXD494": "MXD494 test",
        "DISORDER723": "DISORDER723 test",
        "DM3000_Train_nr25_vs_SL329": "NR25 train vs SL329",
        "DM3000_Train_nr25_vs_MXD494": "NR25 train vs MXD494",
        "DM3000_Train_nr25_vs_DISORDER723": "NR25 train vs DISORDER723",
    }
    for dataset in selected:
        row = row_by(rows, "dataset", dataset)
        target = dataset.replace("DM3000_Train_nr25_vs_", "")
        output.append(
            {
                "dataset": display_names[dataset],
                "role": row["split_type"],
                "proteins": int(row["proteins"]),
                "residues": int(row["residues"]),
                "known_residues": int(row["known_residues"]),
                "disorder_residues": int(row["disordered"]),
                "unknown_residues": int(row["unknown"]),
                "disorder_fraction_known": as_float(row["disorder_rate_known"]),
                "sdr_segments": int(row["sdr_segments"]),
                "ldr_segments": int(row["ldr_segments"]),
                "terminal_segments": int(row["terminal_segments"]),
                "internal_segments": int(row["internal_segments"]),
                "nr25_removed_train_proteins": removed_by_dataset.get(target, "NA"),
                "nr25_kept_train_proteins": kept_by_dataset.get(target, "NA"),
            }
        )
    return output


def table2_main_benchmark(comparison_rows: list[dict[str, str]], stats_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for dataset in DATASETS:
        t33 = row_by(comparison_rows, "variant", "p4_6_region_aware_tcn_esm2_t33_3seed_ensemble", dataset)
        t12 = row_by(comparison_rows, "variant", "p2_region_aware_tcn_3seed_ensemble", dataset)
        stats = row_by(stats_rows, "dataset", dataset)
        output.append(
            {
                "dataset": dataset,
                "method": "RegionAwareTCN + ESM2-t33, 3-seed ensemble",
                "sn": as_float(t33["sn"]),
                "sp": as_float(t33["sp"]),
                "bacc": as_float(t33["bacc"]),
                "mcc": as_float(t33["mcc"]),
                "auc": as_float(t33["auc"]),
                "aupr": as_float(t33["aupr"]),
                "fmax": as_float(t33["fmax"]),
                "sota_method": t33["sota_method"],
                "sota_auc": as_float(t33["sota_auc"]),
                "auc_gap_vs_sota": as_float(t33["auc_gap_vs_sota"]),
                "t12_auc": as_float(t12["auc"]),
                "t33_minus_t12_auc": as_float(stats["auc_delta"]),
                "delta_ci_95": f"{as_float(stats['auc_delta_ci_low']):.6f} to {as_float(stats['auc_delta_ci_high']):.6f}",
                "paired_permutation_p": as_float(stats["paired_permutation_p_one_sided"]),
            }
        )
    return output


def table3_nr25(nr25_rows: list[dict[str, str]], full_rows: list[dict[str, str]], nr25_summary: list[dict[str, str]]) -> list[dict[str, object]]:
    summary_by_dataset = {row["test_set"]: row for row in nr25_summary}
    output: list[dict[str, object]] = []
    for dataset in DATASETS:
        nr25 = row_by(nr25_rows, "dataset", dataset)
        full = row_by(full_rows, "variant", "p4_6_region_aware_tcn_esm2_t33_3seed_ensemble", dataset)
        summary = summary_by_dataset[dataset]
        output.append(
            {
                "dataset": dataset,
                "removed_train_proteins": int(summary["removed_train_records"]),
                "kept_train_proteins": int(summary["kept_train_records"]),
                "full_auc": as_float(full["auc"]),
                "nr25_auc": as_float(nr25["auc"]),
                "auc_delta_nr25_minus_full": as_float(nr25["auc"]) - as_float(full["auc"]),
                "nr25_aupr": as_float(nr25["aupr"]),
                "full_mcc": as_float(full["mcc"]),
                "nr25_mcc": as_float(nr25["mcc"]),
                "mcc_delta_nr25_minus_full": as_float(nr25["mcc"]) - as_float(full["mcc"]),
                "sota_auc": as_float(nr25["sota_auc"]),
                "nr25_gap_vs_sota": as_float(nr25["auc_gap_vs_sota"]),
            }
        )
    return output


def table4_ablation(delta_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in delta_rows:
        output.append(
            {
                "axis": row["comparison_axis"],
                "dataset": row["dataset"],
                "comparison": row["comparison"],
                "auc_delta": as_float(row["auc_delta"]),
                "aupr_delta": as_float(row["aupr_delta"]),
                "mcc_delta": as_float(row["mcc_delta"]),
                "ece_delta": as_float(row["ece_delta"]),
                "brier_delta": as_float(row["brier_delta"]),
                "interpretation": row["interpretation"],
            }
        )
    return output


def table5_calibration_uncertainty(cal_rows: list[dict[str, str]], uncertainty_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    enrich_by_dataset = {
        row["dataset"]: row
        for row in uncertainty_rows
        if row["dataset"] in DATASETS and row["top_uncertain_fraction"] == "0.100000"
    }
    output: list[dict[str, object]] = []
    for row in cal_rows:
        if row["dataset"] not in DATASETS:
            continue
        enrich = enrich_by_dataset[row["dataset"]]
        output.append(
            {
                "dataset": row["dataset"],
                "auc": as_float(row["auc"]),
                "aupr": as_float(row["aupr"]),
                "mcc": as_float(row["mcc"]),
                "raw_ece": as_float(row["raw_ece"]),
                "platt_ece": as_float(row["platt_ece"]),
                "ece_delta": as_float(row["ece_delta"]),
                "raw_brier": as_float(row["raw_brier"]),
                "platt_brier": as_float(row["platt_brier"]),
                "brier_delta": as_float(row["brier_delta"]),
                "top10_uncertain_error_enrichment": as_float(enrich["error_enrichment"]),
            }
        )
    return output


def collect_curve_data(prediction_paths: dict[str, Path]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for dataset, prediction_path in prediction_paths.items():
        records = parse_labeled_fasta(LABELS[dataset])
        predictions = read_prediction_tsv(prediction_path, "\t")
        labels, scores = collect_known_labels_scores(records, predictions)
        output[dataset] = (labels.astype(np.int8, copy=False), scores.astype(np.float64, copy=False))
    return output


def plot_roc_pr(out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    t12 = collect_curve_data(T12_PREDICTIONS)
    t33 = collect_curve_data(T33_PREDICTIONS)
    colors = {"t12": "#64748b", "t33": "#0f766e"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for col, dataset in enumerate(DATASETS):
        for label, data, color in [
            ("ESM2-t12", t12[dataset], colors["t12"]),
            ("ESM2-t33", t33[dataset], colors["t33"]),
        ]:
            y, s = data
            fpr, tpr, _ = roc_curve(y, s)
            roc_auc_value = sklearn_auc(fpr, tpr)
            axes[0, col].plot(fpr, tpr, linewidth=1.6, color=color, label=f"{label} AUC={roc_auc_value:.3f}")
            precision, recall, _ = precision_recall_curve(y, s)
            aupr_value = average_precision(y, s)
            axes[1, col].plot(recall, precision, linewidth=1.6, color=color, label=f"{label} AUPR={aupr_value:.3f}")
        axes[0, col].plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1)
        axes[0, col].set_title(dataset)
        axes[0, col].set_xlabel("False positive rate")
        axes[0, col].set_ylabel("True positive rate")
        axes[0, col].legend(frameon=False, fontsize=8, loc="lower right")
        axes[1, col].set_xlabel("Recall")
        axes[1, col].set_ylabel("Precision")
        axes[1, col].legend(frameon=False, fontsize=8, loc="lower left")
        axes[0, col].set_xlim(0, 1)
        axes[0, col].set_ylim(0, 1)
        axes[1, col].set_xlim(0, 1)
        axes[1, col].set_ylim(0, 1)
    fig.suptitle("Representation upgrade on external IDR benchmarks", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "P5_T12_T33_ROC_PR_CURVES.pdf")
    fig.savefig(out_dir / "P5_T12_T33_ROC_PR_CURVES.png", dpi=300)
    plt.close(fig)


def write_figure_captions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# Figure Captions Draft

## Figure 1. Evidence chain and model framing

Schematic overview of the manuscript evidence chain. P4.5 error analysis identified DISORDER723 internal IDRs as the major remaining failure mode; P4.6 upgraded frozen sequence representations from ESM2-t12 to ESM2-t33 while keeping the sequence-only RegionAwareTCN head fixed; P5 evaluates the resulting model with full DM3000 training, target-specific NR25 training, hard-case stratification, and validation-fitted Platt calibration.

Source asset: `figures/p5/P5_EVIDENCE_CHAIN_MECHANISM.pdf`

## Figure 2. ROC and precision-recall curves for the representation upgrade

ROC and precision-recall curves comparing the ESM2-t12 and ESM2-t33 RegionAwareTCN 3-seed ensembles on SL329, MXD494, and DISORDER723. The t33 representation improves ranking metrics on all three datasets, with the largest AUC and AUPR gain on DISORDER723.

Source asset: `figures/p5/P5_T12_T33_ROC_PR_CURVES.pdf`

## Figure 3. Hard-case stratified performance

AUC of the ESM2-t33 RegionAwareTCN 3-seed ensemble across overall, SDR, LDR, terminal IDR, internal IDR, middle-residue, and low-disorder-content strata. The plot highlights that internal IDRs remain substantially harder than terminal IDRs, especially in DISORDER723.

Source asset: `figures/p5/P5_T33_HARD_CASE_STRATIFIED_AUC.pdf`

## Figure 4. Calibrated uncertainty tracks prediction errors

Error enrichment among the most uncertain residues after validation-fitted Platt calibration. Top-10% uncertainty residues are enriched for errors by 2.57x on SL329, 2.51x on MXD494, and 6.96x on DISORDER723, supporting uncertainty-aware use of the predictor.

Source asset: `figures/p5/P5_T33_PLATT_UNCERTAINTY_ERROR_ENRICHMENT.pdf`

## Supplementary Figure. Reliability diagrams

Reliability diagrams for raw, temperature-scaled, Platt-calibrated, and isotonic-calibrated t33 ensemble predictions on DM1229 validation and the three external test sets.

Source assets: `figures/p4_6/calibration/reliability_p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_*.pdf`
"""
    path.write_text(text, encoding="utf-8")


def write_manuscript_blueprint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# Manuscript Blueprint Draft

## Working Title

RegionAwareTCN: sequence-only, region-aware and uncertainty-calibrated intrinsic disorder prediction with protein language model representations

## Central Claim

RegionAwareTCN combines frozen ESM2-t33 residue representations with region-aware sequence modeling and validation-fitted Platt calibration. In the full DM3000 benchmark setting, it achieves AUC-level SOTA point performance on SL329, MXD494 and DISORDER723. Under target-specific NR25 training it remains competitive but not uniformly SOTA, defining a transparent low-homology generalization boundary.

## Contributions

1. A sequence-only IDR predictor that avoids PDB coordinate, missing-residue, MSA/profile and function-label inputs in the main model.
2. Region-aware supervision and hard-case evaluation for SDR/LDR and terminal/internal disorder patterns.
3. Error-analysis-driven representation upgrade showing that ESM2-t33, rather than smoothing or loss reweighting, closes the DISORDER723 ranking gap.
4. Leakage-aware evaluation with both full DM3000 training and target-specific NR25 training.
5. Calibrated probability output with Platt scaling, reliability metrics, and uncertainty-error enrichment.

## Recommended Main Text Structure

### Introduction

- IDRs are functionally important and difficult to model because disorder is regionally and annotation-wise heterogeneous.
- Modern methods increasingly use PLMs, feature fusion, length-aware predictors, and CAID-style evaluations.
- Current gaps: low-homology leakage control, internal/terminal hard-case analysis, and calibrated probability quality are usually underreported.
- This work focuses on a sequence-only, region-aware and uncertainty-calibrated framework.

### Materials and Methods

- Datasets: DM3000 train, DM1229 validation, SL329, MXD494, DISORDER723.
- Label handling: mask `-1` residues in SL329; evaluate only known residues.
- NR25: remove DM3000 training proteins with MMseqs2 pident >25% against each target test set.
- Model: frozen ESM2-t33 embeddings, one-hot and relative position features, RegionAwareTCN head, SDR/LDR and terminal/internal auxiliary supervision, three-seed score averaging.
- Calibration: fit Platt scaling only on DM1229 validation predictions.
- Statistics: protein-level bootstrap confidence intervals and paired permutation tests for t33 versus t12.

### Results

1. The t33 RegionAwareTCN ensemble reaches AUC-level SOTA point performance in the full DM3000 setting.
2. Representation upgrade is the main driver, especially on DISORDER723.
3. NR25 evaluation reveals a realistic low-homology generalization gap.
4. Hard-case stratification shows internal IDRs remain difficult despite t33 improvement.
5. Platt calibration improves ECE/Brier/NLL without changing ranking metrics.
6. Calibrated uncertainty enriches prediction errors and flags residues needing caution.

### Discussion

- The project should avoid claiming broad statistical dominance over external SOTA because external per-residue predictions are unavailable.
- The strongest mechanistic claim is DISORDER723: t33 improves AUC by 0.021479 with protein-level CI entirely above zero.
- SL329/MXD494 full benchmark AUC point estimates are above SOTA, but t33-vs-t12 deltas are modest and not protein-level significant.
- NR25 results are valuable precisely because they expose the low-homology boundary.
- Future work: function-aware labels, CAID/blind validation, and internal IDR-specialized modeling.

## Claim Wording

Use:

> RegionAwareTCN achieves AUC-level state-of-the-art point performance under the full DM3000 training protocol and provides calibrated, uncertainty-aware probabilities, while remaining competitive under target-specific NR25 evaluation.

Avoid:

> RegionAwareTCN significantly outperforms all existing predictors across all datasets and low-homology settings.
"""
    path.write_text(text, encoding="utf-8")


def write_literature_check(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# Latest Literature Check Draft

Date checked: 2026-08-03

## Sources Checked

- IDP-EDL, Briefings in Bioinformatics 2025: https://academic.oup.com/bib/article/26/2/bbaf182/8116687
- FusionEncoder, Bioinformatics 2025: https://academic.oup.com/bioinformatics/article/41/7/btaf362/8169326
- CAID3, Proteins 2026 / PubMed: https://pubmed.ncbi.nlm.nih.gov/40859602/
- CAID3 full text / PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/
- 2026 review, Modern resources for intrinsic disorder predictions: https://link.springer.com/article/10.1007/s00018-026-06087-3

## Current Direct Benchmark Competitors

- IDP-EDL remains the direct local comparator for SL329 and DISORDER723 in the curated project tables.
- FusionEncoder remains the direct local comparator for MXD494 in the curated project tables.
- FusionEncoder also reports DISORDER723 and MXD494 benchmark results and CAID3 results, reinforcing that PLM-based semantic features and fusion remain current in 2025 methods.

## 2026 Context

- A 2026 review emphasizes that intrinsic disorder prediction is now shaped by protein language models, deep learning, meta-servers, and curated disorder/function databases.
- CAID3 confirms that current community evaluation increasingly emphasizes PLM-era predictors, challenge-style assessment, AUC/Fmax/average precision, and statistical comparisons such as DeLong tests.
- CAID3 also introduced a binding-region-in-IDR subchallenge, which supports treating disorder function prediction as an important future extension rather than a current main claim for this project.

## Manuscript Impact on This Project

The manuscript should position itself against PLM-era methods rather than older profile-only predictors. The direct benchmark claim should remain tied to SL329/MXD494/DISORDER723 point estimates from the curated local comparison table, while CAID3 and recent reviews should be used to motivate:

1. sequence-only and PLM-based representation,
2. transparent low-homology evaluation,
3. AUPR/Fmax reporting under class imbalance,
4. probability calibration and uncertainty as practical reliability features.

## Required Manual Citation Audit Before Submission

Before journal submission, manually verify whether any 2026 method has reported direct SL329, MXD494, or DISORDER723 results exceeding the current local SOTA table. If none is found, keep IDP-EDL and FusionEncoder as the direct benchmark comparators.
"""
    path.write_text(text, encoding="utf-8")


def write_results_narrative(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# Results Narrative Draft

## Full Benchmark Performance

The ESM2-t33 RegionAwareTCN three-seed ensemble reached AUC values of 0.919327, 0.850637 and 0.944611 on SL329, MXD494 and DISORDER723, respectively. These point estimates exceed the currently curated direct SOTA AUC values for all three external benchmarks. The performance claim should be framed as AUC-level SOTA point performance because the strongest external competitors do not provide paired residue-level predictions for direct statistical testing.

## Representation Upgrade

Replacing ESM2-t12 with ESM2-t33 increased AUC on all three external datasets. The gain was modest on SL329 (+0.004055) and MXD494 (+0.004781), but large on DISORDER723 (+0.021479). Protein-level paired resampling supports the DISORDER723 improvement strongly, with a 95% bootstrap CI of 0.013868 to 0.028520 and paired permutation p=0.001996.

## Low-Homology Evaluation

Target-specific NR25 training reduced performance on all three benchmarks relative to the full DM3000 setting. The drop was small on SL329 (-0.002266 AUC), larger on MXD494 (-0.016548), and moderate on DISORDER723 (-0.007736). Thus, the model remains competitive under low-homology evaluation but should not be described as uniformly low-homology SOTA.

## Hard-Case Stratification

The t33 ensemble maintains strong terminal IDR performance but internal IDRs remain difficult. On DISORDER723, terminal IDR AUC was 0.968186 whereas internal IDR AUC was 0.887416 and internal MCC was 0.210033. This supports the manuscript's hard-case narrative: representation upgrade alleviates but does not fully solve internal IDR prediction.

## Calibration and Uncertainty

Platt calibration preserved AUC, AUPR and MCC while reducing probability error. On DISORDER723, ECE dropped from 0.187936 to 0.017608 and Brier score from 0.093709 to 0.032479. Calibrated uncertainty tracked errors: the top 10% most uncertain residues were enriched for errors by 2.57x, 2.51x and 6.96x on SL329, MXD494 and DISORDER723.
"""
    path.write_text(text, encoding="utf-8")


def write_abstract_draft(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# Abstract Draft

Intrinsic disorder prediction has benefited from protein language models, but benchmark-level gains can be difficult to interpret because disorder annotations are regionally heterogeneous, test sets differ in disorder prevalence, and homologous training examples may inflate apparent performance. We developed RegionAwareTCN, a sequence-only residue-level predictor that combines frozen ESM2 representations with temporal convolutional sequence modeling and auxiliary supervision for short/long and terminal/internal disorder patterns. The final model uses ESM2-t33 embeddings, three-seed score averaging, validation-selected thresholds, and Platt calibration fitted only on the DM1229 validation set.

Across SL329, MXD494 and DISORDER723, the ESM2-t33 RegionAwareTCN ensemble achieved AUC values of 0.919327, 0.850637 and 0.944611, respectively, exceeding the curated direct SOTA point AUC values for all three benchmarks under the full DM3000 training protocol. Protein-level paired resampling showed that the ESM2-t33 upgrade produced the strongest statistically supported gain on DISORDER723, improving AUC by 0.021479 over the ESM2-t12 RegionAwareTCN ensemble (95% bootstrap CI 0.013868 to 0.028520; paired permutation p=0.001996). Target-specific NR25 evaluation showed competitive but not uniformly SOTA low-homology performance, with AUC values of 0.917061, 0.834089 and 0.936875 on SL329, MXD494 and DISORDER723. Platt calibration preserved ranking metrics while improving probability quality, reducing DISORDER723 ECE from 0.187936 to 0.017608 and Brier score from 0.093709 to 0.032479. Calibrated uncertainty enriched errors among the top 10% most uncertain residues by 2.57x, 2.51x and 6.96x on SL329, MXD494 and DISORDER723.

These results support a calibrated, leakage-aware and hard-case-stratified sequence-only framework for intrinsic disorder prediction, while also identifying internal IDRs and low-homology generalization as the major remaining limitations.
"""
    path.write_text(text, encoding="utf-8")


def write_methods_draft(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# Methods Draft

## Datasets

The study used DM3000 as the training set and DM1229 as the validation set. Three independent external benchmarks were used for final testing: SL329, MXD494 and DISORDER723. Labels were treated at residue level with three possible states: ordered, disordered and unknown. Unknown residues were excluded from all metric calculations. This primarily affects SL329, which contains 89,582 unknown residues. Thresholds and calibration parameters were selected only on DM1229 and were then fixed before test-set evaluation.

## NR25 Leakage-Control Evaluation

To evaluate low-homology robustness, target-specific NR25 training sets were generated by removing DM3000 training proteins with MMseqs2 percentage identity above 25% against each target benchmark. The resulting training sets retained 2,824 proteins for SL329, 2,677 proteins for MXD494 and 2,576 proteins for DISORDER723. Full DM3000 and NR25 results were reported separately to avoid mixing standard benchmark performance with low-homology generalization claims.

## Sequence Representations

The main model is sequence-only. It does not use PDB coordinates, experimentally missing residues, multiple-sequence alignments, profile features, AlphaFold confidence features or function labels as primary inputs. Residue-level ESM2 embeddings were extracted and cached. P4.6 upgraded the representation from ESM2-t12-35M layer 12 to ESM2-t33-650M layer 33 while keeping the RegionAwareTCN head and evaluation protocol fixed.

## RegionAwareTCN Architecture

RegionAwareTCN applies temporal convolutional sequence modeling to frozen residue representations, residue one-hot features and relative-position features. The model includes auxiliary supervision for short versus long disordered regions and terminal versus internal disorder patterns. The selected final predictor averages scores from three independently trained ESM2-t33 RegionAwareTCN models with seeds 1, 2 and 3.

## Training and Threshold Selection

Models were trained on DM3000 or target-specific NR25 training sets and selected using validation performance on DM1229. The binary decision threshold was selected on DM1229 by maximizing Fmax for the corresponding raw or calibrated score distribution. Test labels were never used for threshold tuning or calibration fitting.

## Calibration

Post-hoc calibration was fitted using DM1229 validation predictions. Raw scores, temperature scaling, Platt scaling and isotonic regression were evaluated. Platt scaling was selected for the main calibrated output because it preserved ranking metrics while improving ECE, Brier score and NLL with lower overfitting risk than isotonic regression.

## Metrics

Residue-level performance was evaluated with Sn, Sp, balanced accuracy, MCC, ROC-AUC, AUPR and Fmax. Calibration quality was evaluated with expected calibration error, Brier score and negative log-likelihood. Stratified analyses were performed for SDR/LDR, terminal/internal IDRs, residue zones, protein length bins and protein disorder-content bins.

## Statistical Testing

The ESM2-t33 and ESM2-t12 RegionAwareTCN ensembles were compared with protein-level paired bootstrap confidence intervals and paired permutation tests. Protein-level resampling was used because residues from the same protein are correlated. External SOTA methods were available only as aggregate published metrics, so they were compared as point references rather than by paired statistical tests.
"""
    path.write_text(text, encoding="utf-8")


def write_discussion_draft(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# Discussion Draft

This project supports a sequence-only, region-aware and calibrated framing for intrinsic disorder prediction. The final ESM2-t33 RegionAwareTCN ensemble reaches AUC-level SOTA point performance on SL329, MXD494 and DISORDER723 under the full DM3000 training protocol. Importantly, the strongest evidence is not merely the final point estimate but the experimental chain leading to it: P4.5 showed that smoothing, position removal and loss reweighting did not close the DISORDER723 gap, whereas P4.6 showed that upgrading the frozen PLM representation substantially improved DISORDER723 ranking and internal-IDR behavior.

The study also clarifies the boundary of the performance claim. Protein-level paired statistics support a strong t33-over-t12 gain on DISORDER723, but the corresponding gains on SL329 and MXD494 are smaller and not significant under conservative protein-level resampling. In addition, the external SOTA methods do not provide residue-level predictions, so direct paired tests against them are not possible. For this reason, the manuscript should claim AUC-level SOTA point performance rather than broad statistical dominance over all existing predictors.

The NR25 experiments are an important strength because they separate standard benchmark performance from low-homology generalization. The t33 model remains competitive under NR25 training and remains above the SL329 SOTA AUC, but MXD494 and DISORDER723 fall below the curated SOTA values. This limitation should be stated explicitly. It improves the credibility of the paper and helps position the work as leakage-aware rather than benchmark-optimized.

Calibration is a practical contribution. Platt scaling markedly improves ECE and Brier score, especially on MXD494 and DISORDER723, without changing AUC, AUPR or MCC. The uncertainty analysis further shows that calibrated uncertainty is biologically useful as a caution signal: residues with the highest uncertainty are much more likely to be prediction errors. This is particularly relevant for disorder boundaries, ambiguous annotations and low-prevalence datasets.

Internal IDRs remain the major unsolved technical problem. Although ESM2-t33 improved DISORDER723 internal AUC, internal MCC remained low. This suggests that internal disorder may require additional signals, such as function-aware labels, binding-region annotations, MoRF labels, or specialized hard-example training. Structure features should not be added casually because local audits found no controlled structure-feature coverage and several benchmarks contain PDB-chain-like identifiers, creating leakage risk.

Overall, the best manuscript position is not a generic claim that a new neural network solves disorder prediction. The stronger contribution is a transparent and calibrated PLM-era benchmark framework that achieves strong full-training performance, exposes low-homology and internal-IDR limitations, and provides probability estimates suitable for uncertainty-aware use.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    manuscript_dir = ROOT / "manuscript"
    table_dir = manuscript_dir / "tables"
    figure_dir = ROOT / "figures/p5"

    dataset_rows = read_tsv(ROOT / "results/dataset_region_summary.tsv")
    nr25_summary = read_tsv(ROOT / "data/nr25_by_test/summary.tsv")
    full_rows = read_tsv(ROOT / "results/p4_6/P4_6_REPRESENTATION_UPGRADE_COMPARISON.tsv")
    nr25_rows = read_tsv(ROOT / "results/p4_6/P4_6_NR25_COMPARISON.tsv")
    calibration_rows = read_tsv(ROOT / "results/p4_6/P4_6_SELECTED_PLATT_CALIBRATION.tsv")
    bootstrap_rows = read_tsv(ROOT / "results/p5/P5_T33_AUC_BOOTSTRAP_CI.tsv")
    delta_rows = read_tsv(ROOT / "results/p5/P5_PAPER_LEVEL_KEY_DELTAS.tsv")
    uncertainty_rows = read_tsv(ROOT / "results/p5/P5_T33_PLATT_UNCERTAINTY_ERROR_ENRICHMENT.tsv")

    table1 = table1_dataset(dataset_rows, nr25_summary)
    table2 = table2_main_benchmark(full_rows, bootstrap_rows)
    table3 = table3_nr25(nr25_rows, full_rows, nr25_summary)
    table4 = table4_ablation(delta_rows)
    table5 = table5_calibration_uncertainty(calibration_rows, uncertainty_rows)

    table_specs = [
        (
            "Table1_dataset_and_nr25_summary",
            "Table 1. Dataset and NR25 leakage-control summary",
            table1,
            [
                "dataset",
                "role",
                "proteins",
                "known_residues",
                "disorder_residues",
                "unknown_residues",
                "disorder_fraction_known",
                "sdr_segments",
                "ldr_segments",
                "terminal_segments",
                "internal_segments",
                "nr25_removed_train_proteins",
                "nr25_kept_train_proteins",
            ],
            "Note: SL329 contains unknown labels and all `-1` residues are masked during evaluation.",
        ),
        (
            "Table2_full_benchmark_sota",
            "Table 2. Full DM3000 benchmark performance",
            table2,
            [
                "dataset",
                "method",
                "sn",
                "sp",
                "bacc",
                "mcc",
                "auc",
                "aupr",
                "fmax",
                "sota_method",
                "sota_auc",
                "auc_gap_vs_sota",
                "t12_auc",
                "t33_minus_t12_auc",
                "delta_ci_95",
                "paired_permutation_p",
            ],
            "Note: protein-level paired statistics compare t33 against local t12 predictions, not against external aggregate SOTA methods.",
        ),
        (
            "Table3_nr25_robustness",
            "Table 3. Target-specific NR25 low-homology robustness",
            table3,
            [
                "dataset",
                "removed_train_proteins",
                "kept_train_proteins",
                "full_auc",
                "nr25_auc",
                "auc_delta_nr25_minus_full",
                "nr25_aupr",
                "full_mcc",
                "nr25_mcc",
                "mcc_delta_nr25_minus_full",
                "sota_auc",
                "nr25_gap_vs_sota",
            ],
            "Note: NR25 results are reported as robustness evidence and are not uniformly SOTA.",
        ),
        (
            "Table4_key_ablation_deltas",
            "Table 4. Paper-level key ablation deltas",
            table4,
            [
                "axis",
                "dataset",
                "comparison",
                "auc_delta",
                "aupr_delta",
                "mcc_delta",
                "ece_delta",
                "brier_delta",
                "interpretation",
            ],
            "",
        ),
        (
            "Table5_calibration_uncertainty",
            "Table 5. Platt calibration and uncertainty-error enrichment",
            table5,
            [
                "dataset",
                "auc",
                "aupr",
                "mcc",
                "raw_ece",
                "platt_ece",
                "ece_delta",
                "raw_brier",
                "platt_brier",
                "brier_delta",
                "top10_uncertain_error_enrichment",
            ],
            "Note: Platt calibration is fitted only on DM1229 validation predictions.",
        ),
    ]

    for basename, title, rows, fields, note in table_specs:
        write_tsv(table_dir / f"{basename}.tsv", rows, fields)
        write_md_table(table_dir / f"{basename}.md", title, rows, fields, note)

    plot_roc_pr(figure_dir)
    write_figure_captions(manuscript_dir / "FIGURE_CAPTIONS.md")
    write_abstract_draft(manuscript_dir / "ABSTRACT_DRAFT.md")
    write_manuscript_blueprint(manuscript_dir / "MANUSCRIPT_BLUEPRINT.md")
    write_methods_draft(manuscript_dir / "METHODS_DRAFT.md")
    write_results_narrative(manuscript_dir / "RESULTS_NARRATIVE_DRAFT.md")
    write_discussion_draft(manuscript_dir / "DISCUSSION_DRAFT.md")
    write_literature_check(manuscript_dir / "LATEST_LITERATURE_CHECK.md")

    readme = """# P5 Manuscript Assembly

This directory contains manuscript-ready assets generated from completed P0-P5 outputs.

## Tables

- `tables/Table1_dataset_and_nr25_summary.md`
- `tables/Table2_full_benchmark_sota.md`
- `tables/Table3_nr25_robustness.md`
- `tables/Table4_key_ablation_deltas.md`
- `tables/Table5_calibration_uncertainty.md`

Each Markdown table has a matching TSV file for journal formatting.

## Draft Text

- `ABSTRACT_DRAFT.md`
- `MANUSCRIPT_BLUEPRINT.md`
- `METHODS_DRAFT.md`
- `RESULTS_NARRATIVE_DRAFT.md`
- `DISCUSSION_DRAFT.md`
- `FIGURE_CAPTIONS.md`
- `LATEST_LITERATURE_CHECK.md`

## Figures

Primary P5 figures are stored under `figures/p5/`.
"""
    (manuscript_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
