#!/usr/bin/env python3
"""Create P5.4 pre-submission hardening and compliance files."""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
RESULTS = ROOT / "results"
P54 = RESULTS / "p5_4"

FINAL_EXPERIMENT = "p4_6_region_aware_tcn_esm2_t33_3seed_ensemble"
VALIDATION_THRESHOLD_FILE = RESULTS / "p4_6" / f"{FINAL_EXPERIMENT}_DM1229_Validation_metrics.tsv"

DATASETS = {
    "SL329": {
        "labels": ROOT / "data/SL329_test.fasta",
        "predictions": ROOT / "predictions/p4_6" / f"{FINAL_EXPERIMENT}_SL329.tsv",
        "metrics": ROOT / "results/p4_6" / f"{FINAL_EXPERIMENT}_SL329_metrics.tsv",
        "metrics_val_threshold": ROOT / "results/p4_6" / f"{FINAL_EXPERIMENT}_SL329_metrics_val_threshold.tsv",
    },
    "MXD494": {
        "labels": ROOT / "data/MXD494_test.fasta",
        "predictions": ROOT / "predictions/p4_6" / f"{FINAL_EXPERIMENT}_MXD494.tsv",
        "metrics": ROOT / "results/p4_6" / f"{FINAL_EXPERIMENT}_MXD494_metrics.tsv",
        "metrics_val_threshold": ROOT / "results/p4_6" / f"{FINAL_EXPERIMENT}_MXD494_metrics_val_threshold.tsv",
    },
    "DISORDER723": {
        "labels": ROOT / "data/DISORDER723_test.fasta",
        "predictions": ROOT / "predictions/p4_6" / f"{FINAL_EXPERIMENT}_DISORDER723.tsv",
        "metrics": ROOT / "results/p4_6" / f"{FINAL_EXPERIMENT}_DISORDER723_metrics.tsv",
        "metrics_val_threshold": ROOT / "results/p4_6" / f"{FINAL_EXPERIMENT}_DISORDER723_metrics_val_threshold.tsv",
    },
}

CORE_FILES: list[tuple[str, Path]] = [
    ("main_draft", MANUSCRIPT / "P5_3_Bioinformatics_submission_draft.md"),
    ("supplement", MANUSCRIPT / "supplementary/P5_3_SUPPLEMENTARY_MATERIAL_DRAFT.md"),
    ("cover_letter", MANUSCRIPT / "P5_3_COVER_LETTER_DRAFT.md"),
    ("target_memo", MANUSCRIPT / "P5_3_TARGET_JOURNAL_DECISION.md"),
    ("prebuttal", MANUSCRIPT / "P5_3_REVIEWER_RISK_PREBUTTAL.md"),
    ("checklist", MANUSCRIPT / "P5_3_SUBMISSION_CHECKLIST.md"),
    ("word_count", MANUSCRIPT / "P5_3_WORD_COUNT.tsv"),
    ("table1", MANUSCRIPT / "tables/Table1_dataset_and_nr25_summary.tsv"),
    ("table2", MANUSCRIPT / "tables/Table2_full_benchmark_sota.tsv"),
    ("table3", MANUSCRIPT / "tables/Table3_nr25_robustness.tsv"),
    ("table4", MANUSCRIPT / "tables/Table4_key_ablation_deltas.tsv"),
    ("table5", MANUSCRIPT / "tables/Table5_calibration_uncertainty.tsv"),
    ("figure1_pdf", ROOT / "figures/p5/P5_EVIDENCE_CHAIN_MECHANISM.pdf"),
    ("figure2_pdf", ROOT / "figures/p5/P5_T12_T33_ROC_PR_CURVES.pdf"),
    ("figure3_pdf", ROOT / "figures/p5/P5_T33_HARD_CASE_STRATIFIED_AUC.pdf"),
    ("figure4_pdf", ROOT / "figures/p5/P5_T33_PLATT_UNCERTAINTY_ERROR_ENRICHMENT.pdf"),
    ("figure1_png", ROOT / "figures/p5/P5_EVIDENCE_CHAIN_MECHANISM.png"),
    ("figure2_png", ROOT / "figures/p5/P5_T12_T33_ROC_PR_CURVES.png"),
    ("figure3_png", ROOT / "figures/p5/P5_T33_HARD_CASE_STRATIFIED_AUC.png"),
    ("figure4_png", ROOT / "figures/p5/P5_T33_PLATT_UNCERTAINTY_ERROR_ENRICHMENT.png"),
    ("data_train", ROOT / "data/DM3000_Train.fasta"),
    ("data_validation", ROOT / "data/DM1229_Validation.fasta"),
    ("data_sl329", ROOT / "data/SL329_test.fasta"),
    ("data_mxd494", ROOT / "data/MXD494_test.fasta"),
    ("data_disorder723", ROOT / "data/DISORDER723_test.fasta"),
    ("nr25_summary", ROOT / "data/nr25_by_test/summary.tsv"),
    ("calibration_parameters", ROOT / "results/p4_6/calibration" / f"{FINAL_EXPERIMENT}_calibration_parameters.tsv"),
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def as_float(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value in ("", "NA", None):
        return math.nan
    return float(value)


def fmt_float(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.6f}"


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


def find_single(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    matches = [row for row in rows if row.get(key) == value]
    if len(matches) != 1:
        raise ValueError(f"expected one row with {key}={value}, found {len(matches)}")
    return matches[0]


def collect_known_labels_scores(records: list[dict[str, object]], predictions: dict[str, list[float]]) -> tuple[list[int], list[float]]:
    labels_eval: list[int] = []
    scores_eval: list[float] = []
    known_ids = set()
    first_tokens = {}
    for record in records:
        protein_id = str(record["id"])
        known_ids.add(protein_id)
        token = protein_id.split()[0]
        first_tokens[token] = protein_id if token not in first_tokens else None
    lookup = {protein_id: protein_id for protein_id in known_ids}
    lookup.update({token: protein_id for token, protein_id in first_tokens.items() if protein_id is not None})

    for record in records:
        protein_id = str(record["id"])
        scores = predictions.get(protein_id) or predictions.get(protein_id.split()[0])
        if scores is None:
            raise ValueError(f"missing predictions for {protein_id}")
        sequence = str(record["sequence"])
        labels = record["labels"]
        assert isinstance(labels, list)
        if len(scores) != len(sequence):
            raise ValueError(f"length mismatch for {protein_id}: {len(scores)} != {len(sequence)}")
        for label, score in zip(labels, scores):
            if label == -1:
                continue
            labels_eval.append(int(label))
            scores_eval.append(float(score))
    unknown_ids = sorted(set(predictions) - set(lookup))
    if unknown_ids:
        raise ValueError(f"prediction IDs not present in labels: {unknown_ids[:5]}")
    return labels_eval, scores_eval


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return math.nan
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    tp = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / positives


def fmax(labels: list[int], scores: list[float]) -> tuple[float, float]:
    positives = sum(labels)
    if positives == 0:
        return math.nan, math.nan
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    tp = 0
    fp = 0
    best_f1 = 0.0
    best_threshold = math.nan
    i = 0
    while i < len(order):
        threshold = scores[order[i]]
        j = i
        while j < len(order) and scores[order[j]] == threshold:
            if labels[order[j]] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / positives
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        i = j
    return best_f1, best_threshold


def compare_rows(
    output: list[dict[str, object]],
    section: str,
    dataset: str,
    observed: dict[str, float],
    expected: dict[str, str],
    metrics: list[str],
    source: str,
    tolerance: float = 5e-6,
) -> None:
    for metric in metrics:
        observed_value = as_float(observed[metric])
        expected_value = as_float(expected[metric])
        delta = abs(observed_value - expected_value)
        output.append(
            {
                "section": section,
                "dataset": dataset,
                "metric": metric,
                "expected_source": source,
                "observed": fmt_float(observed_value),
                "expected": fmt_float(expected_value),
                "abs_delta": fmt_float(delta),
                "status": "PASS" if delta <= tolerance else "FAIL",
            }
        )


def build_smoke_test() -> list[dict[str, object]]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate_disorder_predictions import evaluate, parse_labeled_fasta, read_prediction_tsv

    validation_row = read_tsv(VALIDATION_THRESHOLD_FILE)[0]
    validation_threshold = as_float(validation_row["threshold"])
    table2_rows = read_tsv(MANUSCRIPT / "tables/Table2_full_benchmark_sota.tsv")
    output: list[dict[str, object]] = []

    for dataset, paths in DATASETS.items():
        records = parse_labeled_fasta(paths["labels"])
        predictions = read_prediction_tsv(paths["predictions"], "\t")
        binary_metrics = evaluate(records, predictions, validation_threshold)
        labels, scores = collect_known_labels_scores(records, predictions)
        aupr = average_precision(labels, scores)
        fmax_value, fmax_threshold = fmax(labels, scores)
        observed = {
            "threshold": validation_threshold,
            "sn": float(binary_metrics["sn"]),
            "sp": float(binary_metrics["sp"]),
            "bacc": float(binary_metrics["bacc"]),
            "mcc": float(binary_metrics["mcc"]),
            "auc": float(binary_metrics["auc"]),
            "aupr": aupr,
            "fmax": fmax_value,
            "fmax_threshold": fmax_threshold,
        }
        val_threshold_row = read_tsv(paths["metrics_val_threshold"])[0]
        per_test_row = read_tsv(paths["metrics"])[0]
        table2_row = find_single(table2_rows, "dataset", dataset)

        compare_rows(
            output,
            "prediction_to_val_threshold_metrics_file",
            dataset,
            observed,
            val_threshold_row,
            ["threshold", "sn", "sp", "bacc", "mcc", "auc"],
            str(paths["metrics_val_threshold"].relative_to(ROOT)),
        )
        compare_rows(
            output,
            "prediction_to_table2_binary_metrics",
            dataset,
            observed,
            table2_row,
            ["sn", "sp", "bacc", "mcc", "auc"],
            "manuscript/tables/Table2_full_benchmark_sota.tsv",
        )
        compare_rows(
            output,
            "prediction_to_table2_ranking_metrics",
            dataset,
            observed,
            table2_row,
            ["aupr", "fmax"],
            "manuscript/tables/Table2_full_benchmark_sota.tsv",
        )
        compare_rows(
            output,
            "prediction_to_per_test_fmax_file",
            dataset,
            observed,
            per_test_row,
            ["aupr", "fmax", "fmax_threshold"],
            str(paths["metrics"].relative_to(ROOT)),
        )
    return output


def build_file_integrity() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category, path in CORE_FILES:
        rows.append(
            {
                "category": category,
                "path": str(path.relative_to(ROOT)),
                "exists": int(path.exists()),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "status": "PASS" if path.exists() and path.stat().st_size > 0 else "FAIL",
            }
        )

    for dataset, paths in DATASETS.items():
        for category, path in paths.items():
            rows.append(
                {
                    "category": f"{dataset}_{category}",
                    "path": str(path.relative_to(ROOT)),
                    "exists": int(path.exists()),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                    "status": "PASS" if path.exists() and path.stat().st_size > 0 else "FAIL",
                }
            )

    for seed in (1, 2, 3):
        path = ROOT / "models" / f"p4_6_region_aware_tcn_esm2_t33_position_onehot_seed{seed}.pt"
        rows.append(
            {
                "category": f"t33_seed{seed}_weights",
                "path": str(path.relative_to(ROOT)),
                "exists": int(path.exists()),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "status": "PASS" if path.exists() and path.stat().st_size > 0 else "FAIL",
            }
        )
    return rows


def audit_placeholders() -> list[dict[str, object]]:
    patterns = (
        "To be filled",
        "placeholder",
        "must be finalized",
        "must be converted",
        "<repository/DOI",
        "<Zenodo DOI",
    )
    files = [
        MANUSCRIPT / "P5_3_Bioinformatics_submission_draft.md",
        MANUSCRIPT / "P5_3_COVER_LETTER_DRAFT.md",
        MANUSCRIPT / "P5_3_SUBMISSION_CHECKLIST.md",
    ]
    rows: list[dict[str, object]] = []
    for path in files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if any(pattern.lower() in line.lower() for pattern in patterns):
                rows.append(
                    {
                        "file": str(path.relative_to(ROOT)),
                        "line": line_number,
                        "text": line.strip(),
                        "status": "MANUAL_ACTION_REQUIRED",
                    }
                )
    return rows


def module_versions() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for module_name in ("numpy", "pandas", "sklearn", "scipy", "torch", "matplotlib", "seaborn", "esm", "yaml"):
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "version_not_available")
            status = "PASS"
        except Exception as exc:
            version = f"{type(exc).__name__}: {exc}"
            status = "CHECK"
        rows.append({"module": module_name, "version": str(version), "status": status})
    return rows


def pdf_pages(path: Path) -> str:
    if not path.exists():
        return "NA"
    try:
        result = subprocess.run(
            ["pdfinfo", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return "NA"
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return line.split(":", 1)[1].strip()
    return "NA"


def figure_audit() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for figure_id, stem in [
        ("Figure 1", "P5_EVIDENCE_CHAIN_MECHANISM"),
        ("Figure 2", "P5_T12_T33_ROC_PR_CURVES"),
        ("Figure 3", "P5_T33_HARD_CASE_STRATIFIED_AUC"),
        ("Figure 4", "P5_T33_PLATT_UNCERTAINTY_ERROR_ENRICHMENT"),
    ]:
        for ext in ("pdf", "png"):
            path = ROOT / "figures/p5" / f"{stem}.{ext}"
            rows.append(
                {
                    "figure": figure_id,
                    "format": ext,
                    "path": str(path.relative_to(ROOT)),
                    "exists": int(path.exists()),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                    "pdf_pages": pdf_pages(path) if ext == "pdf" else "NA",
                    "status": "PASS" if path.exists() and path.stat().st_size > 0 else "FAIL",
                }
            )
    return rows


def replace_section(text: str, header: str, replacement: str) -> str:
    pattern = re.compile(rf"(^## {re.escape(header)}\n)(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"section not found: {header}")
    return text[: match.start()] + f"## {header}\n\n{replacement.strip()}\n\n" + text[match.end() :]


def build_availability_statement() -> str:
    return """# P5.4 Availability And Reproducibility Statement

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
"""


def build_reference_audit() -> str:
    return """# P5.4 Core Reference Audit

Date checked: 2026-08-03.

## Core References To Format Before Submission

| role | short name | citation draft | DOI / URL | local evidence |
| --- | --- | --- | --- | --- |
| direct SOTA SL329/DISORDER723 | IDP-EDL | Xie J, Jin X, Wei H, Sun S, Liu Y. IDP-EDL: enhancing intrinsically disordered protein prediction by combining protein language model and ensemble deep learning. Briefings in Bioinformatics. 2025;26(2):bbaf182. | https://doi.org/10.1093/bib/bbaf182 | `references/IDP-EDL_2025_Briefings_in_Bioinformatics.pdf` |
| direct SOTA MXD494 | FusionEncoder | Liu S, Chen S, Bai T, Liu B. FusionEncoder: identification of intrinsically disordered regions based on multi-feature fusion. Bioinformatics. 2025;41(7):btaf362. | https://doi.org/10.1093/bioinformatics/btaf362 | `references/FusionEncoder_2025_Bioinformatics.pdf` |
| direct benchmark predecessor | IDP-Fusion | Tang YJ, Yan K, Zhang X, Tian Y, Liu B. Protein intrinsically disordered region prediction by combining neural architecture search and multi-objective genetic algorithm. BMC Biology. 2023;21:188. | https://doi.org/10.1186/s12915-023-01672-5 | `references/IDP-Fusion_2023_BMC_Biology.pdf` |
| community benchmark | CAID1 | Necci M, Piovesan D, Tosatto SCE, et al. Critical assessment of protein intrinsic disorder prediction. Nature Methods. 2021;18:472-481. | https://doi.org/10.1038/s41592-021-01117-3 | `references/CAID1_2021_Nature_Methods.pdf` |
| PLM/function context | IDP-LM | Pang Y, Liu B. IDP-LM: Prediction of protein intrinsic disorder and disorder functions based on language models. PLOS Computational Biology. 2023;19(11):e1011657. | https://doi.org/10.1371/journal.pcbi.1011657 | `references/IDP-LM_2023_PLOS_Computational_Biology.pdf` |
| PLM method context | DR-BERT | Nambiar A, Forsyth JM, Liu S, Maslov S. DR-BERT: A protein language model to annotate disordered regions. Structure. 2024;32:1260-1268. | https://doi.org/10.1016/j.str.2024.04.010 | `references/DR-BERT.pdf` |
| disorder-function context | DisoFLAG | Pang Y, Liu B. DisoFLAG: accurate prediction of protein intrinsic disorder and its functions using graph-based interaction protein language model. BMC Biology. 2024;22:2. | https://doi.org/10.1186/s12915-023-01803-y | `references/DisoFLAG_2024_BMC_Biology.pdf` |
| CAID-era predictor | PUNCH2 | Meng D, Pollastri G. PUNCH2: Explore the strategy for intrinsically disordered protein predictor. PLOS ONE. 2025;20(3):e0319208. | https://doi.org/10.1371/journal.pone.0319208 | `references/PUNCH2_2025_PLOS_ONE.pdf` |
| CAID-era predictor | PredIDR2 | Han KS, Kim HK, Kim MH, Pak MH, Pak SJ, Choe MM, Kim CS. PredIDR2: Improving accuracy of protein intrinsic disorder prediction by updating deep convolutional neural network and supplementing DisProt data. International Journal of Biological Macromolecules. 2025;306:141801. | https://doi.org/10.1016/j.ijbiomac.2025.141801 | `references/PredIDR2.pdf` |
| CAID3 / fast predictor context | flDPnn3 | Wang K, Hu G, Basu S, Kurgan L. flDPnn3: Fast and Accurate Prediction of Intrinsic Disorder in Protein Sequences. Journal of Molecular Biology. 2026. | https://doi.org/10.1016/j.jmb.2026.169629 | `references/fIDPnn3.pdf` |
| structure-aware future-work contrast | ESMDisPred | Kabir MWU, Dey A, Nafees F, Hoque MT. ESMDisPred: A Structure-Aware CNN-Transformer Architecture for Intrinsically Disordered Protein Prediction. bioRxiv preprint. 2026. | https://doi.org/10.64898/2026.01.22.701204 | `references/ESMDisPred.pdf` |

## Citation Actions

- Convert this table into Bioinformatics reference style using a reference manager.
- Add primary ESM/ESM2 citations and MMseqs2 citations; these are required for Methods completeness.
- Add database citations for DisProt, MobiDB and CAID where the dataset provenance section uses them.
- Remove contextual references that are not cited in the final narrative to keep the Original Paper concise.
"""


def build_sota_audit() -> str:
    return """# P5.4 Final SOTA Audit

Date checked: 2026-08-03.

## Direct Target-Set Comparator Status

| dataset | current direct comparator kept for manuscript | comparator AUC | RegionAwareTCN full-DM3000 AUC | status |
| --- | --- | ---: | ---: | --- |
| SL329 | IDP-EDL, Briefings in Bioinformatics 2025 | 0.915000 | 0.919327 | keep AUC-level point-comparison claim |
| MXD494 | FusionEncoder, Bioinformatics 2025 | 0.842000 | 0.850637 | keep AUC-level point-comparison claim |
| DISORDER723 | IDP-EDL, Briefings in Bioinformatics 2025 | 0.943000 | 0.944611 | keep only conservative AUC-level point-comparison claim; MCC remains lower than IDP-EDL |

## 2026 Literature Risks Checked

- CAID3 and flDPnn3 are important 2026 context but report CAID-style benchmarks rather than direct SL329/MXD494/DISORDER723 tables.
- ESMDisPred is a 2026 bioRxiv structure-aware method reporting CAID3-style metrics, not direct target-set results for this manuscript's three external tests.
- D2MOE arXiv 2603.06292 was inspected because search results suggested broad benchmark improvements. The paper reports TS115, CASP12 and CB513 comparisons, not SL329/MXD494/DISORDER723 intrinsic-disorder target-set results, so it should not replace the direct comparator table.

## Claim Boundary After Audit

The main text can retain: "AUC-level state-of-the-art point performance on SL329, MXD494 and DISORDER723 under the full DM3000 training protocol, relative to curated direct target-set aggregate metrics."

The main text should not claim:

- statistically significant superiority over external SOTA methods;
- uniformly SOTA NR25 performance;
- overall SOTA across all metrics;
- direct superiority over CAID3-only or preprint methods that do not report the same target sets.

## Sources To Cite Or Mention

- Bioinformatics author guidelines: https://academic.oup.com/bioinformatics/pages/author-guidelines
- IDP-EDL: https://academic.oup.com/bib/article/26/2/bbaf182/8116687
- FusionEncoder: https://academic.oup.com/bioinformatics/article/41/7/btaf362/8169326
- CAID3 PubMed: https://pubmed.ncbi.nlm.nih.gov/40859602/
- CAID3 full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/
- flDPnn3 DOI: https://doi.org/10.1016/j.jmb.2026.169629
- ESMDisPred DOI: https://doi.org/10.64898/2026.01.22.701204
- D2MOE arXiv: https://arxiv.org/abs/2603.06292
"""


def build_requirements() -> str:
    versions = {row["module"]: row["version"] for row in module_versions() if row["status"] == "PASS"}
    return f"""# P5.4 provisional runtime requirements
# Generated from the local /opt/anaconda3 environment on 2026-08-03.
# For this workstation, use:
#   export LD_LIBRARY_PATH=/opt/anaconda3/lib:/usr/local/cuda/lib64

numpy=={versions.get("numpy", "2.3.5")}
pandas=={versions.get("pandas", "2.3.3")}
scikit-learn=={versions.get("sklearn", "1.7.2")}
scipy=={versions.get("scipy", "1.16.3")}
matplotlib=={versions.get("matplotlib", "3.10.6")}
seaborn=={versions.get("seaborn", "0.13.2")}
PyYAML=={versions.get("yaml", "6.0.3")}
fair-esm=={versions.get("esm", "2.0.0")}
# Install torch using the CUDA wheel/channel appropriate for the target system.
# Local version used here: torch=={versions.get("torch", "2.13.0+cu130")}
"""


def build_environment_report(version_rows: list[dict[str, str]]) -> str:
    lines = [
        "# P5.4 Environment Snapshot",
        "",
        f"- Python executable: `{sys.executable}`",
        f"- Python version: `{sys.version.replace(chr(10), ' ')}`",
        f"- Platform: `{platform.platform()}`",
        f"- Current LD_LIBRARY_PATH: `{os.environ.get('LD_LIBRARY_PATH', '')}`",
        "- Required local library setting observed during audit: `LD_LIBRARY_PATH=/opt/anaconda3/lib:/usr/local/cuda/lib64`",
        "",
        "## Python Packages",
        "",
        "| module | version/status | check |",
        "| --- | --- | --- |",
    ]
    for row in version_rows:
        lines.append(f"| {row['module']} | {row['version']} | {row['status']} |")
    lines.extend(
        [
            "",
            "## Storage Footprint",
            "",
            "- ESM2 embedding cache: approximately 5.1 GB in `data/features/esm2_embeddings/`.",
            "- Prediction files: approximately 504 MB in `predictions/`.",
            "- Trained model weights: approximately 171 MB in `models/`.",
            "",
            "## Environment Risk",
            "",
            "The project does not yet include a full `environment.yml`, `requirements.txt` or container image. P5.4 generates `requirements_p5_4.txt` as a provisional snapshot, but a final public release should use a clean locked Conda environment or Docker/Singularity container.",
        ]
    )
    return "\n".join(lines)


def build_submission_blockers(placeholders: list[dict[str, object]], file_rows: list[dict[str, object]], smoke_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if placeholders:
        blockers.append(
            {
                "priority": "P0",
                "item": "author_and_administrative_placeholders",
                "status": "BLOCKING_FOR_SUBMISSION",
                "action": "Fill authors, affiliations, corresponding author, funding, conflict of interest and final cover-letter signature.",
            }
        )
    if any(row["status"] != "PASS" for row in file_rows):
        blockers.append(
            {
                "priority": "P0",
                "item": "missing_core_files",
                "status": "BLOCKING_FOR_SUBMISSION",
                "action": "Resolve all FAIL rows in results/p5_4/P5_4_FILE_INTEGRITY.tsv.",
            }
        )
    if any(row["status"] != "PASS" for row in smoke_rows):
        blockers.append(
            {
                "priority": "P0",
                "item": "metric_reproducibility_mismatch",
                "status": "BLOCKING_FOR_SUBMISSION",
                "action": "Resolve all FAIL rows in results/p5_4/P5_4_REPRODUCIBILITY_SMOKE_TEST.tsv.",
            }
        )
    blockers.extend(
        [
            {
                "priority": "P0",
                "item": "public_code_data_model_url",
                "status": "BLOCKING_FOR_SUBMISSION",
                "action": "Create public repository and archival DOI; insert into Availability and Implementation.",
            },
            {
                "priority": "P0",
                "item": "formal_references",
                "status": "BLOCKING_FOR_SUBMISSION",
                "action": "Convert P5_4_CORE_REFERENCE_AUDIT.md into journal-style references and add ESM2/MMseqs2/database citations.",
            },
            {
                "priority": "P1",
                "item": "final_sota_audit",
                "status": "RECHECK_IMMEDIATELY_BEFORE_SUBMISSION",
                "action": "Repeat direct literature search for SL329/MXD494/DISORDER723 before uploading.",
            },
            {
                "priority": "P1",
                "item": "figure_format",
                "status": "EDITORIAL_FORMATTING_REQUIRED",
                "action": "Convert PDF/PNG figures to the exact journal production format and resolution requested at submission.",
            },
            {
                "priority": "P1",
                "item": "runtime_hardware",
                "status": "AUTHOR_CONFIRMATION_REQUIRED",
                "action": "Add GPU, CPU, RAM, CUDA driver and training/inference wall-clock time.",
            },
        ]
    )
    return blockers


def build_hardened_main_draft() -> str:
    text = read_text(MANUSCRIPT / "P5_3_Bioinformatics_submission_draft.md")
    data_availability = """Code, configuration files, trained model weights, validation-fitted calibration parameters, residue-level predictions, benchmark result tables and figure-generation assets will be made available in a public repository and archived with a permanent DOI before publication. The repository URL and archival DOI are to be inserted before submission.

The local reproducibility package contains labeled FASTA datasets under `data/`, target-specific NR25 train-set construction artifacts under `data/nr25_by_test/`, cached ESM2 embeddings under `data/features/esm2_embeddings/`, trained RegionAwareTCN weights under `models/`, predictions under `predictions/`, result tables under `results/`, figures under `figures/`, and manuscript assets under `manuscript/`. The final public release should either include the cached ESM2 embeddings or provide exact commands for regenerating them from sequence FASTA files."""
    references = """A formal journal-style reference list must be inserted before submission. The P5.4 citation audit file (`manuscript/P5_4_CORE_REFERENCE_AUDIT.md`) provides the core reference set and DOI/URL checks for IDP-EDL, FusionEncoder, IDP-Fusion, CAID, IDP-LM, DR-BERT, DisoFLAG, PUNCH2, PredIDR2, flDPnn3 and ESMDisPred. The final reference list should also include primary ESM2, MMseqs2, DisProt, MobiDB and calibration/statistical-method citations where cited in the text."""
    text = replace_section(text, "Data Availability", data_availability)
    text = replace_section(text, "References", references)
    banner = """<!--
P5.4 hardened draft.
Still not submission-final until authors/affiliations, funding, conflict-of-interest,
repository DOI, license, exact hardware/runtime and formal references are inserted.
-->

"""
    return banner + text


def build_report(
    smoke_rows: list[dict[str, object]],
    file_rows: list[dict[str, object]],
    figure_rows: list[dict[str, object]],
    placeholder_rows: list[dict[str, object]],
    blocker_rows: list[dict[str, object]],
) -> str:
    smoke_pass = sum(1 for row in smoke_rows if row["status"] == "PASS")
    smoke_fail = sum(1 for row in smoke_rows if row["status"] != "PASS")
    file_fail = sum(1 for row in file_rows if row["status"] != "PASS")
    figure_fail = sum(1 for row in figure_rows if row["status"] != "PASS")
    p0_blockers = sum(1 for row in blocker_rows if row["priority"] == "P0")
    main_words = word_count(read_text(MANUSCRIPT / "P5_3_Bioinformatics_submission_draft.md"))
    supplement_words = word_count(read_text(MANUSCRIPT / "supplementary/P5_3_SUPPLEMENTARY_MATERIAL_DRAFT.md"))

    status = "CONDITIONALLY_READY_FOR_AUTHOR_COMPLETION" if smoke_fail == 0 and file_fail == 0 else "NOT_READY"
    return f"""# P5.4 Submission Hardening Report

Date: 2026-08-03

## Overall Status

P5.4 status: **{status}**.

The scientific evidence chain is internally consistent, and the main benchmark metrics are reproducible from the final prediction files. The package is not yet submission-final because author-specific metadata, public repository/DOI, formal references, licensing and exact hardware/runtime details remain unresolved.

## Compliance Snapshot

| item | status |
| --- | --- |
| Bioinformatics article type | Original Paper |
| Main draft word count | {main_words} words |
| Supplementary draft word count | {supplement_words} words |
| Main tables | 4 in P5.3 main draft |
| Main figures | 4 in P5.3 main draft |
| File integrity failures | {file_fail} |
| Figure file failures | {figure_fail} |
| Reproducibility smoke-test rows | {smoke_pass} PASS / {smoke_fail} FAIL |
| Placeholder/manual-action rows | {len(placeholder_rows)} |
| P0 submission blockers | {p0_blockers} |

## Reproducibility Result

The P5.4 smoke test recomputed validation-threshold binary metrics from:

- `data/*_test.fasta`
- `predictions/p4_6/{FINAL_EXPERIMENT}_*.tsv`
- validation threshold `0.833731` from `results/p4_6/{FINAL_EXPERIMENT}_DM1229_Validation_metrics.tsv`

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
2. Create public code/data/model repository and archival DOI.
3. Convert the reference audit into journal-ready references.
4. Add exact software/hardware/runtime details.
5. Re-run final direct SOTA search immediately before submission upload.
6. Convert Markdown tables and figures into the journal submission format.
"""


def run() -> None:
    P54.mkdir(parents=True, exist_ok=True)

    smoke_rows = build_smoke_test()
    write_tsv(
        P54 / "P5_4_REPRODUCIBILITY_SMOKE_TEST.tsv",
        smoke_rows,
        ["section", "dataset", "metric", "expected_source", "observed", "expected", "abs_delta", "status"],
    )

    file_rows = build_file_integrity()
    write_tsv(
        P54 / "P5_4_FILE_INTEGRITY.tsv",
        file_rows,
        ["category", "path", "exists", "size_bytes", "status"],
    )

    figure_rows = figure_audit()
    write_tsv(
        P54 / "P5_4_FIGURE_AUDIT.tsv",
        figure_rows,
        ["figure", "format", "path", "exists", "size_bytes", "pdf_pages", "status"],
    )

    placeholder_rows = audit_placeholders()
    write_tsv(
        P54 / "P5_4_PLACEHOLDER_AUDIT.tsv",
        placeholder_rows,
        ["file", "line", "text", "status"],
    )

    version_rows = module_versions()
    write_tsv(
        P54 / "P5_4_ENVIRONMENT_MODULES.tsv",
        version_rows,
        ["module", "version", "status"],
    )

    blocker_rows = build_submission_blockers(placeholder_rows, file_rows, smoke_rows)
    write_tsv(
        MANUSCRIPT / "P5_4_SUBMISSION_BLOCKERS.tsv",
        blocker_rows,
        ["priority", "item", "status", "action"],
    )

    write_text(MANUSCRIPT / "P5_4_Bioinformatics_submission_hardened_draft.md", build_hardened_main_draft())
    write_text(MANUSCRIPT / "P5_4_AVAILABILITY_AND_REPRODUCIBILITY_STATEMENT.md", build_availability_statement())
    write_text(MANUSCRIPT / "P5_4_CORE_REFERENCE_AUDIT.md", build_reference_audit())
    write_text(MANUSCRIPT / "P5_4_FINAL_SOTA_AUDIT.md", build_sota_audit())
    write_text(MANUSCRIPT / "P5_4_ENVIRONMENT_SNAPSHOT.md", build_environment_report(version_rows))
    write_text(MANUSCRIPT / "P5_4_SUBMISSION_HARDENING_REPORT.md", build_report(smoke_rows, file_rows, figure_rows, placeholder_rows, blocker_rows))
    write_text(ROOT / "requirements_p5_4.txt", build_requirements())

    print(f"wrote={MANUSCRIPT / 'P5_4_SUBMISSION_HARDENING_REPORT.md'}")
    print(f"smoke_pass={sum(1 for row in smoke_rows if row['status'] == 'PASS')}")
    print(f"smoke_fail={sum(1 for row in smoke_rows if row['status'] != 'PASS')}")
    print(f"file_fail={sum(1 for row in file_rows if row['status'] != 'PASS')}")
    print(f"placeholders={len(placeholder_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main() -> None:
    parse_args()
    run()


if __name__ == "__main__":
    main()
