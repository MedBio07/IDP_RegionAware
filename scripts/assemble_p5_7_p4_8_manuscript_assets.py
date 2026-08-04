#!/usr/bin/env python3
"""Assemble manuscript-ready P5.7 assets after promoting the P4.8 model."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("SL329", "MXD494", "DISORDER723")
TABLE_DIR = ROOT / "manuscript/tables"
REPORT = ROOT / "reports/P5_7_P4_8_MANUSCRIPT_UPDATE.md"


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


def write_md(path: Path, title: str, note: str, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"# {title}\n\n{markdown_table(rows, fields)}\n\nNote: {note}\n"
    path.write_text(text, encoding="utf-8")


def plot_hard_case_gain(path_pdf: Path, path_png: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f"{row['dataset']}\n{row['stratum']}" for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    for offset, (metric, color) in enumerate(
        (("auc_delta", "#2563eb"), ("aupr_delta", "#16a34a"), ("mcc_delta", "#dc2626"))
    ):
        values = [float(row[metric]) for row in rows]
        ax.bar(x + (offset - 1) * width, values, width=width, label=metric.replace("_delta", "").upper(), color=color)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("P4.8 minus P4.6")
    ax.set_title("Warm-start adapter hard-case gains")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_pdf)
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def row_by(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        detail = ", ".join(f"{key}={value}" for key, value in criteria.items())
        raise ValueError(f"expected one row for {detail}; found {len(matches)}")
    return matches[0]


def uncertainty_top10(rows: list[dict[str, str]], dataset: str) -> dict[str, str]:
    return row_by(rows, method="platt", dataset=dataset, top_uncertain_fraction="0.100000")


def main() -> None:
    main_rows_raw = read_tsv(ROOT / "results/p4_8/P4_8_MAIN_BENCHMARK_REPLACEMENT.tsv")
    bootstrap_rows_raw = read_tsv(ROOT / "results/p4_8/P4_8_P4_7_WARM_VS_P4_6_BOOTSTRAP.tsv")
    nr25_rows_raw = read_tsv(ROOT / "results/p4_8/P4_8_NR25_REPLACEMENT.tsv")
    hard_rows_raw = read_tsv(ROOT / "results/p4_8/P4_8_HARD_CASE_REPLACEMENT.tsv")
    calibration_rows_raw = read_tsv(ROOT / "results/p4_8/P4_8_CALIBRATION_REPLACEMENT.tsv")
    uncertainty_rows_raw = read_tsv(
        ROOT
        / "results/p4_8/calibration/p4_8_region_adapter_moe_tcn_gate002_warm_3seed_ensemble_uncertainty_error_enrichment.tsv"
    )

    benchmark_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        candidate = row_by(main_rows_raw, dataset=dataset, variant="p4_8_region_adapter_moe_tcn_gate002_warm_3seed_ensemble")
        bootstrap = row_by(bootstrap_rows_raw, dataset=dataset)
        benchmark_rows.append(
            {
                "dataset": dataset,
                "method": "RegionAdapterMoETCN warm-start + Platt",
                "sn": as_float(candidate["sn"]),
                "sp": as_float(candidate["sp"]),
                "bacc": as_float(candidate["bacc"]),
                "mcc": as_float(candidate["mcc"]),
                "auc": as_float(candidate["auc"]),
                "aupr": as_float(candidate["aupr"]),
                "fmax": as_float(candidate["fmax"]),
                "sota_method": candidate["sota_method"],
                "sota_auc": as_float(candidate["sota_auc"]),
                "auc_gap_vs_sota": as_float(candidate["auc_gap_vs_sota"]),
                "p4_8_minus_p4_6_auc": as_float(candidate["delta_auc_vs_p4_6_platt"]),
                "p4_8_minus_p4_6_mcc": as_float(candidate["delta_mcc_vs_p4_6_platt"]),
                "auc_delta_ci_95": f"{as_float(bootstrap['auc_delta_ci_low']):.6f} to {as_float(bootstrap['auc_delta_ci_high']):.6f}",
                "paired_permutation_p": as_float(bootstrap["paired_permutation_p_one_sided"]),
            }
        )

    nr25_rows: list[dict[str, object]] = []
    removed = {"SL329": 176, "MXD494": 323, "DISORDER723": 424}
    kept = {"SL329": 2824, "MXD494": 2677, "DISORDER723": 2576}
    for row in nr25_rows_raw:
        dataset = row["dataset"]
        full = row_by(main_rows_raw, dataset=dataset, variant="p4_8_region_adapter_moe_tcn_gate002_warm_3seed_ensemble")
        nr25_rows.append(
            {
                "dataset": dataset,
                "removed_train_proteins": removed[dataset],
                "kept_train_proteins": kept[dataset],
                "full_auc": as_float(full["auc"]),
                "nr25_auc": as_float(row["candidate_auc"]),
                "nr25_minus_full_auc": as_float(row["candidate_auc"]) - as_float(full["auc"]),
                "nr25_aupr": as_float(row["candidate_aupr"]),
                "full_mcc": as_float(full["mcc"]),
                "nr25_mcc": as_float(row["candidate_mcc"]),
                "nr25_minus_full_mcc": as_float(row["candidate_mcc"]) - as_float(full["mcc"]),
                "p4_8_minus_p4_6_nr25_auc": as_float(row["auc_delta"]),
                "sota_auc": as_float(row["candidate_auc"]) - as_float(row["candidate_auc_gap_vs_sota"]),
                "nr25_gap_vs_sota": as_float(row["candidate_auc_gap_vs_sota"]),
            }
        )

    hard_focus: list[dict[str, object]] = []
    selected = {
        ("DISORDER723", "positive_region_location", "internal"),
        ("DISORDER723", "residue_zone", "middle"),
        ("MXD494", "positive_region_location", "internal"),
        ("SL329", "positive_region_location", "internal"),
        ("DISORDER723", "positive_region_length_type", "LDR"),
    }
    for row in hard_rows_raw:
        key = (row["dataset"], row["stratum_group"], row["stratum"])
        if key not in selected:
            continue
        hard_focus.append(
            {
                "dataset": row["dataset"],
                "stratum": row["display_stratum"],
                "reference_auc": as_float(row["reference_auc"]),
                "candidate_auc": as_float(row["candidate_auc"]),
                "auc_delta": as_float(row["auc_delta"]),
                "reference_aupr": as_float(row["reference_aupr"]),
                "candidate_aupr": as_float(row["candidate_aupr"]),
                "aupr_delta": as_float(row["aupr_delta"]),
                "reference_mcc": as_float(row["reference_mcc"]),
                "candidate_mcc": as_float(row["candidate_mcc"]),
                "mcc_delta": as_float(row["mcc_delta"]),
            }
        )

    calibration_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        raw = row_by(
            calibration_rows_raw,
            variant="p4_8_region_adapter_moe_tcn_gate002_warm_3seed_ensemble",
            dataset=dataset,
            method="raw",
        )
        platt = row_by(
            calibration_rows_raw,
            variant="p4_8_region_adapter_moe_tcn_gate002_warm_3seed_ensemble",
            dataset=dataset,
            method="platt",
        )
        top10 = uncertainty_top10(uncertainty_rows_raw, dataset)
        calibration_rows.append(
            {
                "dataset": dataset,
                "auc": as_float(platt["auc"]),
                "aupr": as_float(platt["aupr"]),
                "mcc": as_float(platt["mcc"]),
                "raw_ece": as_float(raw["ece"]),
                "platt_ece": as_float(platt["ece"]),
                "ece_delta": as_float(platt["ece"]) - as_float(raw["ece"]),
                "raw_brier": as_float(raw["brier"]),
                "platt_brier": as_float(platt["brier"]),
                "brier_delta": as_float(platt["brier"]) - as_float(raw["brier"]),
                "top10_uncertain_error_enrichment": as_float(top10["error_enrichment"]),
            }
        )

    benchmark_fields = [
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
        "p4_8_minus_p4_6_auc",
        "p4_8_minus_p4_6_mcc",
        "auc_delta_ci_95",
        "paired_permutation_p",
    ]
    nr25_fields = [
        "dataset",
        "removed_train_proteins",
        "kept_train_proteins",
        "full_auc",
        "nr25_auc",
        "nr25_minus_full_auc",
        "nr25_aupr",
        "full_mcc",
        "nr25_mcc",
        "nr25_minus_full_mcc",
        "p4_8_minus_p4_6_nr25_auc",
        "sota_auc",
        "nr25_gap_vs_sota",
    ]
    hard_fields = [
        "dataset",
        "stratum",
        "reference_auc",
        "candidate_auc",
        "auc_delta",
        "reference_aupr",
        "candidate_aupr",
        "aupr_delta",
        "reference_mcc",
        "candidate_mcc",
        "mcc_delta",
    ]
    calibration_fields = [
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
    ]

    write_tsv(TABLE_DIR / "P5_7_Table2_p4_8_full_benchmark.tsv", benchmark_rows, benchmark_fields)
    write_md(
        TABLE_DIR / "P5_7_Table2_p4_8_full_benchmark.md",
        "P5.7 Table 2. P4.8 full DM3000 benchmark performance",
        "P4.8 is the warm-start RegionAdapterMoETCN ensemble with Platt calibration. Paired statistics compare P4.8 against the P4.6 ESM2-t33 RegionAwareTCN ensemble.",
        benchmark_rows,
        benchmark_fields,
    )
    write_tsv(TABLE_DIR / "P5_7_Table3_p4_8_nr25_robustness.tsv", nr25_rows, nr25_fields)
    write_md(
        TABLE_DIR / "P5_7_Table3_p4_8_nr25_robustness.md",
        "P5.7 Table 3. P4.8 target-specific NR25 low-homology robustness",
        "NR25 results are target-specific seed1 warm-start adapter models and are reported as robustness evidence.",
        nr25_rows,
        nr25_fields,
    )
    write_tsv(TABLE_DIR / "P5_7_Table4_p4_8_hard_case_gain.tsv", hard_focus, hard_fields)
    write_md(
        TABLE_DIR / "P5_7_Table4_p4_8_hard_case_gain.md",
        "P5.7 Table 4. P4.8 hard-case gains over P4.6",
        "The main mechanistic performance signal is improved internal-IDR behavior, especially on DISORDER723.",
        hard_focus,
        hard_fields,
    )
    plot_hard_case_gain(
        ROOT / "figures/p5_7/P5_7_P4_8_HARD_CASE_GAIN.pdf",
        ROOT / "figures/p5_7/P5_7_P4_8_HARD_CASE_GAIN.png",
        hard_focus,
    )
    write_tsv(TABLE_DIR / "P5_7_Table5_p4_8_calibration_uncertainty.tsv", calibration_rows, calibration_fields)
    write_md(
        TABLE_DIR / "P5_7_Table5_p4_8_calibration_uncertainty.md",
        "P5.7 Table 5. P4.8 Platt calibration and uncertainty-error enrichment",
        "Calibration parameters and thresholds are fitted on DM1229 validation predictions only.",
        calibration_rows,
        calibration_fields,
    )

    report = f"""# P5.7 P4.8 Manuscript Update

Date: 2026-08-04

## Main Replacement

The manuscript main model should now be the P4.8 warm-start `RegionAdapterMoETCN` ensemble with validation-fitted Platt calibration. P4.6 remains the warm-start backbone and strongest architecture/representation baseline.

## P4.8 Full Benchmark Table

{markdown_table(benchmark_rows, benchmark_fields)}

## P4.8 NR25 Table

{markdown_table(nr25_rows, nr25_fields)}

## P4.8 Hard-Case Focus

{markdown_table(hard_focus, hard_fields)}

## P4.8 Calibration and Uncertainty

{markdown_table(calibration_rows, calibration_fields)}

## Recommended Claim Boundary

The safest claim is that warm-start region-specialized adapters preserve full-benchmark AUC, provide modest but consistent internal-IDR improvements, and maintain target-specific NR25 robustness relative to P4.6. The strongest direct evidence is DISORDER723 internal IDR improvement: AUC +0.007242, AUPR +0.019317 and MCC +0.027346. The paper should not overstate aggregate performance deltas, because MXD494 full-benchmark AUC changes by only -0.000166 and paired permutation p-values do not indicate strong global dominance.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
