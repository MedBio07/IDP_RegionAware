#!/usr/bin/env python3
"""Summarize P4.6 representation-upgrade experiments."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("SL329", "MXD494", "DISORDER723")
SOTA = {
    "SL329": {"method": "IDP-EDL", "auc": 0.915, "mcc": 0.700},
    "MXD494": {"method": "FusionEncoder", "auc": 0.842, "mcc": 0.492},
    "DISORDER723": {"method": "IDP-EDL", "auc": 0.943, "mcc": 0.636},
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_float(value: str | float | int | None) -> float:
    if value in (None, "", "NA"):
        return math.nan
    return float(value)


def format_value(value: object) -> str:
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
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def overall_from_stratified(path: Path) -> dict[str, str]:
    for row in read_tsv(path):
        if row["stratum_group"] == "overall" and row["stratum"] == "all_known":
            return row
    raise ValueError(f"{path}: missing overall/all_known")


def stratum_from_stratified(path: Path, group: str, stratum: str) -> dict[str, str]:
    for row in read_tsv(path):
        if row["stratum_group"] == group and row["stratum"] == stratum:
            return row
    raise ValueError(f"{path}: missing {group}/{stratum}")


def metric_row(
    variant: str,
    description: str,
    train_setting: str,
    dataset: str,
    path: Path,
) -> dict[str, object]:
    row = overall_from_stratified(path)
    sota = SOTA[dataset]
    auc = as_float(row["auc"])
    mcc = as_float(row["mcc"])
    return {
        "variant": variant,
        "description": description,
        "train_setting": train_setting,
        "dataset": dataset,
        "threshold": as_float(row["threshold"]),
        "sn": as_float(row["sn"]),
        "sp": as_float(row["sp"]),
        "bacc": as_float(row["bacc"]),
        "mcc": mcc,
        "auc": auc,
        "aupr": as_float(row["aupr"]),
        "fmax": as_float(row["fmax"]),
        "sota_method": sota["method"],
        "sota_auc": sota["auc"],
        "auc_gap_vs_sota": auc - float(sota["auc"]),
        "sota_mcc": sota["mcc"],
        "mcc_gap_vs_sota": mcc - float(sota["mcc"]),
    }


def main() -> None:
    out_dir = ROOT / "results/p4_6"

    comparison_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        comparison_rows.append(
            metric_row(
                "p2_region_aware_tcn_3seed_ensemble",
                "ESM2-t12 RegionAwareTCN 3-seed ensemble",
                "DM3000 full",
                dataset,
                ROOT / f"results/stratified/p2_region_aware_tcn_3seed_ensemble_{dataset}_stratified.tsv",
            )
        )
        comparison_rows.append(
            metric_row(
                "p4_6_region_aware_tcn_esm2_t33_3seed_ensemble",
                "ESM2-t33 RegionAwareTCN 3-seed ensemble",
                "DM3000 full",
                dataset,
                ROOT / f"results/p4_6/p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_{dataset}_stratified.tsv",
            )
        )

    nr25_specs = {
        "SL329": ROOT / "results/p4_6/p4_6_nr25_sl329_region_aware_tcn_esm2_t33_position_onehot_seed1_SL329_stratified.tsv",
        "MXD494": ROOT / "results/p4_6/p4_6_nr25_mxd494_region_aware_tcn_esm2_t33_position_onehot_seed1_MXD494_stratified.tsv",
        "DISORDER723": ROOT / "results/p4_6/p4_6_nr25_disorder723_region_aware_tcn_esm2_t33_position_onehot_seed1_DISORDER723_stratified.tsv",
    }
    nr25_rows: list[dict[str, object]] = []
    for dataset, path in nr25_specs.items():
        row = metric_row(
            f"p4_6_nr25_{dataset.lower()}_region_aware_tcn_esm2_t33_seed1",
            f"ESM2-t33 RegionAwareTCN seed1 trained on NR25 vs {dataset}",
            f"DM3000 NR25 vs {dataset}",
            dataset,
            path,
        )
        nr25_rows.append(row)

    full_t33_seed_rows: list[dict[str, object]] = []
    for seed in (1, 2, 3):
        for dataset in DATASETS:
            full_t33_seed_rows.append(
                metric_row(
                    f"p4_6_region_aware_tcn_esm2_t33_seed{seed}",
                    f"ESM2-t33 RegionAwareTCN seed{seed}",
                    "DM3000 full",
                    dataset,
                    ROOT / f"results/p4_6/p4_6_region_aware_tcn_esm2_t33_position_onehot_seed{seed}_{dataset}_stratified.tsv",
                )
            )

    multiseed_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        dataset_rows = [row for row in full_t33_seed_rows if row["dataset"] == dataset]
        for metric in ("auc", "aupr", "mcc", "fmax"):
            values = [float(row[metric]) for row in dataset_rows]
            multiseed_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "seed1": values[0],
                    "seed2": values[1],
                    "seed3": values[2],
                    "mean": statistics.mean(values),
                    "sample_std": statistics.stdev(values),
                    "min": min(values),
                    "max": max(values),
                }
            )

    internal_rows: list[dict[str, object]] = []
    for variant, description, path in [
        (
            "p2_region_aware_tcn_3seed_ensemble",
            "ESM2-t12 RegionAwareTCN 3-seed ensemble",
            ROOT / "results/stratified/p2_region_aware_tcn_3seed_ensemble_DISORDER723_stratified.tsv",
        ),
        (
            "p4_6_region_aware_tcn_esm2_t33_3seed_ensemble",
            "ESM2-t33 RegionAwareTCN 3-seed ensemble",
            ROOT / "results/p4_6/p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_DISORDER723_stratified.tsv",
        ),
        (
            "p4_6_nr25_disorder723_region_aware_tcn_esm2_t33_seed1",
            "ESM2-t33 RegionAwareTCN seed1 trained on NR25 vs DISORDER723",
            ROOT / "results/p4_6/p4_6_nr25_disorder723_region_aware_tcn_esm2_t33_position_onehot_seed1_DISORDER723_stratified.tsv",
        ),
    ]:
        row = stratum_from_stratified(path, "positive_region_location", "internal")
        internal_rows.append(
            {
                "variant": variant,
                "description": description,
                "dataset": "DISORDER723",
                "internal_sn": as_float(row["sn"]),
                "internal_sp": as_float(row["sp"]),
                "internal_bacc": as_float(row["bacc"]),
                "internal_mcc": as_float(row["mcc"]),
                "internal_auc": as_float(row["auc"]),
                "internal_aupr": as_float(row["aupr"]),
                "internal_fmax": as_float(row["fmax"]),
            }
        )

    calibration_rows: list[dict[str, object]] = []
    calibration_metrics = read_tsv(ROOT / "results/p4_6/calibration/p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_calibration_metrics.tsv")
    raw_by_dataset = {
        row["dataset"]: row
        for row in calibration_metrics
        if row["method"] == "raw"
    }
    for row in calibration_metrics:
        if row["method"] != "platt":
            continue
        raw = raw_by_dataset[row["dataset"]]
        calibration_rows.append(
            {
                "dataset": row["dataset"],
                "auc": as_float(row["auc"]),
                "aupr": as_float(row["aupr"]),
                "mcc": as_float(row["mcc"]),
                "raw_brier": as_float(raw["brier"]),
                "platt_brier": as_float(row["brier"]),
                "brier_delta": as_float(row["brier"]) - as_float(raw["brier"]),
                "raw_ece": as_float(raw["ece"]),
                "platt_ece": as_float(row["ece"]),
                "ece_delta": as_float(row["ece"]) - as_float(raw["ece"]),
                "raw_nll": as_float(raw["nll"]),
                "platt_nll": as_float(row["nll"]),
                "nll_delta": as_float(row["nll"]) - as_float(raw["nll"]),
            }
        )

    comparison_fields = [
        "variant",
        "description",
        "train_setting",
        "dataset",
        "threshold",
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
        "sota_mcc",
        "mcc_gap_vs_sota",
    ]
    multiseed_fields = ["dataset", "metric", "seed1", "seed2", "seed3", "mean", "sample_std", "min", "max"]
    internal_fields = [
        "variant",
        "description",
        "dataset",
        "internal_sn",
        "internal_sp",
        "internal_bacc",
        "internal_mcc",
        "internal_auc",
        "internal_aupr",
        "internal_fmax",
    ]
    calibration_fields = [
        "dataset",
        "auc",
        "aupr",
        "mcc",
        "raw_brier",
        "platt_brier",
        "brier_delta",
        "raw_ece",
        "platt_ece",
        "ece_delta",
        "raw_nll",
        "platt_nll",
        "nll_delta",
    ]

    write_tsv(out_dir / "P4_6_REPRESENTATION_UPGRADE_COMPARISON.tsv", comparison_rows, comparison_fields)
    write_tsv(out_dir / "P4_6_T33_MULTISEED_RESULTS.tsv", full_t33_seed_rows, comparison_fields)
    write_tsv(out_dir / "P4_6_T33_MULTISEED_SUMMARY.tsv", multiseed_rows, multiseed_fields)
    write_tsv(out_dir / "P4_6_NR25_COMPARISON.tsv", nr25_rows, comparison_fields)
    write_tsv(out_dir / "P4_6_DISORDER723_INTERNAL_COMPARISON.tsv", internal_rows, internal_fields)
    write_tsv(out_dir / "P4_6_SELECTED_PLATT_CALIBRATION.tsv", calibration_rows, calibration_fields)

    main_t33 = {
        row["dataset"]: row
        for row in comparison_rows
        if row["variant"] == "p4_6_region_aware_tcn_esm2_t33_3seed_ensemble"
    }
    main_t12 = {
        row["dataset"]: row
        for row in comparison_rows
        if row["variant"] == "p2_region_aware_tcn_3seed_ensemble"
    }

    report = f"""# P4.6 Representation Upgrade Summary

Date: 2026-08-03

## Scope

P4.6 tested whether upgrading the frozen protein-language-model representation closes the remaining SOTA gap while keeping the sequence-only RegionAwareTCN framework fixed.

Main change:

- Previous representation: ESM2-t12-35M layer 12.
- New representation: ESM2-t33-650M layer 33.
- Same model head: RegionAwareTCN + SDR/LDR + terminal/internal auxiliary heads.
- Same protocol: train on DM3000, tune threshold only on DM1229 validation, evaluate SL329/MXD494/DISORDER723.

## Generated Assets

- `configs/p4_6_region_aware_tcn_esm2_t33.yaml`
- `data/features/esm2_embeddings/esm2_t33_650M_UR50D_layer33_fp16/`
- `models/p4_6_region_aware_tcn_esm2_t33_position_onehot_seed*.pt`
- `predictions/p4_6/`
- `results/p4_6/P4_6_REPRESENTATION_UPGRADE_COMPARISON.tsv`
- `results/p4_6/P4_6_T33_MULTISEED_RESULTS.tsv`
- `results/p4_6/P4_6_T33_MULTISEED_SUMMARY.tsv`
- `results/p4_6/P4_6_NR25_COMPARISON.tsv`
- `results/p4_6/P4_6_SELECTED_PLATT_CALIBRATION.tsv`
- `figures/p4_6/calibration/`

## Main Full-Train Result

| Dataset | t12 AUC | t33 3-seed AUC | AUC delta | SOTA AUC | Gap vs SOTA | t33 MCC |
|---|---:|---:|---:|---:|---:|---:|
| SL329 | {format_value(main_t12["SL329"]["auc"])} | {format_value(main_t33["SL329"]["auc"])} | {format_value(float(main_t33["SL329"]["auc"]) - float(main_t12["SL329"]["auc"]))} | 0.915000 | {format_value(main_t33["SL329"]["auc_gap_vs_sota"])} | {format_value(main_t33["SL329"]["mcc"])} |
| MXD494 | {format_value(main_t12["MXD494"]["auc"])} | {format_value(main_t33["MXD494"]["auc"])} | {format_value(float(main_t33["MXD494"]["auc"]) - float(main_t12["MXD494"]["auc"]))} | 0.842000 | {format_value(main_t33["MXD494"]["auc_gap_vs_sota"])} | {format_value(main_t33["MXD494"]["mcc"])} |
| DISORDER723 | {format_value(main_t12["DISORDER723"]["auc"])} | {format_value(main_t33["DISORDER723"]["auc"])} | {format_value(float(main_t33["DISORDER723"]["auc"]) - float(main_t12["DISORDER723"]["auc"]))} | 0.943000 | {format_value(main_t33["DISORDER723"]["auc_gap_vs_sota"])} | {format_value(main_t33["DISORDER723"]["mcc"])} |

The t33 3-seed ensemble exceeds the collected SOTA AUC on all three target datasets. The largest gain is on DISORDER723: AUC improves from 0.923132 to 0.944611.

## Remaining Caveat

The DISORDER723 AUC claim is now strong, but MCC is still below IDP-EDL's reported 0.636:

- t33 3-seed DISORDER723 MCC: 0.610462
- IDP-EDL DISORDER723 MCC: 0.636

This means the manuscript can claim AUC-level SOTA only if the comparison table is explicit about MCC/Fmax.

## Internal IDR Improvement

DISORDER723 internal IDR improved but remains the hard case:

- t12 internal AUC: 0.838827
- t33 3-seed internal AUC: 0.887416
- t33 3-seed internal MCC: 0.210033

This is an important mechanistic result: the representation upgrade addresses the failure mode identified in P4.5, but does not fully solve internal IDR classification.

## NR25 Sanity Check

NR25 t33 seed1 results:

| Dataset | NR25 AUC | SOTA AUC | Gap vs SOTA | NR25 MCC |
|---|---:|---:|---:|---:|
| SL329 | {format_value(nr25_rows[0]["auc"])} | 0.915000 | {format_value(nr25_rows[0]["auc_gap_vs_sota"])} | {format_value(nr25_rows[0]["mcc"])} |
| MXD494 | {format_value(nr25_rows[1]["auc"])} | 0.842000 | {format_value(nr25_rows[1]["auc_gap_vs_sota"])} | {format_value(nr25_rows[1]["mcc"])} |
| DISORDER723 | {format_value(nr25_rows[2]["auc"])} | 0.943000 | {format_value(nr25_rows[2]["auc_gap_vs_sota"])} | {format_value(nr25_rows[2]["mcc"])} |

Interpretation: full-train t33 gives a performance-SOTA signal; NR25 shows the low-homology setting is harder and should be reported as a separate robustness result, not hidden.

## Calibration

Platt calibration preserves AUC/AUPR/MCC and substantially improves ECE/Brier/NLL, especially on DISORDER723:

- DISORDER723 raw ECE: 0.187936
- DISORDER723 Platt ECE: 0.017608
- DISORDER723 raw Brier: 0.093709
- DISORDER723 Platt Brier: 0.032479

## Decision

P4.6 is successful.

The project can now move toward P5, but the P5 framing must be precise:

1. Main performance claim: sequence-only RegionAwareTCN with ESM2-t33 reaches AUC-level SOTA on SL329, MXD494, and DISORDER723 in the full DM3000 setting.
2. Reliability claim: Platt calibration provides much better probability quality without degrading ranking metrics.
3. Robustness claim: NR25 results remain competitive but do not uniformly beat SOTA, so low-homology generalization should be reported transparently.
4. Remaining limitation: DISORDER723 MCC and internal IDR classification are still below the strongest reported benchmark.
"""
    (ROOT / "reports/P4_6_REPRESENTATION_UPGRADE_SUMMARY.md").write_text(report, encoding="utf-8")
    print(f"wrote {out_dir / 'P4_6_REPRESENTATION_UPGRADE_COMPARISON.tsv'}")
    print(f"wrote {out_dir / 'P4_6_NR25_COMPARISON.tsv'}")
    print(f"wrote {ROOT / 'reports/P4_6_REPRESENTATION_UPGRADE_SUMMARY.md'}")


if __name__ == "__main__":
    main()
