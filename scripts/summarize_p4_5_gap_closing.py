#!/usr/bin/env python3
"""Summarize P4.5 SOTA gap-closing experiments."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("SL329", "MXD494", "DISORDER723")
SOTA_AUC = {"SL329": 0.915, "MXD494": 0.842, "DISORDER723": 0.943}


VARIANTS = [
    {
        "variant": "main_region_aware_3seed",
        "description": "P2 main candidate: RegionAwareTCN + aux, 3-seed ensemble",
        "metrics": {dataset: ROOT / f"results/fusion/p2_region_aware_tcn_3seed_ensemble_{dataset}_metrics.tsv" for dataset in DATASETS},
        "stratified": {dataset: ROOT / f"results/stratified/p2_region_aware_tcn_3seed_ensemble_{dataset}_stratified.tsv" for dataset in DATASETS},
    },
    {
        "variant": "validation_selected_w3_smoothing",
        "description": "Validation-selected local score smoothing for main candidate",
        "postprocess": ROOT / "results/postprocess/P4_5_POSTPROCESS_COMPARISON.tsv",
        "postprocess_model_id": "p2_region_aware_tcn_3seed_ensemble",
        "stratified": {
            "DISORDER723": ROOT / "results/p4_5/p4_5_region_aware_score_smooth_w3_raw0.50_DISORDER723_stratified.tsv"
        },
    },
    {
        "variant": "region_aware_no_position_seed1",
        "description": "RegionAwareTCN seed1 without relative position feature",
        "stratified": {dataset: ROOT / f"results/p4_5/p4_5_region_aware_tcn_esm2_onehot_seed1_{dataset}_stratified.tsv" for dataset in DATASETS},
    },
    {
        "variant": "main_plus_no_position_equal",
        "description": "Equal score average of main RegionAware 3-seed and no-position seed1",
        "stratified": {dataset: ROOT / f"results/p4_5/p4_5_region_aware_plus_nopos_equal_{dataset}_stratified.tsv" for dataset in DATASETS},
    },
    {
        "variant": "region_aware_focal_g2_seed1",
        "description": "RegionAwareTCN seed1 with focal BCE, gamma=2",
        "stratified": {dataset: ROOT / f"results/p4_5/p4_5_region_aware_tcn_focal_g2_seed1_{dataset}_stratified.tsv" for dataset in DATASETS},
    },
    {
        "variant": "main_plus_focal_equal",
        "description": "Equal score average of main RegionAware 3-seed and focal seed1",
        "stratified": {dataset: ROOT / f"results/p4_5/p4_5_region_aware_plus_focal_equal_{dataset}_stratified.tsv" for dataset in DATASETS},
    },
    {
        "variant": "region_aware_asymmetric_gn2_seed1",
        "description": "RegionAwareTCN seed1 with asymmetric BCE, gamma_neg=2",
        "stratified": {dataset: ROOT / f"results/p4_5/p4_5_region_aware_tcn_asym_gn2_seed1_{dataset}_stratified.tsv" for dataset in DATASETS},
    },
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def float_value(row: dict[str, str], key: str) -> float:
    value = row.get(key, "NA")
    if value in ("", "NA", None):
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
    raise ValueError(f"{path}: missing overall/all_known row")


def internal_from_stratified(path: Path) -> dict[str, str]:
    for row in read_tsv(path):
        if row["stratum_group"] == "positive_region_location" and row["stratum"] == "internal":
            return row
    raise ValueError(f"{path}: missing internal row")


def metric_row_from_variant(variant: dict[str, object], dataset: str) -> dict[str, object] | None:
    if "postprocess" in variant:
        for row in read_tsv(variant["postprocess"]):
            if row["model_id"] == variant["postprocess_model_id"] and row["dataset"] == dataset:
                return {
                    "auc": float_value(row, "selected_auc"),
                    "aupr": float_value(row, "selected_aupr"),
                    "mcc": float_value(row, "selected_mcc"),
                    "fmax": float_value(row, "selected_fmax"),
                    "sn": float_value(row, "selected_sn"),
                    "sp": float_value(row, "selected_sp"),
                    "bacc": float_value(row, "selected_bacc"),
                    "threshold": float_value(row, "selected_threshold"),
                }
        return None

    if "stratified" in variant and dataset in variant["stratified"]:
        row = overall_from_stratified(variant["stratified"][dataset])
        return {
            "auc": float_value(row, "auc"),
            "aupr": float_value(row, "aupr"),
            "mcc": float_value(row, "mcc"),
            "fmax": float_value(row, "fmax"),
            "sn": float_value(row, "sn"),
            "sp": float_value(row, "sp"),
            "bacc": float_value(row, "bacc"),
            "threshold": float_value(row, "threshold"),
        }

    if "metrics" in variant and dataset in variant["metrics"]:
        row = read_tsv(variant["metrics"][dataset])[0]
        return {
            "auc": float_value(row, "auc"),
            "aupr": float_value(row, "aupr"),
            "mcc": float_value(row, "mcc"),
            "fmax": float_value(row, "fmax"),
            "sn": float_value(row, "sn"),
            "sp": float_value(row, "sp"),
            "bacc": float_value(row, "bacc"),
            "threshold": float_value(row, "threshold"),
        }
    return None


def main() -> None:
    comparison_rows: list[dict[str, object]] = []
    internal_rows: list[dict[str, object]] = []

    main_auc_by_dataset: dict[str, float] = {}
    for variant in VARIANTS:
        if variant["variant"] != "main_region_aware_3seed":
            continue
        for dataset in DATASETS:
            metrics = metric_row_from_variant(variant, dataset)
            if metrics:
                main_auc_by_dataset[dataset] = float(metrics["auc"])

    for variant in VARIANTS:
        for dataset in DATASETS:
            metrics = metric_row_from_variant(variant, dataset)
            if metrics is None:
                continue
            auc = float(metrics["auc"])
            comparison_rows.append(
                {
                    "variant": variant["variant"],
                    "description": variant["description"],
                    "dataset": dataset,
                    **metrics,
                    "sota_auc": SOTA_AUC[dataset],
                    "auc_gap_vs_sota": auc - SOTA_AUC[dataset],
                    "auc_delta_vs_main": auc - main_auc_by_dataset[dataset],
                }
            )

        stratified = variant.get("stratified", {})
        if "DISORDER723" in stratified:
            row = internal_from_stratified(stratified["DISORDER723"])
            internal_rows.append(
                {
                    "variant": variant["variant"],
                    "description": variant["description"],
                    "dataset": "DISORDER723",
                    "internal_auc": float_value(row, "auc"),
                    "internal_aupr": float_value(row, "aupr"),
                    "internal_mcc": float_value(row, "mcc"),
                    "internal_fmax": float_value(row, "fmax"),
                    "internal_sn": float_value(row, "sn"),
                    "internal_sp": float_value(row, "sp"),
                }
            )

    comparison_fields = [
        "variant",
        "description",
        "dataset",
        "threshold",
        "sn",
        "sp",
        "bacc",
        "mcc",
        "auc",
        "aupr",
        "fmax",
        "sota_auc",
        "auc_gap_vs_sota",
        "auc_delta_vs_main",
    ]
    internal_fields = [
        "variant",
        "description",
        "dataset",
        "internal_sn",
        "internal_sp",
        "internal_mcc",
        "internal_auc",
        "internal_aupr",
        "internal_fmax",
    ]
    out_dir = ROOT / "results/p4_5"
    write_tsv(out_dir / "P4_5_GAP_CLOSING_COMPARISON.tsv", comparison_rows, comparison_fields)
    write_tsv(out_dir / "P4_5_DISORDER723_INTERNAL_COMPARISON.tsv", internal_rows, internal_fields)

    best_disorder = max((row for row in comparison_rows if row["dataset"] == "DISORDER723"), key=lambda row: float(row["auc"]))
    best_internal = max(internal_rows, key=lambda row: float(row["internal_auc"]))
    report = f"""# P4.5 SOTA Gap-Closing Summary

Date: 2026-07-31

## Scope

P4.5 tested low-cost strategies intended to close the remaining SOTA gap, especially on DISORDER723:

- DISORDER723 residue/segment/protein error analysis.
- Validation-selected local score smoothing.
- Removing relative position from RegionAwareTCN.
- Equal ensembles combining the main model with no-position or focal variants.
- Focal BCE and asymmetric BCE re-training.

## Generated Assets

- `scripts/analyze_disorder_errors.py`
- `scripts/evaluate_score_postprocessing.py`
- `scripts/summarize_p4_5_gap_closing.py`
- `results/error_analysis/`
- `results/postprocess/`
- `results/p4_5/P4_5_GAP_CLOSING_COMPARISON.tsv`
- `results/p4_5/P4_5_DISORDER723_INTERNAL_COMPARISON.tsv`

## Main Result

Best DISORDER723 AUC in this P4.5 round:

- Variant: `{best_disorder["variant"]}`
- AUC: {format_value(best_disorder["auc"])}
- AUC gap vs IDP-EDL 0.943: {format_value(best_disorder["auc_gap_vs_sota"])}

This does not materially improve over the current main model AUC 0.923132 and remains far from IDP-EDL's reported 0.943.

## DISORDER723 Failure Mode

The dominant failure mode is internal/middle disorder:

- Main RegionAware internal AUC: 0.838827
- Main RegionAware internal AUPR: 0.105671
- Main RegionAware internal MCC: 0.158647

Best internal AUC in this P4.5 round:

- Variant: `{best_internal["variant"]}`
- Internal AUC: {format_value(best_internal["internal_auc"])}
- Internal AUPR: {format_value(best_internal["internal_aupr"])}
- Internal MCC: {format_value(best_internal["internal_mcc"])}

Some variants improve internal-region metrics, but they reduce overall AUC and do not close the SOTA gap.

## Interpretation

1. Local smoothing is not the answer. Validation-selected smoothing changes DISORDER723 AUC by only about +0.00005.
2. Removing position confirms a terminal-position bias component: internal metrics improve slightly, but overall performance drops.
3. Focal/asymmetric losses improve specificity or selected internal metrics, but they lower overall AUC.
4. Simple equal ensembles do not preserve the main model's AUC advantage.

## Decision

Do not enter final P5 as a performance-SOTA paper yet.

The current evidence supports a reliability/generalization/calibration manuscript, but not a full SOTA-performance claim. To pursue a higher-impact performance claim, the next technical step should be stronger representation, not more local post-processing:

1. Extract and test a larger PLM, preferably ESM2-t33-650M or ProtT5 if available.
2. Keep the current RegionAware + Platt pipeline fixed as the comparison scaffold.
3. Specifically monitor DISORDER723 internal IDR, 501-1000 aa proteins, LDR, and middle-zone residues.
4. Only after a representation upgrade improves DISORDER723 AUC should multi-seed and NR25 repeats be run.

## Go/No-Go

Current P4.5 first-round result: No-Go for P5 performance-SOTA framing.

Allowed next directions:

- P4.6 representation upgrade experiment.
- Or P5 reliability-focused manuscript framing without claiming broad SOTA.
"""
    (ROOT / "reports/P4_5_SOTA_GAP_CLOSING_SUMMARY.md").write_text(report, encoding="utf-8")
    print(f"wrote {out_dir / 'P4_5_GAP_CLOSING_COMPARISON.tsv'}")
    print(f"wrote {out_dir / 'P4_5_DISORDER723_INTERNAL_COMPARISON.tsv'}")
    print(f"wrote {ROOT / 'reports/P4_5_SOTA_GAP_CLOSING_SUMMARY.md'}")


if __name__ == "__main__":
    main()
