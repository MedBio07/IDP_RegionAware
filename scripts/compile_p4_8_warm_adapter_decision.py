#!/usr/bin/env python3
"""Compile P4.8 warm-start adapter replacement evidence."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compile_p5_evidence_bundle import collect_paired_protein_scores, paired_bootstrap_auc


DATASETS = ("SL329", "MXD494", "DISORDER723")
CALIBRATION_DATASETS = ("DM1229_Validation",) + DATASETS
SOTA = {
    "SL329": {"method": "IDP-EDL", "auc": 0.915, "mcc": 0.700},
    "MXD494": {"method": "FusionEncoder", "auc": 0.842, "mcc": 0.492},
    "DISORDER723": {"method": "IDP-EDL", "auc": 0.943, "mcc": 0.636},
}

P4_6_VARIANT = "p4_6_region_aware_tcn_esm2_t33_3seed_ensemble"
P4_8_VARIANT = "p4_8_region_adapter_moe_tcn_gate002_warm_3seed_ensemble"

P4_6_CALIBRATION_METRICS = (
    ROOT / "results/p4_6/calibration/p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_calibration_metrics.tsv"
)
P4_8_CALIBRATION_METRICS = (
    ROOT / "results/p4_8/calibration/p4_8_region_adapter_moe_tcn_gate002_warm_3seed_ensemble_calibration_metrics.tsv"
)

SELECTED_STRATA = (
    ("overall", "all_known", "Overall"),
    ("positive_region_length_type", "SDR", "Short disorder regions"),
    ("positive_region_length_type", "LDR", "Long disorder regions"),
    ("positive_region_location", "terminal", "Terminal IDRs"),
    ("positive_region_location", "internal", "Internal IDRs"),
    ("residue_zone", "middle", "Middle residues"),
    ("protein_disorder_content_bin", "0-5", "0-5% disorder proteins"),
    ("protein_length_bin", ">1000", ">1000 aa proteins"),
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_float(value: object) -> float:
    if value in (None, "", "NA"):
        return math.nan
    return float(value)


def fmt(value: object) -> str:
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            return "NA"
        return f"{float(value):.6f}"
    return str(value)


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def markdown_table(rows: list[dict[str, object]], fields: list[str], max_rows: int | None = None) -> str:
    selected = rows[:max_rows] if max_rows is not None else rows
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in selected:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def row_by(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        detail = ", ".join(f"{key}={value}" for key, value in criteria.items())
        raise ValueError(f"expected one row for {detail}; found {len(matches)}")
    return matches[0]


def optional_row_by(rows: list[dict[str, str]], **criteria: str) -> dict[str, str] | None:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if not matches:
        return None
    if len(matches) != 1:
        detail = ", ".join(f"{key}={value}" for key, value in criteria.items())
        raise ValueError(f"expected at most one row for {detail}; found {len(matches)}")
    return matches[0]


def calibration_row(
    variant: str,
    source: str,
    dataset: str,
    method: str,
    row: dict[str, str],
    reference: dict[str, str] | None,
) -> dict[str, object]:
    sota = SOTA.get(dataset, {"method": "NA", "auc": math.nan, "mcc": math.nan})
    auc = as_float(row["auc"])
    mcc = as_float(row["mcc"])
    out = {
        "variant": variant,
        "source": source,
        "dataset": dataset,
        "method": method,
        "threshold": as_float(row["threshold"]),
        "sn": as_float(row["sn"]),
        "sp": as_float(row["sp"]),
        "bacc": as_float(row["bacc"]),
        "mcc": mcc,
        "auc": auc,
        "aupr": as_float(row["aupr"]),
        "fmax": as_float(row["fmax"]),
        "brier": as_float(row["brier"]),
        "nll": as_float(row["nll"]),
        "ece": as_float(row["ece"]),
        "sota_method": sota["method"],
        "sota_auc": float(sota["auc"]),
        "auc_gap_vs_sota": auc - float(sota["auc"]) if math.isfinite(float(sota["auc"])) else math.nan,
        "sota_mcc": float(sota["mcc"]),
        "mcc_gap_vs_sota": mcc - float(sota["mcc"]) if math.isfinite(float(sota["mcc"])) else math.nan,
        "prediction_file": row.get("prediction_file", ""),
    }
    if reference is None:
        out.update(
            {
                "delta_auc_vs_p4_6_platt": math.nan,
                "delta_aupr_vs_p4_6_platt": math.nan,
                "delta_mcc_vs_p4_6_platt": math.nan,
                "delta_fmax_vs_p4_6_platt": math.nan,
                "delta_brier_vs_p4_6_platt": math.nan,
                "delta_nll_vs_p4_6_platt": math.nan,
                "delta_ece_vs_p4_6_platt": math.nan,
            }
        )
    else:
        out.update(
            {
                "delta_auc_vs_p4_6_platt": auc - as_float(reference["auc"]),
                "delta_aupr_vs_p4_6_platt": as_float(row["aupr"]) - as_float(reference["aupr"]),
                "delta_mcc_vs_p4_6_platt": mcc - as_float(reference["mcc"]),
                "delta_fmax_vs_p4_6_platt": as_float(row["fmax"]) - as_float(reference["fmax"]),
                "delta_brier_vs_p4_6_platt": as_float(row["brier"]) - as_float(reference["brier"]),
                "delta_nll_vs_p4_6_platt": as_float(row["nll"]) - as_float(reference["nll"]),
                "delta_ece_vs_p4_6_platt": as_float(row["ece"]) - as_float(reference["ece"]),
            }
        )
    return out


def compile_main_benchmark() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    p4_6 = read_tsv(P4_6_CALIBRATION_METRICS)
    p4_8 = read_tsv(P4_8_CALIBRATION_METRICS)

    main_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for dataset in CALIBRATION_DATASETS:
        p4_6_platt = row_by(p4_6, method="platt", dataset=dataset)
        p4_8_platt = row_by(p4_8, method="platt", dataset=dataset)
        for variant, source, rows in (
            (P4_6_VARIANT, "P4.6 baseline", p4_6),
            (P4_8_VARIANT, "P4.8 candidate", p4_8),
        ):
            for method in ("raw", "temperature", "platt", "isotonic"):
                method_row = row_by(rows, method=method, dataset=dataset)
                raw_row = row_by(rows, method="raw", dataset=dataset)
                reference = p4_6_platt if variant == P4_8_VARIANT and dataset in DATASETS else None
                compiled = calibration_row(variant, source, dataset, method, method_row, reference)
                compiled.update(
                    {
                        "delta_brier_vs_raw_same_variant": as_float(method_row["brier"]) - as_float(raw_row["brier"]),
                        "delta_nll_vs_raw_same_variant": as_float(method_row["nll"]) - as_float(raw_row["nll"]),
                        "delta_ece_vs_raw_same_variant": as_float(method_row["ece"]) - as_float(raw_row["ece"]),
                    }
                )
                calibration_rows.append(compiled)
        if dataset in DATASETS:
            main_rows.append(calibration_row(P4_6_VARIANT, "P4.6 baseline", dataset, "platt", p4_6_platt, None))
            main_rows.append(calibration_row(P4_8_VARIANT, "P4.8 candidate", dataset, "platt", p4_8_platt, p4_6_platt))
    return main_rows, calibration_rows


def stratified_path(variant: str, dataset: str) -> Path:
    if variant == P4_6_VARIANT:
        platt = ROOT / f"results/p4_8/stratified/p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_platt_{dataset}_stratified.tsv"
        if platt.exists():
            return platt
        return ROOT / f"results/p4_6/p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_{dataset}_stratified.tsv"
    if variant == P4_8_VARIANT:
        platt = ROOT / f"results/p4_8/stratified/p4_8_region_adapter_moe_tcn_gate002_warm_3seed_ensemble_platt_{dataset}_stratified.tsv"
        if platt.exists():
            return platt
        return ROOT / f"results/p4_7/P4_7_REGION_ADAPTER_MOE_TCN_GATE002_WARM_3SEED_ENSEMBLE_{dataset}_STRATIFIED.tsv"
    raise ValueError(f"unsupported variant: {variant}")


def compile_hard_case_replacement() -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for dataset in DATASETS:
        reference_rows = read_tsv(stratified_path(P4_6_VARIANT, dataset))
        candidate_rows = read_tsv(stratified_path(P4_8_VARIANT, dataset))
        for group, stratum, label in SELECTED_STRATA:
            reference = optional_row_by(reference_rows, stratum_group=group, stratum=stratum)
            candidate = optional_row_by(candidate_rows, stratum_group=group, stratum=stratum)
            if reference is None or candidate is None:
                continue
            rows_out.append(
                {
                    "dataset": dataset,
                    "stratum_group": group,
                    "stratum": stratum,
                    "display_stratum": label,
                    "reference_variant": P4_6_VARIANT,
                    "candidate_variant": P4_8_VARIANT,
                    "reference_auc": as_float(reference["auc"]),
                    "candidate_auc": as_float(candidate["auc"]),
                    "auc_delta": as_float(candidate["auc"]) - as_float(reference["auc"]),
                    "reference_aupr": as_float(reference["aupr"]),
                    "candidate_aupr": as_float(candidate["aupr"]),
                    "aupr_delta": as_float(candidate["aupr"]) - as_float(reference["aupr"]),
                    "reference_mcc": as_float(reference["mcc"]),
                    "candidate_mcc": as_float(candidate["mcc"]),
                    "mcc_delta": as_float(candidate["mcc"]) - as_float(reference["mcc"]),
                    "reference_fmax": as_float(reference["fmax"]),
                    "candidate_fmax": as_float(candidate["fmax"]),
                    "fmax_delta": as_float(candidate["fmax"]) - as_float(reference["fmax"]),
                    "residues": as_float(candidate["residues"]),
                    "positives": as_float(candidate["positives"]),
                }
            )
    return rows_out


def metric_row_from_stratified(
    variant: str,
    description: str,
    train_setting: str,
    dataset: str,
    path: Path,
) -> dict[str, object]:
    rows = read_tsv(path)
    row = row_by(rows, stratum_group="overall", stratum="all_known")
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
        "sota_auc": float(sota["auc"]),
        "auc_gap_vs_sota": auc - float(sota["auc"]),
        "sota_mcc": float(sota["mcc"]),
        "mcc_gap_vs_sota": mcc - float(sota["mcc"]),
    }


def compile_nr25_replacement() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    p4_6_rows = read_tsv(ROOT / "results/p4_6/P4_6_NR25_COMPARISON.tsv")
    p4_8_specs = {
        "SL329": ROOT / "results/p4_8/nr25/p4_8_nr25_sl329_region_adapter_moe_tcn_gate002_warm_seed1_SL329_stratified.tsv",
        "MXD494": ROOT / "results/p4_8/nr25/p4_8_nr25_mxd494_region_adapter_moe_tcn_gate002_warm_seed1_MXD494_stratified.tsv",
        "DISORDER723": ROOT / "results/p4_8/nr25/p4_8_nr25_disorder723_region_adapter_moe_tcn_gate002_warm_seed1_DISORDER723_stratified.tsv",
    }

    candidate_rows: list[dict[str, object]] = []
    replacement_rows: list[dict[str, object]] = []
    if not all(path.exists() for path in p4_8_specs.values()):
        return candidate_rows, replacement_rows

    for dataset, path in p4_8_specs.items():
        candidate = metric_row_from_stratified(
            f"p4_8_nr25_{dataset.lower()}_region_adapter_moe_tcn_gate002_warm_seed1",
            f"ESM2-t33 RegionAdapterMoETCN warm-start seed1 trained on NR25 vs {dataset}",
            f"DM3000 NR25 vs {dataset}",
            dataset,
            path,
        )
        candidate_rows.append(candidate)
        reference = row_by(p4_6_rows, dataset=dataset)
        replacement_rows.append(
            {
                "dataset": dataset,
                "reference_variant": reference["variant"],
                "candidate_variant": candidate["variant"],
                "reference_auc": as_float(reference["auc"]),
                "candidate_auc": as_float(candidate["auc"]),
                "auc_delta": as_float(candidate["auc"]) - as_float(reference["auc"]),
                "reference_aupr": as_float(reference["aupr"]),
                "candidate_aupr": as_float(candidate["aupr"]),
                "aupr_delta": as_float(candidate["aupr"]) - as_float(reference["aupr"]),
                "reference_mcc": as_float(reference["mcc"]),
                "candidate_mcc": as_float(candidate["mcc"]),
                "mcc_delta": as_float(candidate["mcc"]) - as_float(reference["mcc"]),
                "reference_fmax": as_float(reference["fmax"]),
                "candidate_fmax": as_float(candidate["fmax"]),
                "fmax_delta": as_float(candidate["fmax"]) - as_float(reference["fmax"]),
                "reference_auc_gap_vs_sota": as_float(reference["auc_gap_vs_sota"]),
                "candidate_auc_gap_vs_sota": as_float(candidate["auc_gap_vs_sota"]),
                "reference_mcc_gap_vs_sota": as_float(reference["mcc_gap_vs_sota"]),
                "candidate_mcc_gap_vs_sota": as_float(candidate["mcc_gap_vs_sota"]),
            }
        )
    return candidate_rows, replacement_rows


def compile_bootstrap(args: argparse.Namespace) -> list[dict[str, object]]:
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        paired = collect_paired_protein_scores(
            dataset,
            ROOT / f"predictions/p4_6/p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_{dataset}.tsv",
            ROOT / f"predictions/p4_7/p4_7_region_adapter_moe_tcn_esm2_t33_gate002_warm_3seed_ensemble_{dataset}.tsv",
        )
        row = paired_bootstrap_auc(dataset, paired, args.bootstrap, args.permutations, rng)
        row["reference_variant"] = P4_6_VARIANT
        row["candidate_variant"] = P4_8_VARIANT
        rows.append(row)
    return rows


def evidence_status(rows: list[dict[str, object]], hard_case_rows: list[dict[str, object]], nr25_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    candidate_rows = [row for row in rows if row["variant"] == P4_8_VARIANT and row["method"] == "platt"]
    main_deltas = [as_float(row["delta_auc_vs_p4_6_platt"]) for row in candidate_rows]
    mcc_deltas = [as_float(row["delta_mcc_vs_p4_6_platt"]) for row in candidate_rows]
    ece_deltas = [as_float(row["delta_ece_vs_p4_6_platt"]) for row in candidate_rows]
    brier_deltas = [as_float(row["delta_brier_vs_p4_6_platt"]) for row in candidate_rows]

    disorder_internal = optional_row_by(
        [{key: fmt(value) for key, value in row.items()} for row in hard_case_rows],
        dataset="DISORDER723",
        stratum_group="positive_region_location",
        stratum="internal",
    )
    internal_auc_delta = as_float(disorder_internal["auc_delta"]) if disorder_internal else math.nan
    internal_mcc_delta = as_float(disorder_internal["mcc_delta"]) if disorder_internal else math.nan

    decision_rows: list[dict[str, object]] = []
    min_auc_delta = min(main_deltas)
    decision_rows.append(
        {
            "criterion": "external_auc_preservation",
            "status": "pass" if min_auc_delta >= -0.001 else "fail",
            "evidence": f"Minimum Platt AUC delta vs P4.6 across external datasets is {min_auc_delta:.6f}.",
        }
    )
    decision_rows.append(
        {
            "criterion": "external_mcc_balance",
            "status": "pass" if sum(delta >= 0 for delta in mcc_deltas) >= 2 else "mixed",
            "evidence": f"MCC deltas vs P4.6 Platt are {', '.join(f'{delta:.6f}' for delta in mcc_deltas)}.",
        }
    )
    decision_rows.append(
        {
            "criterion": "platt_probability_quality",
            "status": "pass" if float(np.mean(ece_deltas)) <= 0 and float(np.mean(brier_deltas)) <= 0 else "mixed",
            "evidence": (
                f"Mean ECE delta is {float(np.mean(ece_deltas)):.6f}; "
                f"mean Brier delta is {float(np.mean(brier_deltas)):.6f}."
            ),
        }
    )
    decision_rows.append(
        {
            "criterion": "disorder723_internal_idr_gain",
            "status": "pass" if internal_auc_delta > 0 and internal_mcc_delta > 0 else "fail",
            "evidence": (
                f"DISORDER723 internal-IDR AUC delta is {internal_auc_delta:.6f}; "
                f"MCC delta is {internal_mcc_delta:.6f}."
            ),
        }
    )
    if nr25_rows:
        min_nr25_auc_delta = min(as_float(row["auc_delta"]) for row in nr25_rows)
        decision_rows.append(
            {
                "criterion": "nr25_low_homology_check",
                "status": "pass" if min_nr25_auc_delta >= -0.005 else "mixed",
                "evidence": f"Minimum NR25 AUC delta vs P4.6 NR25 is {min_nr25_auc_delta:.6f}.",
            }
        )
    else:
        decision_rows.append(
            {
                "criterion": "nr25_low_homology_check",
                "status": "pending",
                "evidence": "P4.8 NR25 target-specific stratified files are not all available yet.",
            }
        )
    return decision_rows


def final_decision(decision_rows: list[dict[str, object]]) -> str:
    statuses = {str(row["status"]) for row in decision_rows}
    if "fail" in statuses:
        return "Do not replace P4.6 as the manuscript main model yet."
    if "pending" in statuses:
        return "Preliminary go, pending NR25 completion."
    return (
        "Replace P4.6 with the P4.8 warm-start RegionAdapterMoETCN as the manuscript main method, "
        "while retaining P4.6 as the warm-start backbone and strongest ablation baseline."
    )


def write_report(
    path: Path,
    main_rows: list[dict[str, object]],
    bootstrap_rows: list[dict[str, object]],
    calibration_rows: list[dict[str, object]],
    hard_case_rows: list[dict[str, object]],
    nr25_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
) -> None:
    candidate_main = [row for row in main_rows if row["variant"] == P4_8_VARIANT]
    candidate_platt_calibration = [
        row for row in calibration_rows if row["variant"] == P4_8_VARIANT and row["method"] == "platt"
    ]
    hard_focus = [
        row
        for row in hard_case_rows
        if row["dataset"] == "DISORDER723"
        and (row["stratum_group"], row["stratum"]) in {("positive_region_location", "internal"), ("residue_zone", "middle")}
    ]

    text = f"""# P4.8 Warm-Start Adapter Replacement Decision

Date: 2026-08-04

## Scope

P4.8 evaluates whether the P4.7 warm-start `RegionAdapterMoETCN` ensemble should replace the P4.6 `RegionAwareTCN` ensemble as the manuscript main model. The candidate keeps the same sequence-only ESM2-t33/position/one-hot input setting, warm-starts from matched P4.6 seeds, freezes the shared backbone, and trains region adapters plus MoE gate/expert heads.

## Main Platt-Calibrated Benchmark

{markdown_table(candidate_main, [
    "dataset",
    "auc",
    "aupr",
    "mcc",
    "fmax",
    "ece",
    "brier",
    "delta_auc_vs_p4_6_platt",
    "delta_aupr_vs_p4_6_platt",
    "delta_mcc_vs_p4_6_platt",
    "delta_ece_vs_p4_6_platt",
])}

## Paired Protein-Level AUC Test

{markdown_table(bootstrap_rows, [
    "dataset",
    "reference_auc",
    "candidate_auc",
    "auc_delta",
    "auc_delta_ci_low",
    "auc_delta_ci_high",
    "paired_bootstrap_p_delta_le_0",
    "paired_permutation_p_one_sided",
    "n_bootstrap",
    "n_permutation",
])}

## Platt Probability Quality

{markdown_table(candidate_platt_calibration, [
    "dataset",
    "brier",
    "nll",
    "ece",
    "delta_brier_vs_p4_6_platt",
    "delta_nll_vs_p4_6_platt",
    "delta_ece_vs_p4_6_platt",
])}

## DISORDER723 Hard Cases

{markdown_table(hard_focus, [
    "display_stratum",
    "reference_auc",
    "candidate_auc",
    "auc_delta",
    "reference_aupr",
    "candidate_aupr",
    "aupr_delta",
    "reference_mcc",
    "candidate_mcc",
    "mcc_delta",
])}

## NR25 Check

{markdown_table(nr25_rows, [
    "dataset",
    "reference_auc",
    "candidate_auc",
    "auc_delta",
    "reference_mcc",
    "candidate_mcc",
    "mcc_delta",
]) if nr25_rows else "P4.8 NR25 results are pending."}

## Decision Criteria

{markdown_table(decision_rows, ["criterion", "status", "evidence"])}

## Decision

{final_decision(decision_rows)}

## Manuscript Claim Boundary

The replacement, if used, should be framed conservatively: aggregate AUC gains are small and protein-level confidence intervals are expected to be wider than the point deltas, whereas the strongest direct mechanistic signal is improved DISORDER723 internal-IDR performance. The paper should describe P4.8 as a sequence-only region-adapter specialization over the P4.6 RegionAwareTCN backbone, not as a wholesale representation change.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = ROOT / "results/p4_8"

    main_rows, calibration_rows = compile_main_benchmark()
    hard_case_rows = compile_hard_case_replacement()
    nr25_candidate_rows, nr25_replacement_rows = compile_nr25_replacement()
    bootstrap_rows = compile_bootstrap(args)
    decision_rows = evidence_status(main_rows, hard_case_rows, nr25_replacement_rows)

    main_fields = [
        "variant",
        "source",
        "dataset",
        "method",
        "threshold",
        "sn",
        "sp",
        "bacc",
        "mcc",
        "auc",
        "aupr",
        "fmax",
        "brier",
        "nll",
        "ece",
        "sota_method",
        "sota_auc",
        "auc_gap_vs_sota",
        "sota_mcc",
        "mcc_gap_vs_sota",
        "delta_auc_vs_p4_6_platt",
        "delta_aupr_vs_p4_6_platt",
        "delta_mcc_vs_p4_6_platt",
        "delta_fmax_vs_p4_6_platt",
        "delta_brier_vs_p4_6_platt",
        "delta_nll_vs_p4_6_platt",
        "delta_ece_vs_p4_6_platt",
        "prediction_file",
    ]
    calibration_fields = main_fields + [
        "delta_brier_vs_raw_same_variant",
        "delta_nll_vs_raw_same_variant",
        "delta_ece_vs_raw_same_variant",
    ]
    hard_case_fields = [
        "dataset",
        "stratum_group",
        "stratum",
        "display_stratum",
        "reference_variant",
        "candidate_variant",
        "reference_auc",
        "candidate_auc",
        "auc_delta",
        "reference_aupr",
        "candidate_aupr",
        "aupr_delta",
        "reference_mcc",
        "candidate_mcc",
        "mcc_delta",
        "reference_fmax",
        "candidate_fmax",
        "fmax_delta",
        "residues",
        "positives",
    ]
    nr25_comparison_fields = [
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
    nr25_replacement_fields = [
        "dataset",
        "reference_variant",
        "candidate_variant",
        "reference_auc",
        "candidate_auc",
        "auc_delta",
        "reference_aupr",
        "candidate_aupr",
        "aupr_delta",
        "reference_mcc",
        "candidate_mcc",
        "mcc_delta",
        "reference_fmax",
        "candidate_fmax",
        "fmax_delta",
        "reference_auc_gap_vs_sota",
        "candidate_auc_gap_vs_sota",
        "reference_mcc_gap_vs_sota",
        "candidate_mcc_gap_vs_sota",
    ]
    bootstrap_fields = [
        "dataset",
        "proteins",
        "known_residues",
        "positives",
        "negatives",
        "reference_variant",
        "candidate_variant",
        "reference_auc",
        "reference_auc_ci_low",
        "reference_auc_ci_high",
        "candidate_auc",
        "candidate_auc_ci_low",
        "candidate_auc_ci_high",
        "auc_delta",
        "auc_delta_ci_low",
        "auc_delta_ci_high",
        "paired_bootstrap_p_delta_le_0",
        "paired_permutation_p_one_sided",
        "sota_method",
        "sota_auc",
        "candidate_gap_vs_sota",
        "candidate_auc_ci_low_gt_sota",
        "bootstrap_p_candidate_le_sota",
        "n_bootstrap",
        "n_permutation",
    ]

    write_tsv(result_dir / "P4_8_MAIN_BENCHMARK_REPLACEMENT.tsv", main_rows, main_fields)
    write_tsv(result_dir / "P4_8_CALIBRATION_REPLACEMENT.tsv", calibration_rows, calibration_fields)
    write_tsv(result_dir / "P4_8_HARD_CASE_REPLACEMENT.tsv", hard_case_rows, hard_case_fields)
    if nr25_candidate_rows:
        write_tsv(result_dir / "P4_8_NR25_REGION_ADAPTER_MOE_COMPARISON.tsv", nr25_candidate_rows, nr25_comparison_fields)
        write_tsv(result_dir / "P4_8_NR25_REPLACEMENT.tsv", nr25_replacement_rows, nr25_replacement_fields)
    write_tsv(result_dir / "P4_8_P4_7_WARM_VS_P4_6_BOOTSTRAP.tsv", bootstrap_rows, bootstrap_fields)
    write_tsv(result_dir / "P4_8_MODEL_DECISION_CRITERIA.tsv", decision_rows, ["criterion", "status", "evidence"])
    write_report(
        ROOT / "reports/P4_8_WARM_ADAPTER_REPLACEMENT_DECISION.md",
        main_rows,
        bootstrap_rows,
        calibration_rows,
        hard_case_rows,
        nr25_replacement_rows,
        decision_rows,
    )


if __name__ == "__main__":
    main()
