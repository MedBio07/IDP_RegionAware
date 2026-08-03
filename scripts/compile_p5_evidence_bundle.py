#!/usr/bin/env python3
"""Compile P5 manuscript-level evidence from completed P2/P4.6 outputs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_disorder_predictions import build_id_lookup, parse_labeled_fasta, read_prediction_tsv


DATASETS = {
    "SL329": ROOT / "data/SL329_test.fasta",
    "MXD494": ROOT / "data/MXD494_test.fasta",
    "DISORDER723": ROOT / "data/DISORDER723_test.fasta",
}

SOTA = {
    "SL329": {"method": "IDP-EDL", "auc": 0.915, "mcc": 0.700},
    "MXD494": {"method": "FusionEncoder", "auc": 0.842, "mcc": 0.492},
    "DISORDER723": {"method": "IDP-EDL", "auc": 0.943, "mcc": 0.636},
}


@dataclass(frozen=True)
class ProteinScores:
    protein_id: str
    labels: np.ndarray
    reference_scores: np.ndarray
    candidate_scores: np.ndarray


@dataclass(frozen=True)
class AucContributionMatrices:
    pos_counts: np.ndarray
    neg_counts: np.ndarray
    reference_reference: np.ndarray
    candidate_candidate: np.ndarray
    candidate_reference: np.ndarray
    reference_candidate: np.ndarray


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


def metric_value(rows: list[dict[str, str]], key: str, value: str, dataset: str) -> dict[str, str]:
    matches = [row for row in rows if row.get(key) == value and row.get("dataset") == dataset]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {key}={value}, dataset={dataset}; found {len(matches)}")
    return matches[0]


def calibration_value(rows: list[dict[str, str]], method: str, dataset: str) -> dict[str, str]:
    matches = [row for row in rows if row.get("method") == method and row.get("dataset") == dataset]
    if len(matches) != 1:
        raise ValueError(f"expected one calibration row for method={method}, dataset={dataset}; found {len(matches)}")
    return matches[0]


def collect_paired_protein_scores(dataset: str, reference_prediction: Path, candidate_prediction: Path) -> list[ProteinScores]:
    records = parse_labeled_fasta(DATASETS[dataset])
    reference = read_prediction_tsv(reference_prediction, "\t")
    candidate = read_prediction_tsv(candidate_prediction, "\t")
    lookup = build_id_lookup(records)
    rows: list[ProteinScores] = []

    for record in records:
        protein_id = str(record["id"])
        ref_scores = reference.get(protein_id) or reference.get(protein_id.split()[0])
        cand_scores = candidate.get(protein_id) or candidate.get(protein_id.split()[0])
        if ref_scores is None:
            raise ValueError(f"{reference_prediction}: missing predictions for {protein_id}")
        if cand_scores is None:
            raise ValueError(f"{candidate_prediction}: missing predictions for {protein_id}")

        sequence = str(record["sequence"])
        labels = record["labels"]
        assert isinstance(labels, list)
        if len(ref_scores) != len(sequence):
            raise ValueError(f"{reference_prediction}: length mismatch for {protein_id}")
        if len(cand_scores) != len(sequence):
            raise ValueError(f"{candidate_prediction}: length mismatch for {protein_id}")

        label_array = np.asarray(labels, dtype=np.int8)
        known = label_array != -1
        if not np.any(known):
            continue
        rows.append(
            ProteinScores(
                protein_id=protein_id,
                labels=label_array[known].astype(np.int8, copy=False),
                reference_scores=np.asarray(ref_scores, dtype=np.float64)[known],
                candidate_scores=np.asarray(cand_scores, dtype=np.float64)[known],
            )
        )

    for prediction_path, predictions in ((reference_prediction, reference), (candidate_prediction, candidate)):
        unknown = sorted(set(predictions) - set(lookup))
        if unknown:
            raise ValueError(f"{prediction_path}: prediction IDs not present in labels: {', '.join(unknown[:5])}")
    return rows


def auc_from_parts(labels_parts: list[np.ndarray], score_parts: list[np.ndarray]) -> float:
    labels = np.concatenate(labels_parts)
    scores = np.concatenate(score_parts)
    if np.sum(labels == 1) == 0 or np.sum(labels == 0) == 0:
        return math.nan
    return float(roc_auc_score(labels, scores))


def contribution_matrix(pos_parts: list[np.ndarray], neg_parts: list[np.ndarray]) -> np.ndarray:
    """Return Mann-Whitney positive-vs-negative contribution by protein pair."""
    n = len(pos_parts)
    matrix = np.zeros((n, n), dtype=np.float64)
    sorted_negs = [np.sort(values) for values in neg_parts]
    for i, positives in enumerate(pos_parts):
        if len(positives) == 0:
            continue
        for j, negatives in enumerate(sorted_negs):
            if len(negatives) == 0:
                continue
            left = np.searchsorted(negatives, positives, side="left")
            right = np.searchsorted(negatives, positives, side="right")
            matrix[i, j] = float(np.sum(left + 0.5 * (right - left)))
    return matrix


def build_auc_matrices(rows: list[ProteinScores]) -> AucContributionMatrices:
    reference_pos: list[np.ndarray] = []
    reference_neg: list[np.ndarray] = []
    candidate_pos: list[np.ndarray] = []
    candidate_neg: list[np.ndarray] = []
    pos_counts: list[int] = []
    neg_counts: list[int] = []

    for row in rows:
        positives = row.labels == 1
        negatives = row.labels == 0
        pos_counts.append(int(np.sum(positives)))
        neg_counts.append(int(np.sum(negatives)))
        reference_pos.append(row.reference_scores[positives])
        reference_neg.append(row.reference_scores[negatives])
        candidate_pos.append(row.candidate_scores[positives])
        candidate_neg.append(row.candidate_scores[negatives])

    return AucContributionMatrices(
        pos_counts=np.asarray(pos_counts, dtype=np.float64),
        neg_counts=np.asarray(neg_counts, dtype=np.float64),
        reference_reference=contribution_matrix(reference_pos, reference_neg),
        candidate_candidate=contribution_matrix(candidate_pos, candidate_neg),
        candidate_reference=contribution_matrix(candidate_pos, reference_neg),
        reference_candidate=contribution_matrix(reference_pos, candidate_neg),
    )


def weighted_auc(matrix: np.ndarray, pos_counts: np.ndarray, neg_counts: np.ndarray, weights: np.ndarray) -> float:
    positives = float(np.dot(weights, pos_counts))
    negatives = float(np.dot(weights, neg_counts))
    if positives <= 0.0 or negatives <= 0.0:
        return math.nan
    numerator = float(weights @ matrix @ weights)
    return numerator / (positives * negatives)


def percentile_ci(values: list[float], low: float = 2.5, high: float = 97.5) -> tuple[float, float]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if len(array) == 0:
        return math.nan, math.nan
    return float(np.percentile(array, low)), float(np.percentile(array, high))


def paired_bootstrap_auc(
    dataset: str,
    rows: list[ProteinScores],
    n_bootstrap: int,
    n_permutation: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    matrices = build_auc_matrices(rows)
    n = len(rows)
    unit_weights = np.ones(n, dtype=np.float64)
    observed_reference_auc = weighted_auc(
        matrices.reference_reference, matrices.pos_counts, matrices.neg_counts, unit_weights
    )
    observed_candidate_auc = weighted_auc(
        matrices.candidate_candidate, matrices.pos_counts, matrices.neg_counts, unit_weights
    )
    observed_delta = observed_candidate_auc - observed_reference_auc

    reference_aucs: list[float] = []
    candidate_aucs: list[float] = []
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        weights = np.bincount(indices, minlength=n).astype(np.float64, copy=False)
        reference_auc = weighted_auc(matrices.reference_reference, matrices.pos_counts, matrices.neg_counts, weights)
        candidate_auc = weighted_auc(matrices.candidate_candidate, matrices.pos_counts, matrices.neg_counts, weights)
        if math.isfinite(reference_auc) and math.isfinite(candidate_auc):
            reference_aucs.append(reference_auc)
            candidate_aucs.append(candidate_auc)
            deltas.append(candidate_auc - reference_auc)

    permutation_deltas: list[float] = []
    total_positives = float(np.sum(matrices.pos_counts))
    total_negatives = float(np.sum(matrices.neg_counts))
    permutation_denominator = total_positives * total_negatives
    for _ in range(n_permutation):
        keep_candidate = (rng.random(n) < 0.5).astype(np.float64)
        use_reference = 1.0 - keep_candidate
        candidate_numerator = (
            keep_candidate @ matrices.candidate_candidate @ keep_candidate
            + keep_candidate @ matrices.candidate_reference @ use_reference
            + use_reference @ matrices.reference_candidate @ keep_candidate
            + use_reference @ matrices.reference_reference @ use_reference
        )
        reference_numerator = (
            use_reference @ matrices.candidate_candidate @ use_reference
            + use_reference @ matrices.candidate_reference @ keep_candidate
            + keep_candidate @ matrices.reference_candidate @ use_reference
            + keep_candidate @ matrices.reference_reference @ keep_candidate
        )
        permutation_deltas.append((candidate_numerator - reference_numerator) / permutation_denominator)

    reference_low, reference_high = percentile_ci(reference_aucs)
    candidate_low, candidate_high = percentile_ci(candidate_aucs)
    delta_low, delta_high = percentile_ci(deltas)
    p_bootstrap_delta_le_0 = (sum(delta <= 0.0 for delta in deltas) + 1.0) / (len(deltas) + 1.0)
    p_permutation_one_sided = (
        (sum(delta >= observed_delta for delta in permutation_deltas) + 1.0)
        / (len(permutation_deltas) + 1.0)
        if permutation_deltas
        else math.nan
    )
    sota_auc = SOTA[dataset]["auc"]
    p_bootstrap_candidate_le_sota = (
        sum(auc <= sota_auc for auc in candidate_aucs) + 1.0
    ) / (len(candidate_aucs) + 1.0)

    return {
        "dataset": dataset,
        "proteins": len(rows),
        "known_residues": int(sum(len(row.labels) for row in rows)),
        "positives": int(sum(np.sum(row.labels == 1) for row in rows)),
        "negatives": int(sum(np.sum(row.labels == 0) for row in rows)),
        "reference_variant": "p2_region_aware_tcn_3seed_ensemble",
        "candidate_variant": "p4_6_region_aware_tcn_esm2_t33_3seed_ensemble",
        "reference_auc": observed_reference_auc,
        "reference_auc_ci_low": reference_low,
        "reference_auc_ci_high": reference_high,
        "candidate_auc": observed_candidate_auc,
        "candidate_auc_ci_low": candidate_low,
        "candidate_auc_ci_high": candidate_high,
        "auc_delta": observed_delta,
        "auc_delta_ci_low": delta_low,
        "auc_delta_ci_high": delta_high,
        "paired_bootstrap_p_delta_le_0": p_bootstrap_delta_le_0,
        "paired_permutation_p_one_sided": p_permutation_one_sided,
        "sota_method": SOTA[dataset]["method"],
        "sota_auc": sota_auc,
        "candidate_gap_vs_sota": observed_candidate_auc - sota_auc,
        "candidate_auc_ci_low_gt_sota": candidate_low > sota_auc,
        "bootstrap_p_candidate_le_sota": p_bootstrap_candidate_le_sota,
        "n_bootstrap": len(candidate_aucs),
        "n_permutation": len(permutation_deltas),
    }


def compile_bootstrap(args: argparse.Namespace) -> list[dict[str, object]]:
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        paired = collect_paired_protein_scores(
            dataset,
            ROOT / f"predictions/fusion/p2_region_aware_tcn_3seed_ensemble_{dataset}.tsv",
            ROOT / f"predictions/p4_6/p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_{dataset}.tsv",
        )
        rows.append(paired_bootstrap_auc(dataset, paired, args.bootstrap, args.permutations, rng))
    return rows


def row_from_metrics(
    comparison_axis: str,
    variant: str,
    dataset: str,
    metrics: dict[str, str],
    *,
    train_setting: str,
    representation: str,
    model: str,
    calibration: str,
    notes: str,
) -> dict[str, object]:
    sota = SOTA.get(dataset, {"auc": math.nan, "mcc": math.nan})
    return {
        "comparison_axis": comparison_axis,
        "variant": variant,
        "train_setting": train_setting,
        "representation": representation,
        "model": model,
        "calibration": calibration,
        "dataset": dataset,
        "threshold": as_float(metrics.get("threshold")),
        "sn": as_float(metrics.get("sn")),
        "sp": as_float(metrics.get("sp")),
        "bacc": as_float(metrics.get("bacc")),
        "mcc": as_float(metrics.get("mcc")),
        "auc": as_float(metrics.get("auc")),
        "aupr": as_float(metrics.get("aupr")),
        "fmax": as_float(metrics.get("fmax")),
        "ece": as_float(metrics.get("ece")),
        "brier": as_float(metrics.get("brier")),
        "nll": as_float(metrics.get("nll")),
        "sota_auc": float(sota["auc"]),
        "auc_gap_vs_sota": as_float(metrics.get("auc")) - float(sota["auc"]),
        "sota_mcc": float(sota["mcc"]),
        "mcc_gap_vs_sota": as_float(metrics.get("mcc")) - float(sota["mcc"]),
        "notes": notes,
    }


def compile_ablation_tables() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    p2_ensemble = read_tsv(ROOT / "results/fusion/P2_ENSEMBLE_COMPARISON.tsv")
    p4_6_representation = read_tsv(ROOT / "results/p4_6/P4_6_REPRESENTATION_UPGRADE_COMPARISON.tsv")
    p4_6_nr25 = read_tsv(ROOT / "results/p4_6/P4_6_NR25_COMPARISON.tsv")
    p4_6_calibration = read_tsv(
        ROOT / "results/p4_6/calibration/p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_calibration_metrics.tsv"
    )

    rows: list[dict[str, object]] = []
    deltas: list[dict[str, object]] = []
    for dataset in DATASETS:
        generic = metric_value(p2_ensemble, "experiment_id", "p2_generic_tcn_3seed_ensemble", dataset)
        region = metric_value(p2_ensemble, "experiment_id", "p2_region_aware_tcn_3seed_ensemble", dataset)
        t12 = metric_value(p4_6_representation, "variant", "p2_region_aware_tcn_3seed_ensemble", dataset)
        t33 = metric_value(
            p4_6_representation,
            "variant",
            "p4_6_region_aware_tcn_esm2_t33_3seed_ensemble",
            dataset,
        )
        raw = calibration_value(p4_6_calibration, "raw", dataset)
        platt = calibration_value(p4_6_calibration, "platt", dataset)
        nr25 = p4_6_nr25[[row["dataset"] for row in p4_6_nr25].index(dataset)]

        rows.extend(
            [
                row_from_metrics(
                    "architecture",
                    "GenericTCN_3seed",
                    dataset,
                    generic,
                    train_setting="DM3000 full",
                    representation="ESM2-t12",
                    model="GenericTCN",
                    calibration="raw",
                    notes="Architecture control without region-aware auxiliary supervision.",
                ),
                row_from_metrics(
                    "architecture",
                    "RegionAwareTCN_aux_3seed",
                    dataset,
                    region,
                    train_setting="DM3000 full",
                    representation="ESM2-t12",
                    model="RegionAwareTCN",
                    calibration="raw",
                    notes="Region-aware model with SDR/LDR and terminal/internal auxiliary heads.",
                ),
                row_from_metrics(
                    "representation",
                    "RegionAwareTCN_aux_t12_3seed",
                    dataset,
                    t12,
                    train_setting="DM3000 full",
                    representation="ESM2-t12",
                    model="RegionAwareTCN",
                    calibration="raw",
                    notes="Previous frozen PLM representation.",
                ),
                row_from_metrics(
                    "representation",
                    "RegionAwareTCN_aux_t33_3seed",
                    dataset,
                    t33,
                    train_setting="DM3000 full",
                    representation="ESM2-t33",
                    model="RegionAwareTCN",
                    calibration="raw",
                    notes="P4.6 selected representation-upgraded model.",
                ),
                row_from_metrics(
                    "calibration",
                    "RegionAwareTCN_aux_t33_raw",
                    dataset,
                    raw,
                    train_setting="DM3000 full",
                    representation="ESM2-t33",
                    model="RegionAwareTCN",
                    calibration="raw",
                    notes="Uncalibrated t33 ensemble scores.",
                ),
                row_from_metrics(
                    "calibration",
                    "RegionAwareTCN_aux_t33_platt",
                    dataset,
                    platt,
                    train_setting="DM3000 full",
                    representation="ESM2-t33",
                    model="RegionAwareTCN",
                    calibration="Platt",
                    notes="Validation-fitted Platt calibration.",
                ),
                row_from_metrics(
                    "homology",
                    "RegionAwareTCN_aux_t33_full",
                    dataset,
                    t33,
                    train_setting="DM3000 full",
                    representation="ESM2-t33",
                    model="RegionAwareTCN",
                    calibration="raw",
                    notes="Full DM3000 benchmark setting.",
                ),
                row_from_metrics(
                    "homology",
                    f"RegionAwareTCN_aux_t33_NR25_vs_{dataset}",
                    dataset,
                    nr25,
                    train_setting=f"DM3000 NR25 vs {dataset}",
                    representation="ESM2-t33",
                    model="RegionAwareTCN",
                    calibration="raw",
                    notes="Target-specific NR25 low-homology setting.",
                ),
            ]
        )

        def delta_row(axis: str, reference: dict[str, str], candidate: dict[str, str], label: str) -> None:
            deltas.append(
                {
                    "comparison_axis": axis,
                    "dataset": dataset,
                    "comparison": label,
                    "reference_auc": as_float(reference.get("auc")),
                    "candidate_auc": as_float(candidate.get("auc")),
                    "auc_delta": as_float(candidate.get("auc")) - as_float(reference.get("auc")),
                    "reference_aupr": as_float(reference.get("aupr")),
                    "candidate_aupr": as_float(candidate.get("aupr")),
                    "aupr_delta": as_float(candidate.get("aupr")) - as_float(reference.get("aupr")),
                    "reference_mcc": as_float(reference.get("mcc")),
                    "candidate_mcc": as_float(candidate.get("mcc")),
                    "mcc_delta": as_float(candidate.get("mcc")) - as_float(reference.get("mcc")),
                    "reference_ece": as_float(reference.get("ece")),
                    "candidate_ece": as_float(candidate.get("ece")),
                    "ece_delta": as_float(candidate.get("ece")) - as_float(reference.get("ece")),
                    "reference_brier": as_float(reference.get("brier")),
                    "candidate_brier": as_float(candidate.get("brier")),
                    "brier_delta": as_float(candidate.get("brier")) - as_float(reference.get("brier")),
                    "interpretation": "",
                }
            )

        delta_row("architecture", generic, region, "RegionAwareTCN_aux_t12_3seed minus GenericTCN_t12_3seed")
        delta_row("representation", t12, t33, "ESM2-t33 RegionAwareTCN minus ESM2-t12 RegionAwareTCN")
        delta_row("calibration", raw, platt, "Platt-calibrated t33 minus raw t33")
        delta_row("homology", t33, nr25, f"NR25-vs-{dataset} t33 minus full-train t33")

    for row in deltas:
        axis = row["comparison_axis"]
        if axis == "architecture":
            row["interpretation"] = "Region-aware supervision helps MXD494/DISORDER723 but is not uniformly beneficial for SL329."
        elif axis == "representation":
            row["interpretation"] = "ESM2-t33 is the decisive ranking-performance upgrade, especially on DISORDER723."
        elif axis == "calibration":
            row["interpretation"] = "Platt preserves ranking metrics and improves probability quality."
        elif axis == "homology":
            row["interpretation"] = "NR25 remains competitive but is not uniformly SOTA."
    return rows, deltas


def compile_uncertainty_summary() -> list[dict[str, object]]:
    rows = read_tsv(ROOT / "results/p4_6/calibration/p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_uncertainty_error_enrichment.tsv")
    selected = []
    for row in rows:
        if row["method"] != "platt":
            continue
        if row["dataset"] not in DATASETS and row["dataset"] != "DM1229_Validation":
            continue
        if row["top_uncertain_fraction"] not in {"0.010000", "0.050000", "0.100000", "0.200000"}:
            continue
        selected.append(row)
    return selected


def compile_stratified_summary() -> list[dict[str, object]]:
    selected_specs = [
        ("overall", "all_known", "Overall"),
        ("positive_region_length_type", "SDR", "SDR"),
        ("positive_region_length_type", "LDR", "LDR"),
        ("positive_region_location", "terminal", "Terminal IDR"),
        ("positive_region_location", "internal", "Internal IDR"),
        ("residue_zone", "middle", "Middle residues"),
        ("protein_disorder_content_bin", "0-5", "0-5% disorder proteins"),
        ("protein_length_bin", ">1000", ">1000 aa proteins"),
    ]
    output: list[dict[str, object]] = []
    for dataset in DATASETS:
        rows = read_tsv(ROOT / f"results/p4_6/p4_6_region_aware_tcn_esm2_t33_3seed_ensemble_{dataset}_stratified.tsv")
        for group, stratum, label in selected_specs:
            matches = [row for row in rows if row["stratum_group"] == group and row["stratum"] == stratum]
            if not matches:
                continue
            row = matches[0]
            output.append(
                {
                    "dataset": dataset,
                    "stratum_group": group,
                    "stratum": stratum,
                    "display_stratum": label,
                    "residues": as_float(row["residues"]),
                    "positives": as_float(row["positives"]),
                    "auc": as_float(row["auc"]),
                    "aupr": as_float(row["aupr"]),
                    "mcc": as_float(row["mcc"]),
                    "fmax": as_float(row["fmax"]),
                    "sn": as_float(row["sn"]),
                    "sp": as_float(row["sp"]),
                }
            )
    return output


def plot_uncertainty(rows: list[dict[str, object]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = ["SL329", "MXD494", "DISORDER723"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        dataset_rows.sort(key=lambda row: float(row["top_uncertain_fraction"]))
        xs = [float(row["top_uncertain_fraction"]) * 100 for row in dataset_rows]
        ys = [float(row["error_enrichment"]) for row in dataset_rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=dataset)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Most uncertain residues selected (%)")
    ax.set_ylabel("Error enrichment")
    ax.set_title("Platt-calibrated t33 ensemble uncertainty")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "P5_T33_PLATT_UNCERTAINTY_ERROR_ENRICHMENT.pdf")
    fig.savefig(out_dir / "P5_T33_PLATT_UNCERTAINTY_ERROR_ENRICHMENT.png", dpi=300)
    plt.close(fig)


def plot_stratified(rows: list[dict[str, object]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    strata = ["Overall", "SDR", "LDR", "Terminal IDR", "Internal IDR", "Middle residues", "0-5% disorder proteins"]
    datasets = ["SL329", "MXD494", "DISORDER723"]
    x = np.arange(len(strata), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for offset, dataset in enumerate(datasets):
        values = []
        for stratum in strata:
            matches = [row for row in rows if row["dataset"] == dataset and row["display_stratum"] == stratum]
            values.append(float(matches[0]["auc"]) if matches else math.nan)
        ax.bar(x + (offset - 1) * width, values, width=width, label=dataset)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.70, 1.00)
    ax.set_title("t33 RegionAwareTCN hard-case stratified AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(strata, rotation=30, ha="right")
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "P5_T33_HARD_CASE_STRATIFIED_AUC.pdf")
    fig.savefig(out_dir / "P5_T33_HARD_CASE_STRATIFIED_AUC.png", dpi=300)
    plt.close(fig)


def plot_mechanism(out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    ax.set_axis_off()

    boxes = [
        ("P4.5 failure analysis\nDISORDER723 internal IDR is the main gap", 0.04, 0.58, 0.24, 0.24),
        ("Representation upgrade\nESM2-t12 -> ESM2-t33", 0.37, 0.58, 0.22, 0.24),
        ("RegionAwareTCN\nSDR/LDR + terminal/internal auxiliary supervision", 0.70, 0.58, 0.25, 0.24),
        ("Leakage-aware evaluation\nFull DM3000 plus target-specific NR25", 0.19, 0.15, 0.25, 0.22),
        ("Probability reliability\nValidation-fitted Platt calibration", 0.56, 0.15, 0.25, 0.22),
    ]

    for text, x, y, w, h in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=1.2,
            edgecolor="#334155",
            facecolor="#f8fafc",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color="#0f172a")

    arrows = [
        ((0.28, 0.70), (0.37, 0.70)),
        ((0.59, 0.70), (0.70, 0.70)),
        ((0.82, 0.58), (0.69, 0.37)),
        ((0.48, 0.58), (0.35, 0.37)),
        ((0.48, 0.58), (0.62, 0.37)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="#475569"))

    ax.text(
        0.50,
        0.94,
        "P5 manuscript evidence chain: performance, robustness, and calibrated probabilities",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.50,
        0.04,
        "Claim boundary: AUC-level SOTA in full-train setting; competitive but not uniformly SOTA under NR25.",
        ha="center",
        va="center",
        fontsize=10,
        color="#334155",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "P5_EVIDENCE_CHAIN_MECHANISM.pdf")
    fig.savefig(out_dir / "P5_EVIDENCE_CHAIN_MECHANISM.png", dpi=300)
    plt.close(fig)


def markdown_table(rows: list[dict[str, object]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def write_summary(
    path: Path,
    bootstrap_rows: list[dict[str, object]],
    uncertainty_rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top10 = [
        row
        for row in uncertainty_rows
        if row["method"] == "platt" and row["top_uncertain_fraction"] == "0.100000"
    ]
    representation_deltas = [row for row in delta_rows if row["comparison_axis"] == "representation"]
    calibration_deltas = [row for row in delta_rows if row["comparison_axis"] == "calibration"]
    homology_deltas = [row for row in delta_rows if row["comparison_axis"] == "homology"]

    text = f"""# P5 Evidence Supplement Summary

Date: 2026-08-03

## Scope

This P5 evidence bundle adds the three manuscript-level checks requested after P4.6:

1. Protein-level bootstrap confidence intervals and paired permutation tests for the ESM2-t33 upgrade over the ESM2-t12 RegionAwareTCN ensemble.
2. t33 Platt-calibrated uncertainty-error enrichment and hard-case stratified figures.
3. A paper-level ablation table covering representation, architecture, calibration, and NR25 robustness.

## 1. Statistical Support for the Representation Upgrade

The statistical test is paired at the protein level for t33 versus t12 because both models predict the same benchmark proteins. External SOTA methods provide only reported aggregate metrics, so they can only be compared against the t33 bootstrap confidence interval, not by a paired test.

{markdown_table(bootstrap_rows, [
    "dataset",
    "reference_auc",
    "candidate_auc",
    "auc_delta",
    "auc_delta_ci_low",
    "auc_delta_ci_high",
    "paired_permutation_p_one_sided",
    "sota_auc",
    "candidate_gap_vs_sota",
    "candidate_auc_ci_low_gt_sota",
])}

Interpretation:

- The ESM2-t33 upgrade has positive AUC point deltas on all three datasets.
- The protein-level statistical support is strong on DISORDER723 only; SL329 and MXD494 are positive but their protein-level bootstrap CIs cross zero.
- The DISORDER723 gain is the strongest mechanistic result because it directly closes the P4.5 failure gap.
- Against aggregate literature SOTA, the point estimates are above SOTA on all three datasets, but bootstrap lower bounds do not exceed the SOTA scalar; this should be phrased as AUC-level SOTA point performance, not statistical dominance over unavailable external predictions.

## 2. t33 Uncertainty and Hard-Case Evidence

Top 10% Platt uncertainty enrichment:

{markdown_table(top10, [
    "dataset",
    "overall_error_rate",
    "top_uncertain_error_rate",
    "error_enrichment",
    "mean_uncertainty_top",
    "mean_uncertainty_all",
])}

Generated figures:

- `figures/p5/P5_T33_PLATT_UNCERTAINTY_ERROR_ENRICHMENT.pdf`
- `figures/p5/P5_T33_HARD_CASE_STRATIFIED_AUC.pdf`

## 3. Paper-Level Ablation Conclusions

Representation deltas:

{markdown_table(representation_deltas, [
    "dataset",
    "auc_delta",
    "aupr_delta",
    "mcc_delta",
    "interpretation",
])}

Calibration deltas:

{markdown_table(calibration_deltas, [
    "dataset",
    "auc_delta",
    "ece_delta",
    "brier_delta",
    "interpretation",
])}

NR25 deltas:

{markdown_table(homology_deltas, [
    "dataset",
    "auc_delta",
    "mcc_delta",
    "interpretation",
])}

Generated tables:

- `results/p5/P5_T33_AUC_BOOTSTRAP_CI.tsv`
- `results/p5/P5_T33_PLATT_UNCERTAINTY_ERROR_ENRICHMENT.tsv`
- `results/p5/P5_T33_HARD_CASE_STRATIFIED_SUMMARY.tsv`
- `results/p5/P5_PAPER_LEVEL_ABLATION_TABLE.tsv`
- `results/p5/P5_PAPER_LEVEL_KEY_DELTAS.tsv`

Generated mechanism figure:

- `figures/p5/P5_EVIDENCE_CHAIN_MECHANISM.pdf`

## Decision

The added evidence strengthens P5. The safest manuscript framing is:

Sequence-only RegionAwareTCN with ESM2-t33 achieves AUC-level SOTA point performance in the full DM3000 benchmark, provides strong calibrated probability estimates, and remains competitive under NR25 low-homology evaluation. The manuscript should explicitly state that NR25 performance is not uniformly SOTA and that DISORDER723 MCC/internal IDR classification remain the main limitations.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = ROOT / "results/p5"
    figure_dir = ROOT / "figures/p5"

    bootstrap_rows = compile_bootstrap(args)
    ablation_rows, delta_rows = compile_ablation_tables()
    uncertainty_rows = compile_uncertainty_summary()
    stratified_rows = compile_stratified_summary()

    write_tsv(
        result_dir / "P5_T33_AUC_BOOTSTRAP_CI.tsv",
        bootstrap_rows,
        [
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
        ],
    )
    write_tsv(
        result_dir / "P5_PAPER_LEVEL_ABLATION_TABLE.tsv",
        ablation_rows,
        [
            "comparison_axis",
            "variant",
            "train_setting",
            "representation",
            "model",
            "calibration",
            "dataset",
            "threshold",
            "sn",
            "sp",
            "bacc",
            "mcc",
            "auc",
            "aupr",
            "fmax",
            "ece",
            "brier",
            "nll",
            "sota_auc",
            "auc_gap_vs_sota",
            "sota_mcc",
            "mcc_gap_vs_sota",
            "notes",
        ],
    )
    write_tsv(
        result_dir / "P5_PAPER_LEVEL_KEY_DELTAS.tsv",
        delta_rows,
        [
            "comparison_axis",
            "dataset",
            "comparison",
            "reference_auc",
            "candidate_auc",
            "auc_delta",
            "reference_aupr",
            "candidate_aupr",
            "aupr_delta",
            "reference_mcc",
            "candidate_mcc",
            "mcc_delta",
            "reference_ece",
            "candidate_ece",
            "ece_delta",
            "reference_brier",
            "candidate_brier",
            "brier_delta",
            "interpretation",
        ],
    )
    write_tsv(
        result_dir / "P5_T33_PLATT_UNCERTAINTY_ERROR_ENRICHMENT.tsv",
        uncertainty_rows,
        [
            "experiment_id",
            "method",
            "dataset",
            "top_uncertain_fraction",
            "selected_residues",
            "overall_error_rate",
            "top_uncertain_error_rate",
            "error_enrichment",
            "mean_uncertainty_top",
            "mean_uncertainty_all",
        ],
    )
    write_tsv(
        result_dir / "P5_T33_HARD_CASE_STRATIFIED_SUMMARY.tsv",
        stratified_rows,
        [
            "dataset",
            "stratum_group",
            "stratum",
            "display_stratum",
            "residues",
            "positives",
            "auc",
            "aupr",
            "mcc",
            "fmax",
            "sn",
            "sp",
        ],
    )

    plot_uncertainty(uncertainty_rows, figure_dir)
    plot_stratified(stratified_rows, figure_dir)
    plot_mechanism(figure_dir)
    write_summary(ROOT / "reports/P5_EVIDENCE_SUPPLEMENT_SUMMARY.md", bootstrap_rows, uncertainty_rows, delta_rows)


if __name__ == "__main__":
    main()
