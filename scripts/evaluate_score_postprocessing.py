#!/usr/bin/env python3
"""Tune simple score post-processing on validation and evaluate on test sets."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score  # noqa: E402

from evaluate_disorder_predictions import parse_labeled_fasta, read_prediction_tsv  # noqa: E402


DEFAULT_CANDIDATES = (
    (
        "p2_generic_tcn_3seed_ensemble",
        Path("predictions/fusion/p2_generic_tcn_3seed_ensemble_DM1229_Validation.tsv"),
        {
            "SL329": Path("predictions/fusion/p2_generic_tcn_3seed_ensemble_SL329.tsv"),
            "MXD494": Path("predictions/fusion/p2_generic_tcn_3seed_ensemble_MXD494.tsv"),
            "DISORDER723": Path("predictions/fusion/p2_generic_tcn_3seed_ensemble_DISORDER723.tsv"),
        },
    ),
    (
        "p2_region_aware_tcn_3seed_ensemble",
        Path("predictions/fusion/p2_region_aware_tcn_3seed_ensemble_DM1229_Validation.tsv"),
        {
            "SL329": Path("predictions/fusion/p2_region_aware_tcn_3seed_ensemble_SL329.tsv"),
            "MXD494": Path("predictions/fusion/p2_region_aware_tcn_3seed_ensemble_MXD494.tsv"),
            "DISORDER723": Path("predictions/fusion/p2_region_aware_tcn_3seed_ensemble_DISORDER723.tsv"),
        },
    ),
    (
        "p2_tcn_6model_ensemble",
        Path("predictions/fusion/p2_tcn_6model_ensemble_DM1229_Validation.tsv"),
        {
            "SL329": Path("predictions/fusion/p2_tcn_6model_ensemble_SL329.tsv"),
            "MXD494": Path("predictions/fusion/p2_tcn_6model_ensemble_MXD494.tsv"),
            "DISORDER723": Path("predictions/fusion/p2_tcn_6model_ensemble_DISORDER723.tsv"),
        },
    ),
)

DEFAULT_TESTS = {
    "SL329": Path("data/SL329_test.fasta"),
    "MXD494": Path("data/MXD494_test.fasta"),
    "DISORDER723": Path("data/DISORDER723_test.fasta"),
}


@dataclass(frozen=True)
class Transform:
    name: str
    domain: str
    window: int
    raw_weight: float

    @property
    def transform_id(self) -> str:
        if self.name == "identity":
            return "identity"
        return f"{self.domain}_smooth_w{self.window}_raw{self.raw_weight:.2f}"


def fmax(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if int(np.sum(labels)) == 0:
        return math.nan, math.nan
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return math.nan, math.nan
    usable_precision = precision[:-1]
    usable_recall = recall[:-1]
    denom = usable_precision + usable_recall
    f1 = np.divide(
        2.0 * usable_precision * usable_recall,
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    best_index = int(np.argmax(f1))
    return float(f1[best_index]), float(thresholds[best_index])


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    positives = scores >= threshold
    label_positive = labels == 1
    tp = int(np.sum(label_positive & positives))
    fp = int(np.sum((~label_positive) & positives))
    tn = int(np.sum((~label_positive) & (~positives)))
    fn = int(np.sum(label_positive & (~positives)))
    sn = tp / (tp + fn) if tp + fn else math.nan
    sp = tn / (tn + fp) if tn + fp else math.nan
    bacc = (sn + sp) / 2.0 if math.isfinite(sn) and math.isfinite(sp) else math.nan
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else math.nan
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "sn": sn, "sp": sp, "bacc": bacc, "mcc": mcc}


def clamp_probability(value: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, value))


def logit(value: float) -> float:
    value = clamp_probability(value)
    return math.log(value / (1.0 - value))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    half = window // 2
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    output: list[float] = []
    for index in range(len(values)):
        start = max(0, index - half)
        end = min(len(values), index + half + 1)
        output.append((prefix[end] - prefix[start]) / (end - start))
    return output


def apply_transform_to_scores(scores: list[float], transform: Transform) -> list[float]:
    if transform.name == "identity":
        return list(scores)
    raw = [float(score) for score in scores]
    if transform.domain == "score":
        smoothed = moving_average(raw, transform.window)
    elif transform.domain == "logit":
        smoothed_logits = moving_average([logit(score) for score in raw], transform.window)
        smoothed = [sigmoid(value) for value in smoothed_logits]
    else:
        raise ValueError(f"unsupported transform domain: {transform.domain}")
    return [
        transform.raw_weight * original + (1.0 - transform.raw_weight) * smooth
        for original, smooth in zip(raw, smoothed)
    ]


def apply_transform(predictions: dict[str, list[float]], transform: Transform) -> dict[str, list[float]]:
    return {
        protein_id: apply_transform_to_scores(scores, transform)
        for protein_id, scores in predictions.items()
    }


def collect_labels_scores(
    records: list[dict[str, object]],
    predictions: dict[str, list[float]],
) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    scores: list[float] = []
    for record in records:
        protein_id = str(record["id"])
        item_scores = predictions.get(protein_id, predictions.get(protein_id.split()[0]))
        if item_scores is None:
            raise ValueError(f"missing predictions for {protein_id}")
        if len(item_scores) != len(str(record["sequence"])):
            raise ValueError(f"{protein_id}: length mismatch")
        for label, score in zip(record["labels"], item_scores):
            if int(label) == -1:
                continue
            labels.append(int(label))
            scores.append(float(score))
    return labels, scores


def evaluate(
    records: list[dict[str, object]],
    predictions: dict[str, list[float]],
    threshold: float | None = None,
) -> dict[str, float | int]:
    label_list, score_list = collect_labels_scores(records, predictions)
    labels = np.asarray(label_list, dtype=np.int8)
    scores = np.asarray(score_list, dtype=np.float64)
    fmax_value, fmax_threshold = fmax(labels, scores)
    if threshold is None:
        threshold = fmax_threshold
    threshold_row = threshold_metrics(labels, scores, threshold)
    positives = int(np.sum(labels))
    return {
        "residues": int(labels.size),
        "positives": positives,
        "negatives": int(labels.size - positives),
        "threshold": threshold,
        **threshold_row,
        "auc": float(roc_auc_score(labels, scores)) if positives and positives < labels.size else math.nan,
        "aupr": float(average_precision_score(labels, scores)) if positives else math.nan,
        "fmax": fmax_value,
        "fmax_threshold": fmax_threshold,
    }


def transform_grid() -> list[Transform]:
    transforms = [Transform("identity", "score", 1, 1.0)]
    for domain in ("score", "logit"):
        for window in (3, 5, 7, 9, 15, 21, 31):
            for raw_weight in (0.0, 0.25, 0.5, 0.75):
                transforms.append(Transform("smooth", domain, window, raw_weight))
    return transforms


def format_value(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.6f}"
    return value


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fieldnames})


def write_prediction_tsv(path: Path, predictions: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "scores"])
        for protein_id, scores in predictions.items():
            writer.writerow([protein_id, "[" + ", ".join(f"{score:.8g}" for score in scores) + "]"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-labels", type=Path, default=Path("data/DM1229_Validation.fasta"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/postprocess"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("predictions/postprocess"))
    parser.add_argument("--primary-metric", choices=("auc", "fmax", "mcc", "aupr"), default="auc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_records = parse_labeled_fasta(args.validation_labels)
    test_records = {dataset: parse_labeled_fasta(path) for dataset, path in DEFAULT_TESTS.items()}
    transforms = transform_grid()

    grid_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for model_id, validation_path, test_paths in DEFAULT_CANDIDATES:
        validation_predictions_raw = read_prediction_tsv(validation_path, "\t")
        validation_metrics_by_transform: list[tuple[Transform, dict[str, float | int], dict[str, list[float]]]] = []
        for transform in transforms:
            transformed = apply_transform(validation_predictions_raw, transform)
            metrics = evaluate(validation_records, transformed)
            validation_metrics_by_transform.append((transform, metrics, transformed))
            grid_rows.append(
                {
                    "model_id": model_id,
                    "selection_split": "DM1229_Validation",
                    "transform_id": transform.transform_id,
                    "domain": transform.domain,
                    "window": transform.window,
                    "raw_weight": transform.raw_weight,
                    **{f"validation_{key}": value for key, value in metrics.items()},
                }
            )

        selected_transform, selected_validation_metrics, selected_validation_predictions = max(
            validation_metrics_by_transform,
            key=lambda item: (
                float(item[1][args.primary_metric]),
                float(item[1]["fmax"]),
                float(item[1]["mcc"]),
            ),
        )
        identity_metrics = next(metrics for transform, metrics, _ in validation_metrics_by_transform if transform.name == "identity")
        selected_rows.append(
            {
                "model_id": model_id,
                "primary_metric": args.primary_metric,
                "selected_transform_id": selected_transform.transform_id,
                "selected_domain": selected_transform.domain,
                "selected_window": selected_transform.window,
                "selected_raw_weight": selected_transform.raw_weight,
                "identity_validation_auc": identity_metrics["auc"],
                "selected_validation_auc": selected_validation_metrics["auc"],
                "identity_validation_aupr": identity_metrics["aupr"],
                "selected_validation_aupr": selected_validation_metrics["aupr"],
                "identity_validation_mcc": identity_metrics["mcc"],
                "selected_validation_mcc": selected_validation_metrics["mcc"],
                "identity_validation_fmax": identity_metrics["fmax"],
                "selected_validation_fmax": selected_validation_metrics["fmax"],
                "selected_threshold": selected_validation_metrics["fmax_threshold"],
            }
        )
        write_prediction_tsv(
            args.prediction_dir / f"{model_id}_{selected_transform.transform_id}_DM1229_Validation.tsv",
            selected_validation_predictions,
        )

        for dataset, test_path in test_paths.items():
            raw_test_predictions = read_prediction_tsv(test_path, "\t")
            identity_test_metrics = evaluate(test_records[dataset], raw_test_predictions, selected_validation_metrics["fmax_threshold"])
            selected_test_predictions = apply_transform(raw_test_predictions, selected_transform)
            selected_test_metrics = evaluate(test_records[dataset], selected_test_predictions, selected_validation_metrics["fmax_threshold"])
            write_prediction_tsv(
                args.prediction_dir / f"{model_id}_{selected_transform.transform_id}_{dataset}.tsv",
                selected_test_predictions,
            )
            comparison_rows.append(
                {
                    "model_id": model_id,
                    "dataset": dataset,
                    "primary_metric": args.primary_metric,
                    "selected_transform_id": selected_transform.transform_id,
                    "selected_threshold": selected_validation_metrics["fmax_threshold"],
                    **{f"identity_{key}": value for key, value in identity_test_metrics.items()},
                    **{f"selected_{key}": value for key, value in selected_test_metrics.items()},
                    "auc_delta": float(selected_test_metrics["auc"]) - float(identity_test_metrics["auc"]),
                    "aupr_delta": float(selected_test_metrics["aupr"]) - float(identity_test_metrics["aupr"]),
                    "mcc_delta": float(selected_test_metrics["mcc"]) - float(identity_test_metrics["mcc"]),
                    "fmax_delta": float(selected_test_metrics["fmax"]) - float(identity_test_metrics["fmax"]),
                }
            )

    grid_fields = [
        "model_id",
        "selection_split",
        "transform_id",
        "domain",
        "window",
        "raw_weight",
        "validation_residues",
        "validation_positives",
        "validation_negatives",
        "validation_threshold",
        "validation_tp",
        "validation_fp",
        "validation_tn",
        "validation_fn",
        "validation_sn",
        "validation_sp",
        "validation_bacc",
        "validation_mcc",
        "validation_auc",
        "validation_aupr",
        "validation_fmax",
        "validation_fmax_threshold",
    ]
    selected_fields = [
        "model_id",
        "primary_metric",
        "selected_transform_id",
        "selected_domain",
        "selected_window",
        "selected_raw_weight",
        "identity_validation_auc",
        "selected_validation_auc",
        "identity_validation_aupr",
        "selected_validation_aupr",
        "identity_validation_mcc",
        "selected_validation_mcc",
        "identity_validation_fmax",
        "selected_validation_fmax",
        "selected_threshold",
    ]
    comparison_fields = [
        "model_id",
        "dataset",
        "primary_metric",
        "selected_transform_id",
        "selected_threshold",
        "identity_auc",
        "selected_auc",
        "auc_delta",
        "identity_aupr",
        "selected_aupr",
        "aupr_delta",
        "identity_mcc",
        "selected_mcc",
        "mcc_delta",
        "identity_fmax",
        "selected_fmax",
        "fmax_delta",
        "identity_sn",
        "selected_sn",
        "identity_sp",
        "selected_sp",
        "identity_bacc",
        "selected_bacc",
    ]
    write_tsv(args.out_dir / "P4_5_POSTPROCESS_GRID.tsv", grid_rows, grid_fields)
    write_tsv(args.out_dir / "P4_5_POSTPROCESS_SELECTED.tsv", selected_rows, selected_fields)
    write_tsv(args.out_dir / "P4_5_POSTPROCESS_COMPARISON.tsv", comparison_rows, comparison_fields)

    for row in comparison_rows:
        if row["dataset"] == "DISORDER723":
            print(
                f"{row['model_id']}\t{row['selected_transform_id']}\t"
                f"AUC {format_value(row['identity_auc'])}->{format_value(row['selected_auc'])}\t"
                f"MCC {format_value(row['identity_mcc'])}->{format_value(row['selected_mcc'])}"
            )


if __name__ == "__main__":
    main()
