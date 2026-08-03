#!/usr/bin/env python3
"""Fit probability calibration on validation predictions and apply it to test sets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from evaluate_disorder_predictions import build_id_lookup, evaluate, parse_labeled_fasta, read_prediction_tsv, roc_auc


EPS = 1e-7


@dataclass
class TestSpec:
    name: str
    labels: Path
    predictions: Path


class Calibrator:
    name = "base"

    def fit(self, labels: np.ndarray, scores: np.ndarray) -> None:
        return None

    def transform(self, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> dict[str, float | str]:
        return {}


class RawCalibrator(Calibrator):
    name = "raw"

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.clip(scores.astype(np.float64, copy=False), EPS, 1.0 - EPS)


class TemperatureCalibrator(Calibrator):
    name = "temperature"

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, labels: np.ndarray, scores: np.ndarray) -> None:
        logits = torch.tensor(logit(scores), dtype=torch.float32)
        y = torch.tensor(labels.astype(np.float32), dtype=torch.float32)
        log_temperature = torch.zeros((), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            temperature = torch.exp(log_temperature).clamp(min=0.05, max=100.0)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / temperature, y)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_temperature).detach().clamp(min=0.05, max=100.0).cpu())

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return sigmoid(logit(scores) / self.temperature)

    def parameters(self) -> dict[str, float | str]:
        return {"temperature": self.temperature}


class PlattCalibrator(Calibrator):
    name = "platt"

    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)

    def fit(self, labels: np.ndarray, scores: np.ndarray) -> None:
        self.model.fit(logit(scores).reshape(-1, 1), labels.astype(np.int32))

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(logit(scores).reshape(-1, 1))[:, 1]

    def parameters(self) -> dict[str, float | str]:
        return {
            "coef": float(self.model.coef_[0, 0]),
            "intercept": float(self.model.intercept_[0]),
        }


class IsotonicCalibrator(Calibrator):
    name = "isotonic"

    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, labels: np.ndarray, scores: np.ndarray) -> None:
        self.model.fit(np.clip(scores, EPS, 1.0 - EPS), labels.astype(np.float64))

    def transform(self, scores: np.ndarray) -> np.ndarray:
        calibrated = self.model.predict(np.clip(scores, EPS, 1.0 - EPS))
        return np.clip(np.asarray(calibrated, dtype=np.float64), EPS, 1.0 - EPS)

    def parameters(self) -> dict[str, float | str]:
        return {"threshold_count": int(len(self.model.X_thresholds_))}


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores.astype(np.float64, copy=False), EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order]
    tp = np.cumsum(ranked_labels == 1)
    ranks = np.arange(1, len(labels) + 1)
    return float(np.sum((tp / ranks)[ranked_labels == 1]) / positives)


def fmax(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return math.nan, math.nan
    order = np.argsort(-scores, kind="mergesort")
    ranked_scores = scores[order]
    ranked_labels = labels[order]
    tp = 0
    fp = 0
    best_f1 = 0.0
    best_threshold = 0.5
    i = 0
    while i < len(order):
        threshold = float(ranked_scores[i])
        j = i
        while j < len(order) and ranked_scores[j] == ranked_scores[i]:
            if ranked_labels[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / positives
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        i = j
    return best_f1, best_threshold


def parse_test_spec(text: str) -> TestSpec:
    parts = text.split(":", 2)
    if len(parts) != 3:
        raise ValueError("--test entries must use name:labels_fasta:predictions_tsv")
    return TestSpec(parts[0], Path(parts[1]), Path(parts[2]))


def collect_known_labels_scores(
    records: list[dict[str, object]], predictions: dict[str, list[float]]
) -> tuple[np.ndarray, np.ndarray]:
    lookup = build_id_lookup(records)
    labels_out: list[int] = []
    scores_out: list[float] = []
    for record in records:
        protein_id = str(record["id"])
        scores = predictions.get(protein_id)
        if scores is None:
            scores = predictions.get(protein_id.split()[0])
        if scores is None:
            raise ValueError(f"missing predictions for {protein_id}")
        sequence = str(record["sequence"])
        labels = record["labels"]
        assert isinstance(labels, list)
        if len(scores) != len(sequence):
            raise ValueError(f"prediction length mismatch for {protein_id}: {len(scores)} != {len(sequence)}")
        for label, score in zip(labels, scores):
            if label == -1:
                continue
            labels_out.append(int(label))
            scores_out.append(float(score))
    unknown_prediction_ids = sorted(set(predictions) - set(lookup))
    if unknown_prediction_ids:
        examples = ", ".join(unknown_prediction_ids[:5])
        raise ValueError(f"prediction file contains IDs not present in labels: {examples}")
    return np.asarray(labels_out, dtype=np.int8), np.asarray(scores_out, dtype=np.float64)


def transform_predictions(predictions: dict[str, list[float]], calibrator: Calibrator) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    for protein_id, scores in predictions.items():
        transformed = calibrator.transform(np.asarray(scores, dtype=np.float64))
        output[protein_id] = [float(value) for value in transformed]
    return output


def write_prediction_tsv(path: Path, predictions: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "scores"])
        for protein_id, scores in predictions.items():
            writer.writerow([protein_id, "[" + ", ".join(f"{score:.8g}" for score in scores) + "]"])


def calibration_metrics(labels: np.ndarray, scores: np.ndarray, bins: int) -> dict[str, float]:
    clipped = np.clip(scores, EPS, 1.0 - EPS)
    brier = float(np.mean((clipped - labels) ** 2))
    nll = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped)))
    bin_rows, ece, mce = reliability_bins(labels, clipped, bins)
    del bin_rows
    return {"brier": brier, "nll": nll, "ece": ece, "mce": mce}


def reliability_bins(labels: np.ndarray, scores: np.ndarray, bins: int) -> tuple[list[dict[str, object]], float, float]:
    rows: list[dict[str, object]] = []
    n = len(labels)
    ece = 0.0
    mce = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        if index == bins - 1:
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores >= low) & (scores < high)
        count = int(np.sum(mask))
        positives = int(np.sum(labels[mask] == 1)) if count else 0
        mean_score = float(np.mean(scores[mask])) if count else math.nan
        empirical_rate = float(np.mean(labels[mask])) if count else math.nan
        gap = abs(mean_score - empirical_rate) if count else math.nan
        if count:
            ece += (count / n) * gap
            mce = max(mce, gap)
        rows.append(
            {
                "bin_index": index,
                "bin_low": low,
                "bin_high": high,
                "count": count,
                "positives": positives,
                "mean_score": mean_score,
                "empirical_positive_rate": empirical_rate,
                "abs_gap": gap,
            }
        )
    return rows, float(ece), float(mce)


def entropy_uncertainty(scores: np.ndarray) -> np.ndarray:
    p = np.clip(scores.astype(np.float64, copy=False), EPS, 1.0 - EPS)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return entropy / math.log(2.0)


def threshold_metrics_from_arrays(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = scores >= threshold
    positives = labels == 1
    negatives = labels == 0
    tp = int(np.sum(predicted & positives))
    fp = int(np.sum(predicted & negatives))
    tn = int(np.sum((~predicted) & negatives))
    fn = int(np.sum((~predicted) & positives))
    sn = tp / (tp + fn) if (tp + fn) else math.nan
    sp = tn / (tn + fp) if (tn + fp) else math.nan
    bacc = (sn + sp) / 2.0 if math.isfinite(sn) and math.isfinite(sp) else math.nan
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else math.nan
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "sn": sn, "sp": sp, "bacc": bacc, "mcc": mcc}


def metrics_row(
    experiment_id: str,
    method: str,
    dataset: str,
    records: list[dict[str, object]],
    predictions: dict[str, list[float]],
    threshold: float,
    bins: int,
    prediction_file: Path,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray, np.ndarray]:
    labels, scores = collect_known_labels_scores(records, predictions)
    threshold_row = threshold_metrics_from_arrays(labels, scores, threshold)
    calibration_row = calibration_metrics(labels, scores, bins)
    fmax_value, fmax_threshold = fmax(labels, scores)
    bin_rows, _, _ = reliability_bins(labels, scores, bins)
    rows = [
        {
            "experiment_id": experiment_id,
            "method": method,
            "dataset": dataset,
            **row,
        }
        for row in bin_rows
    ]
    row = {
        "experiment_id": experiment_id,
        "method": method,
        "dataset": dataset,
        "proteins": len(records),
        "evaluated_residues": len(labels),
        "positives": int(np.sum(labels == 1)),
        "negatives": int(np.sum(labels == 0)),
        "threshold": threshold,
        **threshold_row,
        "auc": roc_auc([int(value) for value in labels], [float(value) for value in scores]),
        "aupr": average_precision(labels, scores),
        "fmax": fmax_value,
        "fmax_threshold": fmax_threshold,
        **calibration_row,
        "prediction_file": str(prediction_file),
    }
    return row, rows, labels, scores


def uncertainty_rows(
    experiment_id: str,
    method: str,
    dataset: str,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> list[dict[str, object]]:
    uncertainty = entropy_uncertainty(scores)
    errors = (scores >= threshold) != (labels == 1)
    overall_error_rate = float(np.mean(errors))
    rows: list[dict[str, object]] = []
    order = np.argsort(-uncertainty, kind="mergesort")
    for fraction in (0.01, 0.05, 0.10, 0.20, 0.50, 1.00):
        selected = max(1, int(math.ceil(len(order) * fraction)))
        subset = order[:selected]
        error_rate = float(np.mean(errors[subset]))
        enrichment = error_rate / overall_error_rate if overall_error_rate > 0 else math.nan
        rows.append(
            {
                "experiment_id": experiment_id,
                "method": method,
                "dataset": dataset,
                "top_uncertain_fraction": fraction,
                "selected_residues": selected,
                "overall_error_rate": overall_error_rate,
                "top_uncertain_error_rate": error_rate,
                "error_enrichment": enrichment,
                "mean_uncertainty_top": float(np.mean(uncertainty[subset])),
                "mean_uncertainty_all": float(np.mean(uncertainty)),
            }
        )
    return rows


def distance_to_unknown(labels: list[int]) -> list[int | None]:
    unknown = [label == -1 for label in labels]
    if not any(unknown):
        return [None] * len(labels)
    inf = len(labels) + 1
    left = [inf] * len(labels)
    last = -inf
    for index, is_unknown in enumerate(unknown):
        if is_unknown:
            last = index
        left[index] = index - last
    right = [inf] * len(labels)
    last = inf
    for index in range(len(labels) - 1, -1, -1):
        if unknown[index]:
            last = index
        right[index] = last - index
    return [min(left[index], right[index]) for index in range(len(labels))]


def unknown_neighbor_rows(
    experiment_id: str,
    method: str,
    dataset: str,
    records: list[dict[str, object]],
    predictions: dict[str, list[float]],
    threshold: float,
) -> list[dict[str, object]]:
    groups: dict[str, dict[str, list[float] | list[int]]] = {
        "unknown_residue": {"scores": [], "uncertainties": [], "labels": [], "errors": []},
        "known_within_5aa_unknown": {"scores": [], "uncertainties": [], "labels": [], "errors": []},
        "known_far_from_unknown": {"scores": [], "uncertainties": [], "labels": [], "errors": []},
    }
    for record in records:
        protein_id = str(record["id"])
        scores = predictions.get(protein_id) or predictions.get(protein_id.split()[0])
        if scores is None:
            raise ValueError(f"missing predictions for {protein_id}")
        labels = record["labels"]
        assert isinstance(labels, list)
        distances = distance_to_unknown(labels)
        score_array = np.asarray(scores, dtype=np.float64)
        uncertainty_array = entropy_uncertainty(score_array)
        for index, (label, score, uncertainty) in enumerate(zip(labels, score_array, uncertainty_array)):
            distance = distances[index]
            if label == -1:
                group = "unknown_residue"
            elif distance is not None and distance <= 5:
                group = "known_within_5aa_unknown"
            else:
                group = "known_far_from_unknown"
            groups[group]["scores"].append(float(score))
            groups[group]["uncertainties"].append(float(uncertainty))
            if label != -1:
                groups[group]["labels"].append(int(label))
                groups[group]["errors"].append(int((score >= threshold) != (label == 1)))
    rows: list[dict[str, object]] = []
    for group, values in groups.items():
        scores = np.asarray(values["scores"], dtype=np.float64)
        uncertainties = np.asarray(values["uncertainties"], dtype=np.float64)
        known_labels = np.asarray(values["labels"], dtype=np.float64)
        errors = np.asarray(values["errors"], dtype=np.float64)
        rows.append(
            {
                "experiment_id": experiment_id,
                "method": method,
                "dataset": dataset,
                "group": group,
                "residues": int(len(scores)),
                "known_residues": int(len(known_labels)),
                "positives": int(np.sum(known_labels == 1)) if len(known_labels) else 0,
                "mean_score": float(np.mean(scores)) if len(scores) else math.nan,
                "mean_uncertainty": float(np.mean(uncertainties)) if len(uncertainties) else math.nan,
                "uncertainty_q25": float(np.quantile(uncertainties, 0.25)) if len(uncertainties) else math.nan,
                "uncertainty_median": float(np.quantile(uncertainties, 0.50)) if len(uncertainties) else math.nan,
                "uncertainty_q75": float(np.quantile(uncertainties, 0.75)) if len(uncertainties) else math.nan,
                "known_error_rate": float(np.mean(errors)) if len(errors) else math.nan,
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def format_value(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.6f}"
    return value


def make_calibrator(method: str) -> Calibrator:
    if method == "raw":
        return RawCalibrator()
    if method == "temperature":
        return TemperatureCalibrator()
    if method == "platt":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"unsupported calibration method: {method}")


def plot_reliability(
    path: Path,
    experiment_id: str,
    dataset: str,
    bin_rows: list[dict[str, object]],
    methods: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="ideal")
    for method in methods:
        rows = [
            row
            for row in bin_rows
            if row["experiment_id"] == experiment_id and row["dataset"] == dataset and row["method"] == method and int(row["count"]) > 0
        ]
        xs = [float(row["mean_score"]) for row in rows]
        ys = [float(row["empirical_positive_rate"]) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=method)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed disorder fraction")
    ax.set_title(f"{experiment_id} / {dataset}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_uncertainty(
    path: Path,
    experiment_id: str,
    dataset: str,
    rows: list[dict[str, object]],
    methods: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    for method in methods:
        method_rows = [
            row
            for row in rows
            if row["experiment_id"] == experiment_id and row["dataset"] == dataset and row["method"] == method
        ]
        xs = [float(row["top_uncertain_fraction"]) for row in method_rows]
        ys = [float(row["error_enrichment"]) for row in method_rows]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=method)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Top uncertain fraction")
    ax.set_ylabel("Error enrichment")
    ax.set_title(f"{experiment_id} / {dataset}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--validation-labels", required=True, type=Path)
    parser.add_argument("--validation-predictions", required=True, type=Path)
    parser.add_argument("--test", action="append", default=[], help="name:labels_fasta:predictions_tsv")
    parser.add_argument("--methods", default="raw,temperature,platt,isotonic")
    parser.add_argument("--prediction-out-dir", default=Path("predictions/calibration"), type=Path)
    parser.add_argument("--results-out-dir", default=Path("results/calibration"), type=Path)
    parser.add_argument("--figures-out-dir", default=Path("figures/calibration"), type=Path)
    parser.add_argument("--bins", default=10, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    tests = [parse_test_spec(item) for item in args.test]
    validation_records = parse_labeled_fasta(args.validation_labels)
    validation_predictions = read_prediction_tsv(args.validation_predictions, "\t")
    validation_labels, validation_scores = collect_known_labels_scores(validation_records, validation_predictions)

    metric_rows: list[dict[str, object]] = []
    reliability_rows_out: list[dict[str, object]] = []
    uncertainty_rows_out: list[dict[str, object]] = []
    unknown_rows_out: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []

    fitted: dict[str, tuple[Calibrator, float]] = {}
    for method in methods:
        calibrator = make_calibrator(method)
        calibrator.fit(validation_labels, validation_scores)
        calibrated_validation = transform_predictions(validation_predictions, calibrator)
        validation_prediction_path = args.prediction_out_dir / f"{args.experiment_id}_{method}_DM1229_Validation.tsv"
        write_prediction_tsv(validation_prediction_path, calibrated_validation)
        validation_calibrated_labels, validation_calibrated_scores = collect_known_labels_scores(
            validation_records, calibrated_validation
        )
        _, threshold = fmax(validation_calibrated_labels, validation_calibrated_scores)
        fitted[method] = (calibrator, threshold)
        metric_row, bin_rows, labels_array, scores_array = metrics_row(
            experiment_id=args.experiment_id,
            method=method,
            dataset="DM1229_Validation",
            records=validation_records,
            predictions=calibrated_validation,
            threshold=threshold,
            bins=args.bins,
            prediction_file=validation_prediction_path,
        )
        metric_rows.append(metric_row)
        reliability_rows_out.extend(bin_rows)
        uncertainty_rows_out.extend(
            uncertainty_rows(args.experiment_id, method, "DM1229_Validation", labels_array, scores_array, threshold)
        )
        unknown_rows_out.extend(
            unknown_neighbor_rows(
                args.experiment_id, method, "DM1229_Validation", validation_records, calibrated_validation, threshold
            )
        )
        parameter_rows.append(
            {
                "experiment_id": args.experiment_id,
                "method": method,
                "validation_threshold": threshold,
                "parameters_json": json.dumps(calibrator.parameters(), sort_keys=True),
            }
        )

    for spec in tests:
        records = parse_labeled_fasta(spec.labels)
        raw_predictions = read_prediction_tsv(spec.predictions, "\t")
        for method, (calibrator, threshold) in fitted.items():
            calibrated_predictions = transform_predictions(raw_predictions, calibrator)
            prediction_path = args.prediction_out_dir / f"{args.experiment_id}_{method}_{spec.name}.tsv"
            write_prediction_tsv(prediction_path, calibrated_predictions)
            metric_row, bin_rows, labels_array, scores_array = metrics_row(
                experiment_id=args.experiment_id,
                method=method,
                dataset=spec.name,
                records=records,
                predictions=calibrated_predictions,
                threshold=threshold,
                bins=args.bins,
                prediction_file=prediction_path,
            )
            metric_rows.append(metric_row)
            reliability_rows_out.extend(bin_rows)
            uncertainty_rows_out.extend(uncertainty_rows(args.experiment_id, method, spec.name, labels_array, scores_array, threshold))
            unknown_rows_out.extend(unknown_neighbor_rows(args.experiment_id, method, spec.name, records, calibrated_predictions, threshold))

    metrics_fields = [
        "experiment_id",
        "method",
        "dataset",
        "proteins",
        "evaluated_residues",
        "positives",
        "negatives",
        "threshold",
        "tp",
        "fp",
        "tn",
        "fn",
        "sn",
        "sp",
        "bacc",
        "mcc",
        "auc",
        "aupr",
        "fmax",
        "fmax_threshold",
        "brier",
        "nll",
        "ece",
        "mce",
        "prediction_file",
    ]
    reliability_fields = [
        "experiment_id",
        "method",
        "dataset",
        "bin_index",
        "bin_low",
        "bin_high",
        "count",
        "positives",
        "mean_score",
        "empirical_positive_rate",
        "abs_gap",
    ]
    uncertainty_fields = [
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
    ]
    unknown_fields = [
        "experiment_id",
        "method",
        "dataset",
        "group",
        "residues",
        "known_residues",
        "positives",
        "mean_score",
        "mean_uncertainty",
        "uncertainty_q25",
        "uncertainty_median",
        "uncertainty_q75",
        "known_error_rate",
    ]
    parameter_fields = ["experiment_id", "method", "validation_threshold", "parameters_json"]

    write_rows(args.results_out_dir / f"{args.experiment_id}_calibration_metrics.tsv", metric_rows, metrics_fields)
    write_rows(args.results_out_dir / f"{args.experiment_id}_reliability_bins.tsv", reliability_rows_out, reliability_fields)
    write_rows(
        args.results_out_dir / f"{args.experiment_id}_uncertainty_error_enrichment.tsv",
        uncertainty_rows_out,
        uncertainty_fields,
    )
    write_rows(
        args.results_out_dir / f"{args.experiment_id}_unknown_neighbor_uncertainty.tsv",
        unknown_rows_out,
        unknown_fields,
    )
    write_rows(args.results_out_dir / f"{args.experiment_id}_calibration_parameters.tsv", parameter_rows, parameter_fields)

    for dataset in ["DM1229_Validation"] + [spec.name for spec in tests]:
        plot_reliability(
            args.figures_out_dir / f"reliability_{args.experiment_id}_{dataset}.pdf",
            args.experiment_id,
            dataset,
            reliability_rows_out,
            methods,
        )
        plot_uncertainty(
            args.figures_out_dir / f"uncertainty_error_enrichment_{args.experiment_id}_{dataset}.pdf",
            args.experiment_id,
            dataset,
            uncertainty_rows_out,
            methods,
        )


if __name__ == "__main__":
    main()
