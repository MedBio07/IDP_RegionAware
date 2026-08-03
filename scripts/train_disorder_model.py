#!/usr/bin/env python3
"""Train a shallow residue-level disorder baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from evaluate_disorder_predictions import parse_labeled_fasta, roc_auc
from models.features import feature_matrix, parse_feature_list


def collect_matrix(records: list[dict[str, object]], feature_names: list[str], embedding_dir: Path | None) -> tuple[np.ndarray, np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for record in records:
        sequence = str(record["sequence"])
        labels = np.asarray(record["labels"], dtype=np.int8)
        mask = labels != -1
        if not np.any(mask):
            continue
        matrix = feature_matrix(sequence, feature_names, embedding_dir)
        x_parts.append(matrix[mask])
        y_parts.append(labels[mask].astype(np.int8, copy=False))
    if not x_parts:
        raise ValueError("no known residues found")
    return np.vstack(x_parts), np.concatenate(y_parts)


def predict_scores(estimator: dict[str, object], x: np.ndarray) -> np.ndarray:
    scaler = estimator["scaler"]
    model = estimator["model"]
    x_scaled = scaler.transform(x)
    probabilities = model.predict_proba(x_scaled)
    classes = list(model.classes_)
    return probabilities[:, classes.index(1)]


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    tp = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / positives


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
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


def select_fmax_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return math.nan, math.nan
    order = np.argsort(-scores, kind="mergesort")
    tp = 0
    fp = 0
    best_f1 = -1.0
    best_threshold = 0.5
    i = 0
    while i < len(order):
        threshold = float(scores[order[i]])
        j = i
        while j < len(order) and float(scores[order[j]]) == threshold:
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


def write_prediction_rows(path: Path, records: list[dict[str, object]], estimator: dict[str, object], feature_names: list[str], embedding_dir: Path | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "scores"])
        for record in records:
            sequence = str(record["sequence"])
            matrix = feature_matrix(sequence, feature_names, embedding_dir)
            scores = predict_scores(estimator, matrix)
            writer.writerow([record["id"], "[" + ", ".join(f"{score:.8g}" for score in scores) + "]"])


def format_metric(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.6f}"
    return value


def write_metrics(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "train_set",
        "validation_set",
        "features",
        "model_type",
        "train_residues",
        "train_positives",
        "train_negatives",
        "validation_residues",
        "validation_positives",
        "validation_negatives",
        "threshold",
        "validation_auc",
        "validation_aupr",
        "validation_fmax",
        "validation_sn",
        "validation_sp",
        "validation_bacc",
        "validation_mcc",
        "fit_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow({key: format_metric(row.get(key, "")) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--features", default="esm,position", help="Comma-separated features: esm,position,onehot.")
    parser.add_argument("--embedding-dir", type=Path)
    parser.add_argument("--model-out", required=True, type=Path)
    parser.add_argument("--metrics-out", required=True, type=Path)
    parser.add_argument("--validation-predictions-out", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_names = parse_feature_list(args.features)
    if "esm" in feature_names and args.embedding_dir is None:
        raise ValueError("--embedding-dir is required when using esm features")
    train_records = parse_labeled_fasta(args.train)
    val_records = parse_labeled_fasta(args.validation)
    x_train, y_train = collect_matrix(train_records, feature_names, args.embedding_dir)
    x_val, y_val = collect_matrix(val_records, feature_names, args.embedding_dir)

    started = time.perf_counter()
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    positive = int(np.sum(y_train == 1))
    negative = int(np.sum(y_train == 0))
    total = positive + negative
    weights = np.where(
        y_train == 1,
        total / (2.0 * positive) if positive else 1.0,
        total / (2.0 * negative) if negative else 1.0,
    ).astype(np.float32)
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=args.alpha,
        max_iter=args.max_iter,
        tol=1e-3,
        average=True,
        random_state=args.seed,
    )
    model.fit(x_train_scaled, y_train, sample_weight=weights)
    fit_seconds = time.perf_counter() - started
    estimator = {
        "scaler": scaler,
        "model": model,
        "feature_names": feature_names,
        "embedding_dir": str(args.embedding_dir) if args.embedding_dir else None,
    }

    val_scores = predict_scores(estimator, x_val)
    validation_fmax, threshold = select_fmax_threshold(y_val, val_scores)
    threshold_row = threshold_metrics(y_val, val_scores, threshold)
    auc = roc_auc(y_val.astype(int).tolist(), val_scores.astype(float).tolist())
    aupr = average_precision(y_val, val_scores)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment_id": args.experiment_id,
        "train": str(args.train),
        "validation": str(args.validation),
        "features": feature_names,
        "embedding_dir": str(args.embedding_dir) if args.embedding_dir else None,
        "model_type": "sklearn.SGDClassifier(log_loss, class-balanced sample weights)",
        "seed": args.seed,
        "threshold": threshold,
        "validation_auc": auc,
        "validation_aupr": aupr,
        "validation_fmax": validation_fmax,
        "fit_seconds": fit_seconds,
    }
    estimator["metadata"] = metadata
    with args.model_out.open("wb") as handle:
        pickle.dump(estimator, handle)

    metrics_row = {
        "experiment_id": args.experiment_id,
        "train_set": str(args.train),
        "validation_set": str(args.validation),
        "features": ",".join(feature_names),
        "model_type": metadata["model_type"],
        "train_residues": int(len(y_train)),
        "train_positives": positive,
        "train_negatives": negative,
        "validation_residues": int(len(y_val)),
        "validation_positives": int(np.sum(y_val == 1)),
        "validation_negatives": int(np.sum(y_val == 0)),
        "threshold": threshold,
        "validation_auc": auc,
        "validation_aupr": aupr,
        "validation_fmax": validation_fmax,
        "validation_sn": threshold_row["sn"],
        "validation_sp": threshold_row["sp"],
        "validation_bacc": threshold_row["bacc"],
        "validation_mcc": threshold_row["mcc"],
        "fit_seconds": fit_seconds,
    }
    write_metrics(args.metrics_out, metrics_row)
    if args.validation_predictions_out is not None:
        write_prediction_rows(args.validation_predictions_out, val_records, estimator, feature_names, args.embedding_dir)

    print(json.dumps({key: format_metric(value) for key, value in metrics_row.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
