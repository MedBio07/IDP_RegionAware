#!/usr/bin/env python3
"""Average residue-level disorder prediction TSV files."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_disorder_predictions import evaluate, parse_labeled_fasta, read_prediction_tsv, roc_auc
from train_sequence_disorder_model import average_precision, fmax


def average_predictions(paths: list[Path], delimiter: str) -> dict[str, list[float]]:
    if not paths:
        raise ValueError("at least one prediction file is required")
    prediction_sets = [read_prediction_tsv(path, delimiter) for path in paths]
    first = prediction_sets[0]
    protein_ids = list(first)
    if len(set(protein_ids)) != len(protein_ids):
        raise ValueError(f"duplicate IDs in first prediction file: {paths[0]}")
    expected_ids = set(protein_ids)
    for path, predictions in zip(paths[1:], prediction_sets[1:]):
        if set(predictions) != expected_ids:
            missing = sorted(expected_ids - set(predictions))
            extra = sorted(set(predictions) - expected_ids)
            detail = []
            if missing:
                detail.append(f"missing={missing[:5]}")
            if extra:
                detail.append(f"extra={extra[:5]}")
            raise ValueError(f"{path}: prediction IDs differ from first file ({'; '.join(detail)})")

    averaged: dict[str, list[float]] = {}
    for protein_id in protein_ids:
        length = len(first[protein_id])
        totals = [0.0] * length
        for path, predictions in zip(paths, prediction_sets):
            scores = predictions[protein_id]
            if len(scores) != length:
                raise ValueError(f"{path}: length mismatch for {protein_id}: {len(scores)} != {length}")
            for index, score in enumerate(scores):
                totals[index] += float(score)
        averaged[protein_id] = [value / len(prediction_sets) for value in totals]
    return averaged


def write_predictions(path: Path, predictions: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "scores"])
        for protein_id, scores in predictions.items():
            writer.writerow([protein_id, "[" + ", ".join(f"{score:.8g}" for score in scores) + "]"])


def labels_and_scores(records: list[dict[str, object]], predictions: dict[str, list[float]]) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    scores: list[float] = []
    for record in records:
        protein_id = str(record["id"])
        item_scores = predictions.get(protein_id)
        if item_scores is None:
            item_scores = predictions.get(protein_id.split()[0])
        if item_scores is None:
            raise ValueError(f"missing predictions for {protein_id}")
        sequence = str(record["sequence"])
        if len(item_scores) != len(sequence):
            raise ValueError(f"prediction length mismatch for {protein_id}: {len(item_scores)} != {len(sequence)}")
        for label, score in zip(record["labels"], item_scores):
            if label == -1:
                continue
            labels.append(int(label))
            scores.append(float(score))
    return labels, scores


def write_metrics(
    path: Path,
    dataset: str,
    records: list[dict[str, object]],
    predictions: dict[str, list[float]],
    threshold: float | None,
) -> None:
    labels, scores = labels_and_scores(records, predictions)
    fmax_value, fmax_threshold = fmax(labels, scores)
    selected_threshold = fmax_threshold if threshold is None else threshold
    metrics = evaluate(records, predictions, selected_threshold)
    row = {
        "dataset": dataset,
        **metrics,
        "aupr": average_precision(labels, scores),
        "fmax": fmax_value,
        "fmax_threshold": fmax_threshold,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    ]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow({key: format_value(row[key]) for key in fieldnames})


def format_value(value: object) -> object:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.6f}"
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--metrics-out", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--delimiter", default="\t")
    args = parser.parse_args()
    if args.delimiter == "\\t":
        args.delimiter = "\t"
    if args.metrics_out and not args.labels:
        raise ValueError("--labels is required with --metrics-out")
    if args.metrics_out and not args.dataset:
        raise ValueError("--dataset is required with --metrics-out")
    return args


def main() -> None:
    args = parse_args()
    predictions = average_predictions(args.inputs, args.delimiter)
    write_predictions(args.out, predictions)
    if args.metrics_out:
        assert args.labels is not None
        assert args.dataset is not None
        records = parse_labeled_fasta(args.labels)
        write_metrics(args.metrics_out, args.dataset, records, predictions, args.threshold)


if __name__ == "__main__":
    main()
