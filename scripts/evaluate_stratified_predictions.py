#!/usr/bin/env python3
"""Evaluate disorder predictions overall and by predefined hard-case strata."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from annotate_disorder_regions import (
    disorder_content_bin,
    is_terminal_segment,
    iter_disorder_segments,
    length_bin,
    terminal_cutoff,
)
from evaluate_disorder_predictions import build_id_lookup, parse_labeled_fasta, read_prediction_tsv, roc_auc


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


def residue_zone(position: int, sequence_length: int) -> str:
    cutoff = terminal_cutoff(sequence_length)
    if position <= cutoff:
        return "n_terminal"
    if position >= sequence_length - cutoff + 1:
        return "c_terminal"
    return "middle"


def segment_maps(labels: list[int], sequence_length: int) -> tuple[dict[int, str], dict[int, str]]:
    length_type_by_pos: dict[int, str] = {}
    location_by_pos: dict[int, str] = {}
    for start, end, seg_len in iter_disorder_segments(labels):
        length_type = "SDR" if seg_len < 30 else "LDR"
        location = "terminal" if is_terminal_segment(start, end, sequence_length) else "internal"
        for position in range(start, end + 1):
            length_type_by_pos[position] = length_type
            location_by_pos[position] = location
    return length_type_by_pos, location_by_pos


def collect_residue_rows(
    records: list[dict[str, object]], predictions: dict[str, list[float]]
) -> list[dict[str, object]]:
    lookup = build_id_lookup(records)
    rows: list[dict[str, object]] = []
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

        sequence_length = len(sequence)
        known = sum(1 for label in labels if label in (0, 1))
        disordered = sum(1 for label in labels if label == 1)
        dc_bin = disorder_content_bin(disordered, known)
        len_bin = length_bin(sequence_length)
        length_type_by_pos, location_by_pos = segment_maps(labels, sequence_length)

        for position, (label, score) in enumerate(zip(labels, scores), start=1):
            if label == -1:
                continue
            rows.append(
                {
                    "protein_id": protein_id,
                    "position": position,
                    "label": int(label),
                    "score": float(score),
                    "protein_length_bin": len_bin,
                    "protein_disorder_content_bin": dc_bin,
                    "residue_zone": residue_zone(position, sequence_length),
                    "positive_region_length_type": length_type_by_pos.get(position, "non_disorder"),
                    "positive_region_location": location_by_pos.get(position, "non_disorder"),
                }
            )

    unknown_prediction_ids = sorted(set(predictions) - set(lookup))
    if unknown_prediction_ids:
        examples = ", ".join(unknown_prediction_ids[:5])
        raise ValueError(f"prediction file contains ids not present in labels: {examples}")
    return rows


def metrics_for_rows(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    proteins = len({str(row["protein_id"]) for row in rows})
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        predicted_positive = score >= threshold
        if label == 1 and predicted_positive:
            tp += 1
        elif label == 1:
            fn += 1
        elif predicted_positive:
            fp += 1
        else:
            tn += 1

    positives = tp + fn
    negatives = tn + fp
    sn = tp / positives if positives else math.nan
    sp = tn / negatives if negatives else math.nan
    bacc = (sn + sp) / 2.0 if math.isfinite(sn) and math.isfinite(sp) else math.nan
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else math.nan
    aupr = average_precision(labels, scores)
    fmax_value, fmax_threshold = fmax(labels, scores)
    auc = roc_auc(labels, scores)
    return {
        "proteins": proteins,
        "residues": len(rows),
        "positives": positives,
        "negatives": negatives,
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sn": sn,
        "sp": sp,
        "bacc": bacc,
        "mcc": mcc,
        "auc": auc,
        "aupr": aupr,
        "fmax": fmax_value,
        "fmax_threshold": fmax_threshold,
    }


def format_value(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.6f}"
    return value


def subset_by(rows: list[dict[str, object]], key: str) -> dict[str, list[dict[str, object]]]:
    subsets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        subsets[str(row[key])].append(row)
    return dict(sorted(subsets.items()))


def positive_type_subset(rows: list[dict[str, object]], key: str, value: str) -> list[dict[str, object]]:
    return [row for row in rows if int(row["label"]) == 0 or str(row[key]) == value]


def stratified_rows(dataset: str, rows: list[dict[str, object]], threshold: float) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []

    def add(group: str, stratum: str, subset: list[dict[str, object]]) -> None:
        if not subset:
            return
        metrics = metrics_for_rows(subset, threshold)
        output.append({"dataset": dataset, "stratum_group": group, "stratum": stratum, **metrics})

    add("overall", "all_known", rows)
    for group_key in ("protein_length_bin", "protein_disorder_content_bin", "residue_zone"):
        for stratum, subset in subset_by(rows, group_key).items():
            add(group_key, stratum, subset)
    for stratum in ("SDR", "LDR"):
        add("positive_region_length_type", stratum, positive_type_subset(rows, "positive_region_length_type", stratum))
    for stratum in ("terminal", "internal"):
        add("positive_region_location", stratum, positive_type_subset(rows, "positive_region_location", stratum))
    return output


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "stratum_group",
        "stratum",
        "proteins",
        "residues",
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
        for row in rows:
            writer.writerow({key: format_value(row[key]) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path, help="Labeled FASTA file.")
    parser.add_argument("--predictions", required=True, type=Path, help="Prediction TSV file.")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold for binary metrics.")
    parser.add_argument("--delimiter", default="\t", help="Prediction TSV delimiter. Default: tab.")
    parser.add_argument("--out", required=True, type=Path, help="Output TSV path.")
    args = parser.parse_args()
    if args.delimiter == "\\t":
        args.delimiter = "\t"
    return args


def main() -> None:
    args = parse_args()
    records = parse_labeled_fasta(args.labels)
    predictions = read_prediction_tsv(args.predictions, args.delimiter)
    residue_rows = collect_residue_rows(records, predictions)
    rows = stratified_rows(args.dataset, residue_rows, args.threshold)
    write_rows(args.out, rows)


if __name__ == "__main__":
    main()
