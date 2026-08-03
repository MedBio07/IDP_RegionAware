#!/usr/bin/env python3
"""Analyze residue, segment, and protein-level errors for disorder predictors."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from annotate_disorder_regions import (  # noqa: E402
    disorder_content_bin,
    is_terminal_segment,
    iter_disorder_segments,
    length_bin,
    terminal_cutoff,
)
from evaluate_disorder_predictions import parse_labeled_fasta, read_prediction_tsv, roc_auc  # noqa: E402


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
    index = 0
    while index < len(order):
        threshold = scores[order[index]]
        next_index = index
        while next_index < len(order) and scores[order[next_index]] == threshold:
            if labels[order[next_index]] == 1:
                tp += 1
            else:
                fp += 1
            next_index += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        index = next_index
    return best_f1, best_threshold


def threshold_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        positive = score >= threshold
        if label == 1 and positive:
            tp += 1
        elif label == 1:
            fn += 1
        elif positive:
            fp += 1
        else:
            tn += 1
    sn = tp / (tp + fn) if tp + fn else math.nan
    sp = tn / (tn + fp) if tn + fp else math.nan
    bacc = (sn + sp) / 2.0 if math.isfinite(sn) and math.isfinite(sp) else math.nan
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else math.nan
    precision = tp / (tp + fp) if tp + fp else math.nan
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sn": sn,
        "sp": sp,
        "bacc": bacc,
        "mcc": mcc,
        "precision": precision,
    }


def residue_zone(position: int, sequence_length: int) -> str:
    cutoff = terminal_cutoff(sequence_length)
    if position <= cutoff:
        return "n_terminal"
    if position >= sequence_length - cutoff + 1:
        return "c_terminal"
    return "middle"


def segment_length_bin(segment_length: int) -> str:
    if segment_length <= 4:
        return "01-04"
    if segment_length <= 9:
        return "05-09"
    if segment_length <= 29:
        return "10-29"
    if segment_length <= 99:
        return "30-99"
    return "100+"


def score_bin(score: float) -> str:
    lower = min(9, max(0, int(score * 10.0)))
    return f"{lower / 10:.1f}-{(lower + 1) / 10:.1f}"


def collect_residue_rows(records: list[dict[str, object]], scores_by_id: dict[str, list[float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        protein_id = str(record["id"])
        scores = scores_by_id.get(protein_id, scores_by_id.get(protein_id.split()[0]))
        if scores is None:
            raise ValueError(f"missing predictions for {protein_id}")
        sequence = str(record["sequence"])
        labels = record["labels"]
        if len(scores) != len(sequence):
            raise ValueError(f"{protein_id}: prediction length mismatch {len(scores)} != {len(sequence)}")
        known = sum(1 for label in labels if label in (0, 1))
        disordered = sum(1 for label in labels if label == 1)
        dc_bin = disorder_content_bin(disordered, known)
        len_bin = length_bin(len(sequence))
        segment_meta_by_position: dict[int, tuple[str, str, str]] = {}
        for start, end, segment_length in iter_disorder_segments(labels):
            location = "terminal" if is_terminal_segment(start, end, len(sequence)) else "internal"
            region_type = "SDR" if segment_length < 30 else "LDR"
            length_group = segment_length_bin(segment_length)
            for position in range(start, end + 1):
                segment_meta_by_position[position] = (region_type, location, length_group)
        for position, (label, score) in enumerate(zip(labels, scores), start=1):
            if label == -1:
                continue
            region_type, location, segment_bin = segment_meta_by_position.get(
                position, ("non_disorder", "non_disorder", "non_disorder")
            )
            rows.append(
                {
                    "protein_id": protein_id,
                    "position": position,
                    "label": int(label),
                    "score": float(score),
                    "protein_length_bin": len_bin,
                    "protein_disorder_content_bin": dc_bin,
                    "residue_zone": residue_zone(position, len(sequence)),
                    "positive_region_length_type": region_type,
                    "positive_region_location": location,
                    "positive_segment_length_bin": segment_bin,
                    "score_bin": score_bin(float(score)),
                }
            )
    return rows


def metrics_for_rows(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    fmax_value, fmax_threshold = fmax(labels, scores)
    threshold_row = threshold_metrics(labels, scores, threshold)
    return {
        "proteins": len({str(row["protein_id"]) for row in rows}),
        "residues": len(rows),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "threshold": threshold,
        **threshold_row,
        "auc": roc_auc(labels, scores),
        "aupr": average_precision(labels, scores),
        "fmax": fmax_value,
        "fmax_threshold": fmax_threshold,
    }


def grouped_residue_metrics(
    dataset: str,
    model_id: str,
    rows: list[dict[str, object]],
    threshold: float,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []

    def add(group: str, stratum: str, subset: list[dict[str, object]]) -> None:
        if not subset:
            return
        output.append({"dataset": dataset, "model_id": model_id, "stratum_group": group, "stratum": stratum, **metrics_for_rows(subset, threshold)})

    add("overall", "all_known", rows)
    for key in (
        "protein_length_bin",
        "protein_disorder_content_bin",
        "residue_zone",
        "positive_segment_length_bin",
        "score_bin",
    ):
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[key])].append(row)
        for stratum in sorted(grouped):
            add(key, stratum, grouped[stratum])
    for key, values in (
        ("positive_region_length_type", ("SDR", "LDR")),
        ("positive_region_location", ("terminal", "internal")),
    ):
        for value in values:
            subset = [row for row in rows if int(row["label"]) == 0 or str(row[key]) == value]
            add(key, value, subset)
    return output


def segment_rows(
    dataset: str,
    model_id: str,
    records: list[dict[str, object]],
    scores_by_id: dict[str, list[float]],
    threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        protein_id = str(record["id"])
        scores = scores_by_id.get(protein_id, scores_by_id.get(protein_id.split()[0]))
        if scores is None:
            raise ValueError(f"missing predictions for {protein_id}")
        labels = record["labels"]
        sequence_length = len(str(record["sequence"]))
        for segment_index, (start, end, segment_length) in enumerate(iter_disorder_segments(labels), start=1):
            segment_scores = [float(score) for score in scores[start - 1 : end]]
            called = [score >= threshold for score in segment_scores]
            recalled_residues = sum(called)
            rows.append(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "protein_id": protein_id,
                    "segment_index": segment_index,
                    "start": start,
                    "end": end,
                    "segment_length": segment_length,
                    "segment_length_bin": segment_length_bin(segment_length),
                    "region_length_type": "SDR" if segment_length < 30 else "LDR",
                    "region_location": "terminal" if is_terminal_segment(start, end, sequence_length) else "internal",
                    "mean_score": sum(segment_scores) / len(segment_scores),
                    "max_score": max(segment_scores),
                    "min_score": min(segment_scores),
                    "residue_recall": recalled_residues / segment_length,
                    "detected_any": int(recalled_residues > 0),
                    "detected_half": int(recalled_residues >= math.ceil(segment_length / 2)),
                    "detected_all": int(recalled_residues == segment_length),
                }
            )
    return rows


def grouped_segment_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []

    def add(group: str, stratum: str, subset: list[dict[str, object]]) -> None:
        if not subset:
            return
        residues = sum(int(row["segment_length"]) for row in subset)
        output.append(
            {
                "dataset": subset[0]["dataset"],
                "model_id": subset[0]["model_id"],
                "stratum_group": group,
                "stratum": stratum,
                "segments": len(subset),
                "residues": residues,
                "mean_segment_length": residues / len(subset),
                "mean_score": sum(float(row["mean_score"]) for row in subset) / len(subset),
                "mean_residue_recall": sum(float(row["residue_recall"]) for row in subset) / len(subset),
                "detected_any_fraction": sum(int(row["detected_any"]) for row in subset) / len(subset),
                "detected_half_fraction": sum(int(row["detected_half"]) for row in subset) / len(subset),
                "detected_all_fraction": sum(int(row["detected_all"]) for row in subset) / len(subset),
            }
        )

    add("overall", "all_segments", rows)
    for key in ("segment_length_bin", "region_length_type", "region_location"):
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[key])].append(row)
        for stratum in sorted(grouped):
            add(key, stratum, grouped[stratum])
    return output


def protein_rows(
    dataset: str,
    model_id: str,
    records: list[dict[str, object]],
    scores_by_id: dict[str, list[float]],
    threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        protein_id = str(record["id"])
        scores = scores_by_id.get(protein_id, scores_by_id.get(protein_id.split()[0]))
        if scores is None:
            raise ValueError(f"missing predictions for {protein_id}")
        labels = [int(label) for label in record["labels"] if int(label) != -1]
        known_scores = [float(score) for label, score in zip(record["labels"], scores) if int(label) != -1]
        if not labels:
            continue
        threshold_row = threshold_metrics(labels, known_scores, threshold)
        positives = sum(labels)
        negatives = len(labels) - positives
        auc = roc_auc(labels, known_scores) if positives and negatives else math.nan
        aupr = average_precision(labels, known_scores) if positives else math.nan
        fmax_value, fmax_threshold = fmax(labels, known_scores) if positives else (math.nan, math.nan)
        rows.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "protein_id": protein_id,
                "length": len(str(record["sequence"])),
                "known_residues": len(labels),
                "positives": positives,
                "negatives": negatives,
                "disorder_content": positives / len(labels),
                "protein_length_bin": length_bin(len(str(record["sequence"]))),
                "protein_disorder_content_bin": disorder_content_bin(positives, len(labels)),
                **threshold_row,
                "auc": auc,
                "aupr": aupr,
                "fmax": fmax_value,
                "fmax_threshold": fmax_threshold,
                "mean_positive_score": (
                    sum(score for label, score in zip(labels, known_scores) if label == 1) / positives
                    if positives
                    else math.nan
                ),
                "mean_negative_score": (
                    sum(score for label, score in zip(labels, known_scores) if label == 0) / negatives
                    if negatives
                    else math.nan
                ),
                "error_residues": int(threshold_row["fp"]) + int(threshold_row["fn"]),
            }
        )
    return sorted(rows, key=lambda row: (int(row["fn"]) + int(row["fp"]), int(row["fn"])), reverse=True)


def parse_prediction_arg(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--prediction must use model_id=path")
    model_id, path_text = text.split("=", 1)
    if not model_id:
        raise argparse.ArgumentTypeError("empty model_id")
    return model_id, Path(path_text)


def parse_threshold_arg(text: str) -> tuple[str, float]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--threshold must use model_id=value")
    model_id, value = text.split("=", 1)
    return model_id, float(value)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prediction", action="append", required=True, type=parse_prediction_arg)
    parser.add_argument("--threshold", action="append", default=[], type=parse_threshold_arg)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--delimiter", default="\t")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = parse_labeled_fasta(args.labels)
    thresholds = dict(args.threshold)

    residue_metric_rows: list[dict[str, object]] = []
    segment_detail_rows: list[dict[str, object]] = []
    segment_metric_rows: list[dict[str, object]] = []
    protein_detail_rows: list[dict[str, object]] = []

    for model_id, prediction_path in args.prediction:
        predictions = read_prediction_tsv(prediction_path, args.delimiter)
        residue_rows = collect_residue_rows(records, predictions)
        labels = [int(row["label"]) for row in residue_rows]
        scores = [float(row["score"]) for row in residue_rows]
        default_threshold = fmax(labels, scores)[1]
        threshold = thresholds.get(model_id, default_threshold)

        residue_metric_rows.extend(grouped_residue_metrics(args.dataset, model_id, residue_rows, threshold))
        model_segment_rows = segment_rows(args.dataset, model_id, records, predictions, threshold)
        segment_detail_rows.extend(model_segment_rows)
        segment_metric_rows.extend(grouped_segment_metrics(model_segment_rows))
        protein_detail_rows.extend(protein_rows(args.dataset, model_id, records, predictions, threshold))

    residue_fields = [
        "dataset",
        "model_id",
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
        "precision",
        "auc",
        "aupr",
        "fmax",
        "fmax_threshold",
    ]
    segment_detail_fields = [
        "dataset",
        "model_id",
        "protein_id",
        "segment_index",
        "start",
        "end",
        "segment_length",
        "segment_length_bin",
        "region_length_type",
        "region_location",
        "mean_score",
        "max_score",
        "min_score",
        "residue_recall",
        "detected_any",
        "detected_half",
        "detected_all",
    ]
    segment_metric_fields = [
        "dataset",
        "model_id",
        "stratum_group",
        "stratum",
        "segments",
        "residues",
        "mean_segment_length",
        "mean_score",
        "mean_residue_recall",
        "detected_any_fraction",
        "detected_half_fraction",
        "detected_all_fraction",
    ]
    protein_fields = [
        "dataset",
        "model_id",
        "protein_id",
        "length",
        "known_residues",
        "positives",
        "negatives",
        "disorder_content",
        "protein_length_bin",
        "protein_disorder_content_bin",
        "tp",
        "fp",
        "tn",
        "fn",
        "sn",
        "sp",
        "bacc",
        "mcc",
        "precision",
        "auc",
        "aupr",
        "fmax",
        "fmax_threshold",
        "mean_positive_score",
        "mean_negative_score",
        "error_residues",
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_dir / f"P4_5_{args.dataset}_RESIDUE_ERROR_STRATA.tsv", residue_metric_rows, residue_fields)
    write_tsv(args.out_dir / f"P4_5_{args.dataset}_SEGMENT_ERROR_DETAIL.tsv", segment_detail_rows, segment_detail_fields)
    write_tsv(args.out_dir / f"P4_5_{args.dataset}_SEGMENT_ERROR_SUMMARY.tsv", segment_metric_rows, segment_metric_fields)
    write_tsv(args.out_dir / f"P4_5_{args.dataset}_PROTEIN_ERROR_RANKING.tsv", protein_detail_rows, protein_fields)

    overall = [
        row
        for row in residue_metric_rows
        if row["stratum_group"] == "overall" and row["stratum"] == "all_known"
    ]
    for row in overall:
        print(
            f"{row['dataset']}\t{row['model_id']}\t"
            f"AUC={format_value(row['auc'])}\tAUPR={format_value(row['aupr'])}\t"
            f"MCC={format_value(row['mcc'])}\tFmax={format_value(row['fmax'])}"
        )


if __name__ == "__main__":
    main()
