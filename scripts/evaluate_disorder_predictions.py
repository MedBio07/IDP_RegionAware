#!/usr/bin/env python3
"""Evaluate residue-level intrinsic disorder predictions.

Prediction TSV input supports either format:

1. Wide format with one row per protein:
   id<TAB>scores
   P12345<TAB>[0.1, 0.8, 0.3]

2. Long format with one row per residue:
   id<TAB>position<TAB>score
   P12345<TAB>1<TAB>0.1

Positions in long format are 1-based. Labels with value -1 are ignored.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ID_COLUMNS = ("id", "sequence_id", "protein_id", "name", "header")
SCORES_COLUMNS = ("scores", "predictions", "prediction_scores", "probabilities", "probs")
SCORE_COLUMNS = ("score", "prediction", "probability", "prob", "disorder_score")
POSITION_COLUMNS = ("position", "pos", "residue_index", "idx", "index")


def parse_labeled_fasta(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    header: str | None = None
    seq_parts: list[str] = []
    label_text: str | None = None

    def flush() -> None:
        nonlocal header, seq_parts, label_text
        if header is None:
            return
        if label_text is None:
            raise ValueError(f"{path}: missing label line for {header}")
        sequence = "".join(seq_parts)
        labels = ast.literal_eval(label_text)
        if not isinstance(labels, list):
            raise ValueError(f"{path}: labels are not a list for {header}")
        if any(label not in (-1, 0, 1) for label in labels):
            raise ValueError(f"{path}: labels must be -1, 0, or 1 for {header}")
        if len(sequence) != len(labels):
            raise ValueError(
                f"{path}: sequence/label length mismatch for {header}: "
                f"{len(sequence)} != {len(labels)}"
            )
        records.append({"id": header, "sequence": sequence, "labels": labels})
        header = None
        seq_parts = []
        label_text = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
            elif line.startswith("["):
                label_text = line
            else:
                seq_parts.append(line)
    flush()
    return records


def parse_score_list(text: str) -> list[float]:
    value = text.strip()
    if not value:
        raise ValueError("empty score list")
    if value.startswith("["):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"score list is not a list: {text[:80]}")
        scores = [float(item) for item in parsed]
    else:
        scores = [float(item) for item in value.replace(",", " ").split()]
    for score in scores:
        if not math.isfinite(score):
            raise ValueError(f"non-finite score: {score}")
    return scores


def first_present(fieldnames: Iterable[str], choices: Iterable[str]) -> str | None:
    names = set(fieldnames)
    for choice in choices:
        if choice in names:
            return choice
    return None


def read_prediction_tsv(path: Path, delimiter: str) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.readline()
        if not sample:
            raise ValueError(f"{path}: empty prediction file")
        handle.seek(0)

        first_cells = [cell.strip() for cell in sample.rstrip("\n").split(delimiter)]
        has_header = any(cell.lower() in ID_COLUMNS + SCORES_COLUMNS + SCORE_COLUMNS for cell in first_cells)

        if has_header:
            reader = csv.DictReader(handle, delimiter=delimiter)
            assert reader.fieldnames is not None
            fieldnames = [name.strip() for name in reader.fieldnames]
            reader.fieldnames = fieldnames

            id_col = first_present(fieldnames, ID_COLUMNS)
            position_col = first_present(fieldnames, POSITION_COLUMNS)
            scores_col = first_present(fieldnames, SCORES_COLUMNS)
            score_col = first_present(fieldnames, SCORE_COLUMNS)
            if id_col is None:
                raise ValueError(f"{path}: prediction header needs an id column")

            if scores_col is not None and position_col is None:
                predictions: dict[str, list[float]] = {}
                for row_number, row in enumerate(reader, start=2):
                    protein_id = (row.get(id_col) or "").strip()
                    if not protein_id:
                        raise ValueError(f"{path}:{row_number}: missing id")
                    if protein_id in predictions:
                        raise ValueError(f"{path}:{row_number}: duplicate id {protein_id}")
                    predictions[protein_id] = parse_score_list(row.get(scores_col) or "")
                return predictions

            if position_col is not None and score_col is not None:
                by_id: dict[str, dict[int, float]] = defaultdict(dict)
                for row_number, row in enumerate(reader, start=2):
                    protein_id = (row.get(id_col) or "").strip()
                    if not protein_id:
                        raise ValueError(f"{path}:{row_number}: missing id")
                    position = int(row.get(position_col) or "0")
                    if position < 1:
                        raise ValueError(f"{path}:{row_number}: positions must be 1-based")
                    if position in by_id[protein_id]:
                        raise ValueError(f"{path}:{row_number}: duplicate position {position} for {protein_id}")
                    score = float(row.get(score_col) or "nan")
                    if not math.isfinite(score):
                        raise ValueError(f"{path}:{row_number}: non-finite score")
                    by_id[protein_id][position] = score
                return positions_to_lists(path, by_id)

            raise ValueError(
                f"{path}: header must describe either id+scores or id+position+score columns"
            )

        reader = csv.reader(handle, delimiter=delimiter)
        rows = [row for row in reader if row]
        if not rows:
            raise ValueError(f"{path}: empty prediction file")
        width = len(rows[0])
        if width == 2:
            predictions = {}
            for row_number, row in enumerate(rows, start=1):
                if len(row) != 2:
                    raise ValueError(f"{path}:{row_number}: expected 2 columns")
                protein_id = row[0].strip()
                if protein_id in predictions:
                    raise ValueError(f"{path}:{row_number}: duplicate id {protein_id}")
                predictions[protein_id] = parse_score_list(row[1])
            return predictions
        if width == 3:
            by_id = defaultdict(dict)
            for row_number, row in enumerate(rows, start=1):
                if len(row) != 3:
                    raise ValueError(f"{path}:{row_number}: expected 3 columns")
                protein_id = row[0].strip()
                position = int(row[1])
                if position < 1:
                    raise ValueError(f"{path}:{row_number}: positions must be 1-based")
                if position in by_id[protein_id]:
                    raise ValueError(f"{path}:{row_number}: duplicate position {position} for {protein_id}")
                score = float(row[2])
                if not math.isfinite(score):
                    raise ValueError(f"{path}:{row_number}: non-finite score")
                by_id[protein_id][position] = score
            return positions_to_lists(path, by_id)
        raise ValueError(f"{path}: expected 2 or 3 columns without a header")


def positions_to_lists(path: Path, by_id: dict[str, dict[int, float]]) -> dict[str, list[float]]:
    predictions: dict[str, list[float]] = {}
    for protein_id, scores_by_position in by_id.items():
        positions = sorted(scores_by_position)
        expected = list(range(1, len(positions) + 1))
        if positions != expected:
            missing = sorted(set(expected) - set(positions))
            extra = sorted(set(positions) - set(expected))
            detail = f"missing={missing[:5]}" if missing else f"extra={extra[:5]}"
            raise ValueError(f"{path}: positions for {protein_id} are not contiguous from 1 ({detail})")
        predictions[protein_id] = [scores_by_position[position] for position in positions]
    return predictions


def build_id_lookup(records: list[dict[str, object]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    first_tokens: dict[str, str | None] = {}
    for record in records:
        protein_id = str(record["id"])
        if protein_id in lookup:
            raise ValueError(f"duplicate FASTA id: {protein_id}")
        lookup[protein_id] = protein_id
        token = protein_id.split()[0]
        if token in first_tokens:
            first_tokens[token] = None
        else:
            first_tokens[token] = protein_id
    for token, protein_id in first_tokens.items():
        if protein_id is not None:
            lookup.setdefault(token, protein_id)
    return lookup


def roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan

    order = sorted(range(len(scores)), key=lambda index: scores[index])
    rank_sum_positive = 0.0
    rank = 1
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        average_rank = (rank + rank + (j - i) - 1) / 2.0
        for k in range(i, j):
            if labels[order[k]] == 1:
                rank_sum_positive += average_rank
        rank += j - i
        i = j
    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def evaluate(
    records: list[dict[str, object]],
    predictions: dict[str, list[float]],
    threshold: float,
) -> dict[str, int | float]:
    lookup = build_id_lookup(records)
    labels_eval: list[int] = []
    scores_eval: list[float] = []
    matched = 0

    for record in records:
        protein_id = str(record["id"])
        scores = predictions.get(protein_id)
        if scores is None:
            token = protein_id.split()[0]
            scores = predictions.get(token)
        if scores is None:
            raise ValueError(f"missing predictions for {protein_id}")

        sequence = str(record["sequence"])
        labels = record["labels"]
        assert isinstance(labels, list)
        if len(scores) != len(sequence):
            raise ValueError(
                f"prediction length mismatch for {protein_id}: "
                f"{len(scores)} != {len(sequence)}"
            )
        matched += 1
        for label, score in zip(labels, scores):
            if label == -1:
                continue
            labels_eval.append(int(label))
            scores_eval.append(score)

    unknown_prediction_ids = sorted(set(predictions) - set(lookup))
    if unknown_prediction_ids:
        examples = ", ".join(unknown_prediction_ids[:5])
        raise ValueError(f"prediction file contains ids not present in labels: {examples}")

    tp = fp = tn = fn = 0
    for label, score in zip(labels_eval, scores_eval):
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
    auc = roc_auc(labels_eval, scores_eval)

    return {
        "proteins": matched,
        "evaluated_residues": len(labels_eval),
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
    }


def format_value(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.6f}"


def write_result(row: dict[str, str | int | float], out_path: Path | None) -> None:
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
    ]
    output = sys.stdout if out_path is None else out_path.open("w", encoding="utf-8", newline="\n")
    try:
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow({key: format_value(row[key]) if key != "dataset" else row[key] for key in fieldnames})
    finally:
        if out_path is not None:
            output.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path, help="Labeled FASTA file.")
    parser.add_argument("--predictions", required=True, type=Path, help="Prediction TSV file.")
    parser.add_argument("--dataset", required=True, help="Dataset name to write in the output row.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold for Sn/Sp/BACC/MCC.")
    parser.add_argument("--delimiter", default="\t", help="Prediction TSV delimiter. Default: tab.")
    parser.add_argument("--out", type=Path, help="Optional output TSV path. Defaults to stdout.")
    args = parser.parse_args()
    if args.delimiter == "\\t":
        args.delimiter = "\t"
    return args


def main() -> None:
    args = parse_args()
    records = parse_labeled_fasta(args.labels)
    predictions = read_prediction_tsv(args.predictions, args.delimiter)
    metrics = evaluate(records, predictions, args.threshold)
    row: dict[str, str | int | float] = {"dataset": args.dataset, **metrics}
    write_result(row, args.out)


if __name__ == "__main__":
    main()
