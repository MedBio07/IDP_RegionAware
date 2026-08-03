#!/usr/bin/env python3
"""Predict residue-level disorder probabilities with a trained shallow baseline."""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_disorder_predictions import parse_labeled_fasta
from models.features import feature_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.model.open("rb") as handle:
        estimator = pickle.load(handle)
    feature_names = estimator["feature_names"]
    embedding_dir = estimator.get("embedding_dir")
    embedding_path = Path(embedding_dir) if embedding_dir else None
    scaler = estimator["scaler"]
    model = estimator["model"]
    classes = list(model.classes_)
    positive_index = classes.index(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = parse_labeled_fasta(args.fasta)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "scores"])
        for record in records:
            matrix = feature_matrix(str(record["sequence"]), feature_names, embedding_path)
            scores = model.predict_proba(scaler.transform(matrix))[:, positive_index]
            writer.writerow([record["id"], "[" + ", ".join(f"{score:.8g}" for score in scores) + "]"])


if __name__ == "__main__":
    main()
