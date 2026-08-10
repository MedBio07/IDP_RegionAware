#!/usr/bin/env python3
"""Compare paired protein-level AUC for one or more prediction-file pairs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compile_p5_evidence_bundle import (  # noqa: E402
    DATASETS,
    collect_paired_protein_scores,
    fmt,
    paired_bootstrap_auc,
)


def parse_comparison(value: str) -> tuple[str, Path, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3 or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "comparison must use DATASET:REFERENCE_TSV:CANDIDATE_TSV"
        )

    dataset, reference_text, candidate_text = parts
    if dataset not in DATASETS:
        valid = ", ".join(sorted(DATASETS))
        raise argparse.ArgumentTypeError(
            f"unknown dataset {dataset!r}; expected one of: {valid}"
        )

    reference = Path(reference_text).expanduser()
    candidate = Path(candidate_text).expanduser()
    for label, path in (("reference", reference), ("candidate", candidate)):
        if not path.is_file():
            raise argparse.ArgumentTypeError(
                f"{label} prediction file does not exist or is not a file: {path}"
            )
    return dataset, reference, candidate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        action="append",
        required=True,
        type=parse_comparison,
        metavar="DATASET:REFERENCE_TSV:CANDIDATE_TSV",
        help="paired prediction files; repeat once per dataset",
    )
    parser.add_argument("--reference-variant", required=True)
    parser.add_argument("--candidate-variant", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.bootstrap < 0:
        parser.error("--bootstrap must be non-negative")
    if args.permutations < 0:
        parser.error("--permutations must be non-negative")

    datasets = [dataset for dataset, _, _ in args.comparison]
    duplicates = sorted({dataset for dataset in datasets if datasets.count(dataset) > 1})
    if duplicates:
        parser.error(f"duplicate dataset comparison(s): {', '.join(duplicates)}")
    return args


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("at least one comparison is required")

    fieldnames = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)

    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fieldnames})


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    output_rows: list[dict[str, object]] = []

    for dataset, reference_path, candidate_path in args.comparison:
        paired = collect_paired_protein_scores(dataset, reference_path, candidate_path)
        row = paired_bootstrap_auc(
            dataset,
            paired,
            args.bootstrap,
            args.permutations,
            rng,
        )
        row["reference_variant"] = args.reference_variant
        row["candidate_variant"] = args.candidate_variant
        output_rows.append(row)

    write_results(args.out, output_rows)


if __name__ == "__main__":
    main()
