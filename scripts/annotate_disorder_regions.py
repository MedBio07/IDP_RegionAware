#!/usr/bin/env python3
"""Annotate intrinsic-disorder regions in labeled three-line FASTA files.

The input FASTA format is:

1. header
2. amino-acid sequence
3. Python-list labels, where 1 is disorder, 0 is order, and -1 is ignored

Outputs:
- protein-level annotation TSV
- disorder segment TSV
- dataset summary TSV
- optional test-set-only summary TSV
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from evaluate_disorder_predictions import parse_labeled_fasta


DEFAULT_DATASETS = (
    ("DM3000_Train", "train", Path("data/DM3000_Train.fasta")),
    ("DM1229_Validation", "validation", Path("data/DM1229_Validation.fasta")),
    ("SL329", "test", Path("data/SL329_test.fasta")),
    ("MXD494", "test", Path("data/MXD494_test.fasta")),
    ("DISORDER723", "test", Path("data/DISORDER723_test.fasta")),
    (
        "DM3000_Train_nr25_vs_SL329",
        "nr25_train",
        Path("data/nr25_by_test/DM3000_Train_nr25_vs_SL329.fasta"),
    ),
    (
        "DM3000_Train_nr25_vs_MXD494",
        "nr25_train",
        Path("data/nr25_by_test/DM3000_Train_nr25_vs_MXD494.fasta"),
    ),
    (
        "DM3000_Train_nr25_vs_DISORDER723",
        "nr25_train",
        Path("data/nr25_by_test/DM3000_Train_nr25_vs_DISORDER723.fasta"),
    ),
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    split_type: str
    path: Path


def length_bin(length: int) -> str:
    if length <= 200:
        return "<=200"
    if length <= 500:
        return "201-500"
    if length <= 1000:
        return "501-1000"
    return ">1000"


def disorder_content_bin(disordered: int, known: int) -> str:
    if known == 0:
        return "no_known"
    rate = disordered / known
    if rate <= 0.05:
        return "0-5"
    if rate <= 0.20:
        return "5-20"
    if rate < 0.80:
        return "20-80"
    return "80-100"


def terminal_cutoff(length: int) -> int:
    return max(10, int(math.ceil(length * 0.10)))


def is_terminal_segment(start: int, end: int, sequence_length: int) -> bool:
    cutoff = terminal_cutoff(sequence_length)
    return start <= cutoff or end >= sequence_length - cutoff + 1


def iter_disorder_segments(labels: list[int]) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    start: int | None = None
    for index, label in enumerate(labels, start=1):
        if label == 1 and start is None:
            start = index
        elif label != 1 and start is not None:
            end = index - 1
            segments.append((start, end, end - start + 1))
            start = None
    if start is not None:
        end = len(labels)
        segments.append((start, end, end - start + 1))
    return segments


def parse_dataset_arg(value: str) -> DatasetSpec:
    if "=" in value:
        name, path_text = value.split("=", 1)
    elif ":" in value:
        name, path_text = value.split(":", 1)
    else:
        raise argparse.ArgumentTypeError("Dataset must be NAME=PATH or NAME:PATH")
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Dataset name is empty")
    return DatasetSpec(name=name, split_type=infer_split_type(name), path=Path(path_text.strip()))


def infer_split_type(name: str) -> str:
    lower = name.lower()
    if "nr25" in lower:
        return "nr25_train"
    if "validation" in lower or "valid" in lower:
        return "validation"
    if "test" in lower or lower in {"sl329", "mxd494", "disorder723"}:
        return "test"
    return "train"


def summarize_dataset(spec: DatasetSpec) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    records = parse_labeled_fasta(spec.path)
    protein_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    counters: Counter[str] = Counter()

    for record in records:
        protein_id = str(record["id"])
        sequence = str(record["sequence"])
        labels = record["labels"]
        assert isinstance(labels, list)

        length = len(sequence)
        known = sum(1 for label in labels if label in (0, 1))
        disordered = sum(1 for label in labels if label == 1)
        ordered = sum(1 for label in labels if label == 0)
        unknown = sum(1 for label in labels if label == -1)
        dc_bin = disorder_content_bin(disordered, known)
        len_bin = length_bin(length)
        segments = iter_disorder_segments(labels)

        per_protein = Counter()
        for segment_index, (start, end, seg_len) in enumerate(segments, start=1):
            region_length_type = "SDR" if seg_len < 30 else "LDR"
            terminal_status = "terminal" if is_terminal_segment(start, end, length) else "internal"
            per_protein["disorder_segments"] += 1
            per_protein[f"{region_length_type.lower()}_segments"] += 1
            per_protein[f"{terminal_status}_segments"] += 1
            per_protein[f"{region_length_type.lower()}_residues"] += seg_len
            per_protein[f"{terminal_status}_idr_residues"] += seg_len
            segment_rows.append(
                {
                    "dataset": spec.name,
                    "split_type": spec.split_type,
                    "protein_id": protein_id,
                    "segment_index": segment_index,
                    "start": start,
                    "end": end,
                    "length": seg_len,
                    "region_length_type": region_length_type,
                    "terminal_status": terminal_status,
                    "sequence_length": length,
                    "known_residues": known,
                    "protein_disorder_content": format_float(disordered / known if known else math.nan),
                    "disorder_content_bin": dc_bin,
                    "protein_length_bin": len_bin,
                }
            )

        protein_rows.append(
            {
                "dataset": spec.name,
                "split_type": spec.split_type,
                "protein_id": protein_id,
                "sequence_length": length,
                "known_residues": known,
                "disordered": disordered,
                "ordered": ordered,
                "unknown": unknown,
                "disorder_rate_known": format_float(disordered / known if known else math.nan),
                "disorder_content_bin": dc_bin,
                "protein_length_bin": len_bin,
                "disorder_segments": per_protein["disorder_segments"],
                "sdr_segments": per_protein["sdr_segments"],
                "ldr_segments": per_protein["ldr_segments"],
                "terminal_segments": per_protein["terminal_segments"],
                "internal_segments": per_protein["internal_segments"],
                "sdr_residues": per_protein["sdr_residues"],
                "ldr_residues": per_protein["ldr_residues"],
                "terminal_idr_residues": per_protein["terminal_idr_residues"],
                "internal_idr_residues": per_protein["internal_idr_residues"],
                "has_disorder": int(per_protein["disorder_segments"] > 0),
                "has_sdr": int(per_protein["sdr_segments"] > 0),
                "has_ldr": int(per_protein["ldr_segments"] > 0),
                "has_terminal_idr": int(per_protein["terminal_segments"] > 0),
                "has_internal_idr": int(per_protein["internal_segments"] > 0),
            }
        )

        counters["proteins"] += 1
        counters["residues"] += length
        counters["known_residues"] += known
        counters["disordered"] += disordered
        counters["ordered"] += ordered
        counters["unknown"] += unknown
        counters["disorder_segments"] += per_protein["disorder_segments"]
        counters["sdr_segments"] += per_protein["sdr_segments"]
        counters["ldr_segments"] += per_protein["ldr_segments"]
        counters["terminal_segments"] += per_protein["terminal_segments"]
        counters["internal_segments"] += per_protein["internal_segments"]
        counters["sdr_residues"] += per_protein["sdr_residues"]
        counters["ldr_residues"] += per_protein["ldr_residues"]
        counters["terminal_idr_residues"] += per_protein["terminal_idr_residues"]
        counters["internal_idr_residues"] += per_protein["internal_idr_residues"]
        counters[f"length_{len_bin}"] += 1
        counters[f"dc_{dc_bin}"] += 1
        if per_protein["disorder_segments"] > 0:
            counters["proteins_with_disorder"] += 1
        if per_protein["sdr_segments"] > 0:
            counters["proteins_with_sdr"] += 1
        if per_protein["ldr_segments"] > 0:
            counters["proteins_with_ldr"] += 1
        if per_protein["terminal_segments"] > 0:
            counters["proteins_with_terminal_idr"] += 1
        if per_protein["internal_segments"] > 0:
            counters["proteins_with_internal_idr"] += 1

    summary_row: dict[str, object] = {
        "dataset": spec.name,
        "split_type": spec.split_type,
        "path": str(spec.path),
        "proteins": counters["proteins"],
        "residues": counters["residues"],
        "known_residues": counters["known_residues"],
        "disordered": counters["disordered"],
        "ordered": counters["ordered"],
        "unknown": counters["unknown"],
        "disorder_rate_known": format_float(
            counters["disordered"] / counters["known_residues"] if counters["known_residues"] else math.nan
        ),
        "proteins_with_disorder": counters["proteins_with_disorder"],
        "proteins_with_sdr": counters["proteins_with_sdr"],
        "proteins_with_ldr": counters["proteins_with_ldr"],
        "proteins_with_terminal_idr": counters["proteins_with_terminal_idr"],
        "proteins_with_internal_idr": counters["proteins_with_internal_idr"],
        "disorder_segments": counters["disorder_segments"],
        "sdr_segments": counters["sdr_segments"],
        "ldr_segments": counters["ldr_segments"],
        "terminal_segments": counters["terminal_segments"],
        "internal_segments": counters["internal_segments"],
        "sdr_residues": counters["sdr_residues"],
        "ldr_residues": counters["ldr_residues"],
        "terminal_idr_residues": counters["terminal_idr_residues"],
        "internal_idr_residues": counters["internal_idr_residues"],
        "length_<=200": counters["length_<=200"],
        "length_201-500": counters["length_201-500"],
        "length_501-1000": counters["length_501-1000"],
        "length_>1000": counters["length_>1000"],
        "dc_0-5": counters["dc_0-5"],
        "dc_5-20": counters["dc_5-20"],
        "dc_20-80": counters["dc_20-80"],
        "dc_80-100": counters["dc_80-100"],
        "dc_no_known": counters["dc_no_known"],
    }
    return summary_row, protein_rows, segment_rows


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.6f}"


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def default_specs() -> list[DatasetSpec]:
    return [DatasetSpec(name=name, split_type=split_type, path=path) for name, split_type, path in DEFAULT_DATASETS]


def run(
    dataset_specs: list[DatasetSpec],
    protein_out: Path,
    segments_out: Path,
    summary_out: Path,
    test_summary_out: Path | None,
) -> None:
    summary_rows: list[dict[str, object]] = []
    protein_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    for spec in dataset_specs:
        summary_row, dataset_proteins, dataset_segments = summarize_dataset(spec)
        summary_rows.append(summary_row)
        protein_rows.extend(dataset_proteins)
        segment_rows.extend(dataset_segments)

    summary_fields = [
        "dataset",
        "split_type",
        "path",
        "proteins",
        "residues",
        "known_residues",
        "disordered",
        "ordered",
        "unknown",
        "disorder_rate_known",
        "proteins_with_disorder",
        "proteins_with_sdr",
        "proteins_with_ldr",
        "proteins_with_terminal_idr",
        "proteins_with_internal_idr",
        "disorder_segments",
        "sdr_segments",
        "ldr_segments",
        "terminal_segments",
        "internal_segments",
        "sdr_residues",
        "ldr_residues",
        "terminal_idr_residues",
        "internal_idr_residues",
        "length_<=200",
        "length_201-500",
        "length_501-1000",
        "length_>1000",
        "dc_0-5",
        "dc_5-20",
        "dc_20-80",
        "dc_80-100",
        "dc_no_known",
    ]
    protein_fields = [
        "dataset",
        "split_type",
        "protein_id",
        "sequence_length",
        "known_residues",
        "disordered",
        "ordered",
        "unknown",
        "disorder_rate_known",
        "disorder_content_bin",
        "protein_length_bin",
        "disorder_segments",
        "sdr_segments",
        "ldr_segments",
        "terminal_segments",
        "internal_segments",
        "sdr_residues",
        "ldr_residues",
        "terminal_idr_residues",
        "internal_idr_residues",
        "has_disorder",
        "has_sdr",
        "has_ldr",
        "has_terminal_idr",
        "has_internal_idr",
    ]
    segment_fields = [
        "dataset",
        "split_type",
        "protein_id",
        "segment_index",
        "start",
        "end",
        "length",
        "region_length_type",
        "terminal_status",
        "sequence_length",
        "known_residues",
        "protein_disorder_content",
        "disorder_content_bin",
        "protein_length_bin",
    ]

    write_tsv(summary_out, summary_rows, summary_fields)
    write_tsv(protein_out, protein_rows, protein_fields)
    write_tsv(segments_out, segment_rows, segment_fields)
    if test_summary_out is not None:
        test_rows = [row for row in summary_rows if row["split_type"] == "test"]
        write_tsv(test_summary_out, test_rows, summary_fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset_arg,
        help="Dataset as NAME=PATH. If omitted, main DM3000/DM1229/test/NR25 files are used.",
    )
    parser.add_argument(
        "--protein-out",
        type=Path,
        default=Path("results/region_annotations/protein_region_annotations.tsv"),
    )
    parser.add_argument(
        "--segments-out",
        type=Path,
        default=Path("results/region_annotations/disorder_segments.tsv"),
    )
    parser.add_argument("--summary-out", type=Path, default=Path("results/dataset_region_summary.tsv"))
    parser.add_argument("--test-summary-out", type=Path, default=Path("results/testset_region_summary.tsv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = args.dataset if args.dataset else default_specs()
    run(
        dataset_specs=specs,
        protein_out=args.protein_out,
        segments_out=args.segments_out,
        summary_out=args.summary_out,
        test_summary_out=args.test_summary_out,
    )


if __name__ == "__main__":
    main()
