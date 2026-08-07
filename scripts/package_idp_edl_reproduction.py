#!/usr/bin/env python3
"""Validate and package the public IDP-EDL reproduction results.

The local prediction tables are deliberately treated as private validation
inputs.  The package contains a reduced residue table and its deterministic
gzip representation; sequence, amino-acid, header, and label columns never
leave the validation boundary.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction.idp_edl.data import (  # noqa: E402
    DATASET_ORDER,
    default_sample_paths,
    normalize_for_prott5,
    parse_author_samples,
)
from reproduction.idp_edl.metrics import compute_binary_metrics  # noqa: E402


DATASETS = ("sl329", "mxd494", "disorder723")
MAX_LENGTH = 1024
MAX_RETAINED_RESIDUES = MAX_LENGTH - 1
THRESHOLD = 0.5
METRIC_TOLERANCE = 1e-6
PACKAGE_VERSION = "idp-edl-reproduction-public-v1"

SOURCE_RESIDUE_COLUMNS = (
    "dataset",
    "protein_id",
    "header",
    "position",
    "aa",
    "model_aa",
    "label",
    "included_in_metrics",
    "prediction",
    "score",
    "max_length",
    "truncated_residues",
)

PUBLIC_RESIDUE_COLUMNS = (
    "dataset",
    "protein_id",
    "position",
    "prediction",
    "score",
    "max_length",
    "truncated_residues",
)

SOURCE_SUMMARY_COLUMNS = (
    "dataset",
    "status",
    "proteins",
    "original_residues",
    "retained_residues",
    "truncated_residues",
    "label_0_retained",
    "label_1_retained",
    "label_2_excluded",
    "evaluated_residues",
    "accuracy",
    "auc",
    "sensitivity",
    "specificity",
    "bacc",
    "mcc",
    "fmax",
    "threshold",
)

SUMMARY_COLUMNS = SOURCE_SUMMARY_COLUMNS[:12] + ("aupr",) + SOURCE_SUMMARY_COLUMNS[12:]

PAPER_COMPARISON_COLUMNS = (
    "dataset",
    "metric",
    "paper_reported",
    "local_recomputed",
    "delta_local_minus_paper",
    "paper_source",
    "status",
)

# Values transcribed from the IDP-EDL paper tables.  Keeping them here makes
# the comparison auditable and prevents a paper value from being inferred
# from a generated result file.
PAPER_VALUES: Dict[str, Dict[str, float]] = {
    "sl329": {
        "Sn": 0.690,
        "Sp": 0.970,
        "BACC": 0.828,
        "MCC": 0.700,
        "AUC": 0.915,
    },
    "mxd494": {
        "Sn": 0.679,
        "Sp": 0.843,
        "BACC": 0.761,
        "MCC": 0.488,
        "AUC": 0.837,
    },
    "disorder723": {
        "Sn": 0.603,
        "Sp": 0.984,
        "BACC": 0.793,
        "MCC": 0.636,
        "AUC": 0.943,
    },
}

PAPER_SOURCES = {
    "sl329": "IDP-EDL Table 5",
    "mxd494": "IDP-EDL Table 4",
    "disorder723": "IDP-EDL Table 6",
}


class PackageValidationError(ValueError):
    """Raised when a local result violates the IDP-EDL publication contract."""


def _resolve(root: Path, value: Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else Path(root) / value


def _fail(path: Path, message: str) -> None:
    raise PackageValidationError("{}: {}".format(path, message))


def _require_file(path: Path) -> None:
    if not path.is_file():
        _fail(path, "required input file is missing")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_metadata(path: Path, project_root: Path, rows: Optional[int] = None) -> Dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        display_path = str(resolved)
    result: Dict[str, Any] = {
        "path": display_path,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _read_tsv(path: Path, expected_columns: Sequence[str]) -> List[Dict[str, str]]:
    _require_file(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != tuple(expected_columns):
            _fail(
                path,
                "unexpected columns; expected {}, got {}".format(
                    list(expected_columns), reader.fieldnames
                ),
            )
        rows: List[Dict[str, str]] = []
        for line_number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                _fail(path, "malformed fields at line {}".format(line_number))
            if any(value == "" for value in row.values()):
                _fail(path, "empty field at line {}".format(line_number))
            rows.append(row)
    if not rows:
        _fail(path, "contains no data rows")
    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    _require_file(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        _fail(path, "invalid JSON: {}".format(exc))
    if not isinstance(value, dict):
        _fail(path, "top-level JSON value must be an object")
    return value


def _parse_int(value: Any, path: Path, field: str) -> int:
    text = str(value).strip()
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        _fail(path, "{} must be an integer, got {!r}".format(field, value))
    if text != str(parsed):
        _fail(path, "{} must use an integer representation, got {!r}".format(field, value))
    return parsed


def _parse_float(value: Any, path: Path, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        _fail(path, "{} must be numeric, got {!r}".format(field, value))
    if not math.isfinite(parsed):
        _fail(path, "{} must be finite, got {!r}".format(field, value))
    return parsed


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return "{:.9f}".format(value)
    return str(value)


def _write_tsv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(columns),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_value(row.get(column, "")) for column in columns})


def _tsv_bytes(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    """Render a deterministic TSV without leaving an uncompressed artifact."""

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=list(columns),
        delimiter="\t",
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _format_value(row.get(column, "")) for column in columns})
    return handle.getvalue().encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def _write_deterministic_gzip(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=9,
            mtime=0,
        ) as compressed:
            compressed.write(content)


def _compare_number(path: Path, field: str, actual: Any, expected: float) -> None:
    value = _parse_float(actual, path, field)
    if abs(value - expected) > METRIC_TOLERANCE:
        _fail(
            path,
            "{} mismatch: expected {:.12f}, got {:.12f}".format(field, expected, value),
        )


def _compare_summary_value(path: Path, field: str, actual: Any, expected: Any) -> None:
    if field in {
        "proteins",
        "original_residues",
        "retained_residues",
        "truncated_residues",
        "label_0_retained",
        "label_1_retained",
        "label_2_excluded",
        "evaluated_residues",
    }:
        if _parse_int(actual, path, field) != int(expected):
            _fail(path, "{} mismatch: expected {}, got {!r}".format(field, expected, actual))
        return
    if field == "dataset" or field == "status":
        if str(actual) != str(expected):
            _fail(path, "{} mismatch: expected {!r}, got {!r}".format(field, expected, actual))
        return
    _compare_number(path, field, actual, float(expected))


def _expected_summary(
    dataset: str,
    examples: Sequence[Any],
    grouped_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    labels: List[int] = []
    scores: List[float] = []
    for example in examples:
        for row in grouped_rows[example.identifier]:
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))

    metrics = compute_binary_metrics(labels, scores, threshold=THRESHOLD)
    evaluated = [(label, score) for label, score in zip(labels, scores) if label in (0, 1)]
    if evaluated and any(label == 1 for label, _ in evaluated):
        aupr = float(
            average_precision_score(
                [label for label, _ in evaluated],
                [score for _, score in evaluated],
            )
        )
    else:
        aupr = None
    result: Dict[str, Any] = {
        "dataset": dataset,
        "status": "predicted",
        "proteins": len(examples),
        "original_residues": sum(len(example.sequence) for example in examples),
        "retained_residues": len(labels),
        "truncated_residues": sum(
            max(0, len(example.sequence) - MAX_RETAINED_RESIDUES) for example in examples
        ),
        "label_0_retained": sum(label == 0 for label in labels),
        "label_1_retained": sum(label == 1 for label in labels),
        "label_2_excluded": sum(label == 2 for label in labels),
        "evaluated_residues": sum(label in (0, 1) for label in labels),
        "accuracy": metrics["accuracy"],
        "auc": metrics["auc"],
        "aupr": aupr,
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "bacc": metrics["bacc"],
        "mcc": metrics["mcc"],
        "fmax": metrics["fmax"],
        "threshold": THRESHOLD,
        "tp": metrics.get("tp"),
        "tn": metrics.get("tn"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
    }
    return result


def _validate_source_summary(
    summary_tsv_path: Path,
    summary_json_path: Path,
    dataset: str,
    expected: Mapping[str, Any],
) -> None:
    tsv_rows = _read_tsv(summary_tsv_path, SOURCE_SUMMARY_COLUMNS)
    if len(tsv_rows) != 1:
        _fail(summary_tsv_path, "expected one dataset summary row, got {}".format(len(tsv_rows)))
    tsv_row = tsv_rows[0]
    for field in SOURCE_SUMMARY_COLUMNS:
        _compare_summary_value(summary_tsv_path, field, tsv_row[field], expected[field])

    value = _read_json(summary_json_path)
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        _fail(summary_json_path, "metadata object is missing")
    protocol_checks = {
        "max_length": MAX_LENGTH,
        "batch_size": 4,
        "eos_tokens": 1,
    }
    for field, expected_value in protocol_checks.items():
        if metadata.get(field) != expected_value:
            _fail(
                summary_json_path,
                "metadata.{} must be {}, got {!r}".format(field, expected_value, metadata.get(field)),
            )
    if metadata.get("label_2_policy") != "retained in residue TSV and excluded from metrics":
        _fail(summary_json_path, "metadata.label_2_policy is inconsistent with the protocol")

    json_rows = value.get("datasets")
    if not isinstance(json_rows, list) or len(json_rows) != 1 or not isinstance(json_rows[0], dict):
        _fail(summary_json_path, "datasets must contain exactly one summary object")
    json_row = json_rows[0]
    for field in SOURCE_SUMMARY_COLUMNS:
        if field not in json_row:
            _fail(summary_json_path, "dataset summary is missing field {!r}".format(field))
        _compare_summary_value(summary_json_path, field, json_row[field], expected[field])
    for field in ("tp", "tn", "fp", "fn"):
        if field in json_row and json_row[field] != expected[field]:
            _fail(
                summary_json_path,
                "{} mismatch: expected {}, got {!r}".format(field, expected[field], json_row[field]),
            )


def _author_examples(
    sample_sequences: Path,
    sample_labels: Path,
) -> Dict[str, List[Any]]:
    examples = parse_author_samples(sample_sequences, sample_labels, DATASETS)
    by_dataset: Dict[str, List[Any]] = {dataset: [] for dataset in DATASETS}
    seen: Dict[str, set] = {dataset: set() for dataset in DATASETS}
    for example in examples:
        if example.dataset not in by_dataset:
            _fail(sample_sequences, "unsupported dataset suffix {!r}".format(example.dataset))
        if example.identifier in seen[example.dataset]:
            _fail(
                sample_sequences,
                "duplicate protein identifier {} in {}".format(example.identifier, example.dataset),
            )
        seen[example.dataset].add(example.identifier)
        by_dataset[example.dataset].append(example)
    return by_dataset


def _validate_residue_rows(
    path: Path,
    dataset: str,
    rows: Sequence[Mapping[str, str]],
    examples: Sequence[Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    expected_by_id = {example.identifier: example for example in examples}
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_positions: Dict[str, set] = defaultdict(set)
    public_rows: List[Dict[str, Any]] = []

    for line_number, source_row in enumerate(rows, 2):
        row_dataset = source_row["dataset"]
        protein_id = source_row["protein_id"]
        if row_dataset != dataset:
            _fail(path, "line {} has dataset {!r}, expected {!r}".format(line_number, row_dataset, dataset))
        if protein_id not in expected_by_id:
            _fail(path, "line {} references unknown protein {!r}".format(line_number, protein_id))
        example = expected_by_id[protein_id]
        position = _parse_int(source_row["position"], path, "position at line {}".format(line_number))
        if position < 1 or position > MAX_RETAINED_RESIDUES:
            _fail(path, "line {} has an invalid position {}".format(line_number, position))
        if source_row["header"] != example.sequence_header:
            _fail(path, "line {} header does not match author FASTA".format(line_number))
        if position > len(example.sequence):
            _fail(path, "line {} exceeds the author sequence length".format(line_number))
        expected_aa = example.sequence[position - 1]
        expected_model_aa = normalize_for_prott5(expected_aa)
        if source_row["aa"] != expected_aa:
            _fail(path, "line {} aa does not match author FASTA".format(line_number))
        if source_row["model_aa"] != expected_model_aa:
            _fail(path, "line {} model_aa does not match the tokenizer protocol".format(line_number))
        label = source_row["label"]
        if label not in ("0", "1", "2") or label != example.labels[position - 1]:
            _fail(path, "line {} label does not match the author FASTA".format(line_number))
        included = _parse_int(
            source_row["included_in_metrics"], path, "included_in_metrics at line {}".format(line_number)
        )
        if included not in (0, 1) or included != int(label in ("0", "1")):
            _fail(path, "line {} has an invalid included_in_metrics value".format(line_number))
        prediction = _parse_int(
            source_row["prediction"], path, "prediction at line {}".format(line_number)
        )
        if prediction not in (0, 1):
            _fail(path, "line {} prediction must be 0 or 1".format(line_number))
        score = _parse_float(source_row["score"], path, "score at line {}".format(line_number))
        if score < 0.0 or score > 1.0:
            _fail(path, "line {} score is outside [0, 1]".format(line_number))
        if prediction != int(score > THRESHOLD):
            _fail(path, "line {} prediction is inconsistent with score > 0.5".format(line_number))
        max_length = _parse_int(source_row["max_length"], path, "max_length at line {}".format(line_number))
        if max_length != MAX_LENGTH:
            _fail(path, "line {} max_length must be {}".format(line_number, MAX_LENGTH))
        truncated = _parse_int(
            source_row["truncated_residues"], path, "truncated_residues at line {}".format(line_number)
        )
        expected_truncated = max(0, len(example.sequence) - MAX_RETAINED_RESIDUES)
        if truncated != expected_truncated:
            _fail(
                path,
                "line {} truncated_residues mismatch: expected {}, got {}".format(
                    line_number, expected_truncated, truncated
                ),
            )
        if position in seen_positions[protein_id]:
            _fail(path, "line {} duplicates position {} for {}".format(line_number, position, protein_id))
        seen_positions[protein_id].add(position)

        validated = {
            "dataset": dataset,
            "protein_id": protein_id,
            "position": position,
            "label": int(label),
            "prediction": prediction,
            "score": score,
            "max_length": max_length,
            "truncated_residues": truncated,
        }
        groups[protein_id].append(validated)
        public_rows.append(
            {
                "dataset": dataset,
                "protein_id": protein_id,
                "position": position,
                "prediction": prediction,
                "score": score,
                "max_length": max_length,
                "truncated_residues": truncated,
            }
        )

    if set(groups) != set(expected_by_id):
        missing = sorted(set(expected_by_id).difference(groups))
        extra = sorted(set(groups).difference(expected_by_id))
        _fail(path, "protein set mismatch; missing={}, extra={}".format(missing, extra))

    ordered_groups: Dict[str, List[Dict[str, Any]]] = {}
    for example in examples:
        group = sorted(groups[example.identifier], key=lambda item: item["position"])
        retained = min(len(example.sequence), MAX_RETAINED_RESIDUES)
        positions = [item["position"] for item in group]
        if positions != list(range(1, retained + 1)):
            _fail(
                path,
                "positions for {} are not the consecutive range 1..{}".format(
                    example.identifier, retained
                ),
            )
        ordered_groups[example.identifier] = group

    expected_row_count = sum(
        min(len(example.sequence), MAX_RETAINED_RESIDUES) for example in examples
    )
    if len(rows) != expected_row_count:
        _fail(path, "row count mismatch: expected {}, got {}".format(expected_row_count, len(rows)))
    summary = _expected_summary(dataset, examples, ordered_groups)
    return public_rows, summary


def _source_paths(outputs_root: Path, dataset: str) -> Tuple[Path, Path, Path]:
    directory = outputs_root / "{}_fp32_batch4".format(dataset)
    return (
        directory / "idp_edl_residue_predictions.tsv",
        directory / "idp_edl_summary.tsv",
        directory / "idp_edl_summary.json",
    )


def _metric_for_paper(summary: Mapping[str, Any], metric: str) -> float:
    mapping = {
        "Sn": "sensitivity",
        "Sp": "specificity",
        "BACC": "bacc",
        "MCC": "mcc",
        "AUC": "auc",
    }
    value = summary[mapping[metric]]
    if value is None:
        raise PackageValidationError("{} is unavailable for paper comparison".format(metric))
    return float(value)


def package_results(
    outputs_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    sample_sequences: Optional[Path] = None,
    sample_labels: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate local predictions and write the public reproduction package."""

    root = Path(project_root or ROOT).resolve()
    outputs = _resolve(root, outputs_root or Path("reproduction/idp_edl/outputs"))
    output = _resolve(root, output_dir or Path("results/reproduction/idp_edl"))
    default_sequences, default_labels = default_sample_paths(root)
    sequences = _resolve(root, sample_sequences or default_sequences)
    labels = _resolve(root, sample_labels or default_labels)
    by_dataset = _author_examples(sequences, labels)

    validated: Dict[str, Dict[str, Any]] = {}
    for dataset in DATASETS:
        residue_path, summary_tsv_path, summary_json_path = _source_paths(outputs, dataset)
        residue_rows = _read_tsv(residue_path, SOURCE_RESIDUE_COLUMNS)
        public_rows, expected_summary = _validate_residue_rows(
            residue_path, dataset, residue_rows, by_dataset[dataset]
        )
        _validate_source_summary(summary_tsv_path, summary_json_path, dataset, expected_summary)
        validated[dataset] = {
            "residue_path": residue_path,
            "summary_tsv_path": summary_tsv_path,
            "summary_json_path": summary_json_path,
            "residue_rows_count": len(residue_rows),
            "public_rows": public_rows,
            "summary": expected_summary,
        }

    output.mkdir(parents=True, exist_ok=True)
    public_file_records: Dict[str, Dict[str, Any]] = {}
    for dataset in DATASETS:
        item = validated[dataset]
        public_tsv_path = output / "{}_residue_predictions.tsv".format(dataset)
        public_gzip_path = output / "{}_residue_predictions.tsv.gz".format(dataset)
        try:
            public_tsv_path.unlink()
        except FileNotFoundError:
            pass
        uncompressed = _tsv_bytes(item["public_rows"], PUBLIC_RESIDUE_COLUMNS)
        _write_deterministic_gzip(public_gzip_path, uncompressed)
        public_file_records[dataset] = {
            "content": {
                "filename": public_tsv_path.name,
                "sha256": hashlib.sha256(uncompressed).hexdigest(),
                "size_bytes": len(uncompressed),
                "rows": len(item["public_rows"]),
            },
            "compressed": _file_metadata(public_gzip_path, root, len(item["public_rows"])),
        }

    summary_rows = [validated[dataset]["summary"] for dataset in DATASETS]
    summary_tsv_path = output / "summary.tsv"
    _write_tsv(summary_tsv_path, summary_rows, SUMMARY_COLUMNS)

    summary_json = {
        "metadata": {
            "package_version": PACKAGE_VERSION,
            "datasets": list(DATASETS),
            "max_length": MAX_LENGTH,
            "max_retained_residues": MAX_RETAINED_RESIDUES,
            "eos_tokens": 1,
            "threshold": THRESHOLD,
            "prediction_rule": "prediction = int(score > 0.5)",
            "metric_rule": "compute_binary_metrics on labels 0/1; label 2 excluded",
            "labels_sequence_intentionally_omitted": True,
            "aa_header_intentionally_omitted": True,
            "aggregate_label_counts_included": True,
            "full_source_tables_not_copied": True,
        },
        "datasets": summary_rows,
    }
    summary_json_path = output / "summary.json"
    _write_json(summary_json_path, summary_json)

    paper_rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        summary = validated[dataset]["summary"]
        for metric in ("Sn", "Sp", "BACC", "MCC", "AUC"):
            paper_value = PAPER_VALUES[dataset][metric]
            local_value = _metric_for_paper(summary, metric)
            paper_rows.append(
                {
                    "dataset": dataset.upper(),
                    "metric": metric,
                    "paper_reported": paper_value,
                    "local_recomputed": local_value,
                    "delta_local_minus_paper": local_value - paper_value,
                    "paper_source": PAPER_SOURCES[dataset],
                    "status": "reproduced_within_tolerance"
                    if abs(local_value - paper_value) <= METRIC_TOLERANCE
                    else "not_reproduced",
                }
            )
    paper_path = output / "paper_comparison.tsv"
    _write_tsv(paper_path, paper_rows, PAPER_COMPARISON_COLUMNS)

    datasets_manifest: Dict[str, Any] = {}
    for dataset in DATASETS:
        item = validated[dataset]
        source_residue_path = item["residue_path"]
        source_residue_meta = _file_metadata(
            source_residue_path, root, item["residue_rows_count"]
        )
        datasets_manifest[dataset] = {
            "proteins": item["summary"]["proteins"],
            "raw_residue_rows": item["residue_rows_count"],
            "source": {
                "residue_tsv": source_residue_meta,
                "summary_tsv": _file_metadata(item["summary_tsv_path"], root),
                "summary_json": _file_metadata(item["summary_json_path"], root),
            },
            "public": public_file_records[dataset],
        }

    package_files = {
        "summary_tsv": _file_metadata(summary_tsv_path, root),
        "summary_json": _file_metadata(summary_json_path, root),
        "paper_comparison_tsv": _file_metadata(paper_path, root),
    }
    manifest: Dict[str, Any] = {
        "manifest_version": 1,
        "package_version": PACKAGE_VERSION,
        "protocol": {
            "source_outputs_pattern": "reproduction/idp_edl/outputs/{sl329,mxd494,disorder723}_fp32_batch4",
            "source_residue_columns": list(SOURCE_RESIDUE_COLUMNS),
            "public_residue_columns": list(PUBLIC_RESIDUE_COLUMNS),
            "max_length": MAX_LENGTH,
            "max_retained_residues": MAX_RETAINED_RESIDUES,
            "eos_tokens": 1,
            "threshold": THRESHOLD,
            "prediction_rule": "prediction = int(score > 0.5)",
            "metric_rule": "compute_binary_metrics on labels 0/1; label 2 excluded",
            "label_2_policy": "retained for internal validation only and excluded from metrics",
            "labels_sequence_intentionally_omitted": True,
            "aa_header_intentionally_omitted": True,
            "aggregate_label_counts_included": True,
            "full_source_tables_not_copied": True,
            "attachment_npy_not_used_as_residue_results": True,
        },
        "inputs": {
            "sample_sequences": _file_metadata(sequences, root),
            "sample_labels": _file_metadata(labels, root),
        },
        "datasets": datasets_manifest,
        "package_files": package_files,
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("reproduction/idp_edl/outputs"),
        help="root containing the three *_fp32_batch4 input directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reproduction/idp_edl"),
        help="public package output directory",
    )
    parser.add_argument("--sample-sequences", type=Path, help="override author sample_sequences.fasta")
    parser.add_argument("--sample-labels", type=Path, help="override author sample_labels.fasta")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        manifest = package_results(
            outputs_root=args.outputs_root,
            output_dir=args.output_dir,
            sample_sequences=args.sample_sequences,
            sample_labels=args.sample_labels,
        )
    except (OSError, PackageValidationError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output_dir": str(args.output_dir), "datasets": list(manifest["datasets"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
