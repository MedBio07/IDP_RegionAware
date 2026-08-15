#!/usr/bin/env python3
"""Validate a complete IDP-EDL residue CSV archive and publish v2 outputs.

The archive is a private validation input.  The public residue tables contain
only dataset identity, canonical protein identity, one-based position, and
the four model scores.  Sequence, amino acid, reference label, and evaluable
status are used for validation and metric recomputation but are never copied
to the public residue tables.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction.idp_edl.metrics import compute_binary_metrics  # noqa: E402


DATASETS = ("SL329", "MXD494", "DISORDER723")
DATASET_KEYS = {dataset: dataset.lower() for dataset in DATASETS}
ARCHIVE_MEMBERS = {
    dataset: "residue_predictions_{}.csv".format(DATASET_KEYS[dataset])
    for dataset in DATASETS
}
SOURCE_RESIDUE_COLUMNS = (
    "protein_id",
    "residue_index",
    "amino_acid",
    "true_label",
    "evaluable",
    "idp_edl_g_score",
    "idp_edl_l_score",
    "idp_edl_s_score",
    "idp_edl_score",
)
SCORE_COLUMNS = SOURCE_RESIDUE_COLUMNS[5:]
PUBLIC_RESIDUE_COLUMNS = ("dataset", "protein_id", "position") + SCORE_COLUMNS
SUMMARY_COLUMNS = (
    "dataset",
    "proteins",
    "rows",
    "evaluated",
    "unknown",
    "positive",
    "negative",
    "accuracy",
    "auc",
    "aupr",
    "sn",
    "sp",
    "bacc",
    "mcc",
    "fmax",
    "threshold",
    "tp",
    "tn",
    "fp",
    "fn",
)
COMPONENT_SUMMARY_COLUMNS = ("dataset", "predictor") + SUMMARY_COLUMNS[1:]
PAPER_COMPARISON_COLUMNS = (
    "dataset",
    "metric",
    "paper_reported",
    "local_recomputed",
    "delta_local_minus_paper",
    "paper_source",
    "status",
)

THRESHOLD = 0.5
METRIC_TOLERANCE = 1e-6
PACKAGE_VERSION = "v2"

# Values transcribed from the IDP-EDL paper tables.  They are references for
# the generated comparison table; no paper value is used in validation.
PAPER_VALUES: Dict[str, Dict[str, float]] = {
    "sl329": {"Sn": 0.690, "Sp": 0.970, "BACC": 0.828, "MCC": 0.700, "AUC": 0.915},
    "mxd494": {"Sn": 0.679, "Sp": 0.843, "BACC": 0.761, "MCC": 0.488, "AUC": 0.837},
    "disorder723": {"Sn": 0.603, "Sp": 0.984, "BACC": 0.793, "MCC": 0.636, "AUC": 0.943},
}
PAPER_SOURCES = {
    "sl329": "IDP-EDL Table 5",
    "mxd494": "IDP-EDL Table 4",
    "disorder723": "IDP-EDL Table 6",
}


class PackageValidationError(ValueError):
    """Raised when an archive or local reference file violates the v2 contract."""


@dataclass(frozen=True)
class DatasetRecord:
    """One local three-line FASTA record."""

    dataset: str
    protein_id: str
    sequence: str
    labels: Tuple[int, ...]


def _resolve(root: Path, value: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _fail(path: Path, message: str) -> None:
    raise PackageValidationError("{}: {}".format(path, message))


def _require_file(path: Path) -> None:
    if not path.is_file():
        _fail(path, "required input file is missing")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path, project_root: Path) -> str:
    """Return a public, non-absolute provenance path."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        return path.name


def _file_metadata(
    path: Path,
    project_root: Path,
    rows: Optional[int] = None,
    public_name: Optional[str] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "path": public_name or _display_path(path, project_root),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }
    if rows is not None:
        metadata["rows"] = int(rows)
    return metadata


def _read_text(path: Path) -> List[str]:
    _require_file(path)
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        _fail(path, "could not read UTF-8 input: {}".format(exc))


def _parse_labels(raw: str, path: Path, line_number: int, protein_id: str) -> Tuple[int, ...]:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        _fail(path, "line {} labels for {} are not a list: {}".format(line_number, protein_id, exc))
    if type(value) is not list:
        _fail(path, "line {} labels for {} must be a list".format(line_number, protein_id))
    labels: List[int] = []
    for position, label in enumerate(value, 1):
        if type(label) is not int or label not in (-1, 0, 1):
            _fail(
                path,
                "line {} label {} for {} must be -1, 0, or 1".format(
                    line_number, position, protein_id
                ),
            )
        labels.append(int(label))
    return tuple(labels)


def _read_dataset_fasta(path: Path, dataset: str) -> List[DatasetRecord]:
    """Read the project's exact three-line ``header/sequence/labels`` format."""

    lines = _read_text(path)
    if not lines:
        _fail(path, "contains no records")
    if any(not line.strip() for line in lines):
        _fail(path, "blank lines are not allowed in the three-line record format")
    if len(lines) % 3:
        _fail(path, "expected three lines per record, found {} lines".format(len(lines)))

    records: List[DatasetRecord] = []
    seen_ids: set = set()
    for offset in range(0, len(lines), 3):
        header_line = offset + 1
        raw_header = lines[offset].strip()
        raw_sequence = lines[offset + 1].strip()
        raw_labels = lines[offset + 2].strip()
        if not raw_header.startswith(">") or not raw_header[1:].strip():
            _fail(path, "line {} is not a non-empty FASTA header".format(header_line))
        header = raw_header[1:].strip()
        protein_id = header.split()[0]
        id_key = protein_id.casefold()
        if id_key in seen_ids:
            _fail(path, "duplicate protein ID {!r} (case-insensitive)".format(protein_id))
        seen_ids.add(id_key)
        sequence = "".join(raw_sequence.split()).upper()
        if not sequence or not sequence.isalpha() or not sequence.isupper():
            _fail(path, "line {} has an invalid amino-acid sequence for {}".format(offset + 2, protein_id))
        labels = _parse_labels(raw_labels, path, offset + 3, protein_id)
        if len(sequence) != len(labels):
            _fail(
                path,
                "sequence/label length mismatch for {}: {} != {}".format(
                    protein_id, len(sequence), len(labels)
                ),
            )
        records.append(DatasetRecord(dataset, protein_id, sequence, labels))
    return records


def _read_csv_member(path: Path, member_name: str, content: bytes) -> List[Dict[str, str]]:
    try:
        text = io.TextIOWrapper(io.BytesIO(content), encoding="utf-8", newline="")
        reader = csv.DictReader(text)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != SOURCE_RESIDUE_COLUMNS:
            _fail(
                path,
                "{} has unexpected columns; expected {}, got {}".format(
                    member_name, list(SOURCE_RESIDUE_COLUMNS), list(fieldnames)
                ),
            )
        rows: List[Dict[str, str]] = []
        for line_number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                _fail(path, "{} has malformed fields at line {}".format(member_name, line_number))
            if any(value == "" for value in row.values()):
                _fail(path, "{} has an empty field at line {}".format(member_name, line_number))
            rows.append(dict(row))
    except UnicodeError as exc:
        _fail(path, "{} is not valid UTF-8: {}".format(member_name, exc))
    if not rows:
        _fail(path, "{} contains no data rows".format(member_name))
    return rows


def _parse_int(value: Any, path: Path, field: str) -> int:
    text = str(value)
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        _fail(path, "{} must be an integer, got {!r}".format(field, value))
    if text != str(parsed):
        _fail(path, "{} must use an integer representation, got {!r}".format(field, value))
    return parsed


def _parse_bool(value: Any, path: Path, field: str) -> bool:
    text = str(value).strip().casefold()
    if text in ("1", "true"):
        return True
    if text in ("0", "false"):
        return False
    _fail(path, "{} must be one of 0, 1, true, or false, got {!r}".format(field, value))


def _parse_score(value: Any, path: Path, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        _fail(path, "{} must be numeric, got {!r}".format(field, value))
    if not math.isfinite(parsed):
        _fail(path, "{} must be finite, got {!r}".format(field, value))
    if parsed < 0.0 or parsed > 1.0:
        _fail(path, "{} must be in [0, 1], got {!r}".format(field, value))
    return parsed


def _format_value(value: Any, significant_digits: int = 12) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return "{:.{}g}".format(value, significant_digits)
    return str(value)


def _tsv_bytes(
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
    significant_digits: int = 12,
) -> bytes:
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
        writer.writerow(
            {
                column: _format_value(row.get(column, ""), significant_digits)
                for column in columns
            }
        )
    return handle.getvalue().encode("utf-8")


def _write_tsv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_tsv_bytes(rows, columns))


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


def _is_ignored_archive_metadata(info: zipfile.ZipInfo) -> bool:
    parts = PurePosixPath(info.filename).parts
    basename = parts[-1] if parts else ""
    return info.is_dir() or "__MACOSX" in parts or basename.startswith("._")


def _open_archive(path: Path) -> Tuple[zipfile.ZipFile, Dict[str, zipfile.ZipInfo], int]:
    _require_file(path)
    try:
        archive = zipfile.ZipFile(path, "r")
        infos = archive.infolist()
    except (OSError, zipfile.BadZipFile) as exc:
        _fail(path, "invalid zip archive: {}".format(exc))
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        archive.close()
        _fail(path, "archive contains duplicate member names")

    expected = {name.casefold(): name for name in ARCHIVE_MEMBERS.values()}
    by_name: Dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        if _is_ignored_archive_metadata(info):
            continue
        basename = PurePosixPath(info.filename).name
        logical_name = expected.get(basename.casefold())
        if logical_name is None:
            archive.close()
            _fail(path, "archive contains unexpected file {}".format(info.filename))
        if logical_name in by_name:
            archive.close()
            _fail(path, "archive contains duplicate source member {}".format(logical_name))
        by_name[logical_name] = info

    missing = sorted(set(ARCHIVE_MEMBERS.values()).difference(by_name))
    if missing:
        archive.close()
        _fail(path, "archive is missing source members {}".format(missing))
    return archive, by_name, len(infos)


def _archive_member_metadata(
    info: zipfile.ZipInfo,
    content: bytes,
    logical_name: str,
) -> Dict[str, Any]:
    return {
        "filename": logical_name,
        "crc": int(info.CRC),
        "crc32": "{:08x}".format(info.CRC),
        "size": int(info.file_size),
        "size_bytes": int(info.file_size),
        "compressed_size_bytes": int(info.compress_size),
        "sha256": _sha256_bytes(content),
    }


def _read_archive_rows(
    archive_path: Path,
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, Dict[str, Any]], int]:
    archive, infos, total_member_count = _open_archive(archive_path)
    rows_by_dataset: Dict[str, List[Dict[str, str]]] = {}
    members: Dict[str, Dict[str, Any]] = {}
    try:
        for dataset in DATASETS:
            member_name = ARCHIVE_MEMBERS[dataset]
            info = infos[member_name]
            try:
                content = archive.read(info)
            except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
                _fail(archive_path, "could not read {}: {}".format(member_name, exc))
            if len(content) != info.file_size:
                _fail(
                    archive_path,
                    "{} size mismatch: zip header {}, read {}".format(
                        member_name, info.file_size, len(content)
                    ),
                )
            members[dataset] = _archive_member_metadata(info, content, member_name)
            rows_by_dataset[dataset] = _read_csv_member(archive_path, member_name, content)
    finally:
        archive.close()
    return rows_by_dataset, members, total_member_count


def _validate_dataset_rows(
    path: Path,
    dataset: str,
    records: Sequence[DatasetRecord],
    source_rows: Sequence[Mapping[str, str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    expected_by_id = {record.protein_id.casefold(): record for record in records}
    groups: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for line_number, source_row in enumerate(source_rows, 2):
        raw_id = source_row["protein_id"]
        id_key = raw_id.casefold()
        if id_key not in expected_by_id:
            _fail(path, "{} line {} references unknown protein {!r}".format(ARCHIVE_MEMBERS[dataset], line_number, raw_id))
        record = expected_by_id[id_key]
        position = _parse_int(
            source_row["residue_index"], path, "{} line {} residue_index".format(ARCHIVE_MEMBERS[dataset], line_number)
        )
        if position < 1 or position > len(record.sequence):
            _fail(
                path,
                "{} line {} has extra/out-of-range position {} for {}".format(
                    ARCHIVE_MEMBERS[dataset], line_number, position, record.protein_id
                ),
            )
        if source_row["amino_acid"] != record.sequence[position - 1]:
            _fail(
                path,
                "{} line {} amino_acid mismatch for {} position {}".format(
                    ARCHIVE_MEMBERS[dataset], line_number, record.protein_id, position
                ),
            )
        true_label = _parse_int(
            source_row["true_label"], path, "{} line {} true_label".format(ARCHIVE_MEMBERS[dataset], line_number)
        )
        if true_label != record.labels[position - 1]:
            _fail(
                path,
                "{} line {} true_label mismatch for {} position {}: expected {}, got {}".format(
                    ARCHIVE_MEMBERS[dataset], line_number, record.protein_id, position,
                    record.labels[position - 1], true_label,
                ),
            )
        evaluable = _parse_bool(
            source_row["evaluable"], path, "{} line {} evaluable".format(ARCHIVE_MEMBERS[dataset], line_number)
        )
        expected_evaluable = record.labels[position - 1] in (0, 1)
        if evaluable != expected_evaluable:
            _fail(
                path,
                "{} line {} evaluable mismatch for {} position {}".format(
                    ARCHIVE_MEMBERS[dataset], line_number, record.protein_id, position
                ),
            )
        scores = {
            column: _parse_score(
                source_row[column], path, "{} line {} {}".format(ARCHIVE_MEMBERS[dataset], line_number, column)
            )
            for column in SCORE_COLUMNS
        }
        by_protein = groups.setdefault(record.protein_id, {})
        if position in by_protein:
            _fail(
                path,
                "{} line {} duplicates {} position {}".format(
                    ARCHIVE_MEMBERS[dataset], line_number, record.protein_id, position
                ),
            )
        by_protein[position] = {
            "dataset": dataset,
            "protein_id": record.protein_id,
            "position": position,
            "true_label": true_label,
            "evaluable": evaluable,
            **scores,
        }

    expected_ids = set(expected_by_id)
    observed_ids = {protein_id.casefold() for protein_id in groups}
    if observed_ids != expected_ids:
        missing = sorted(expected_ids.difference(observed_ids))
        extra = sorted(observed_ids.difference(expected_ids))
        _fail(
            path,
            "{} protein set mismatch; missing={}, extra={}".format(
                ARCHIVE_MEMBERS[dataset], missing, extra
            ),
        )

    ordered_rows: List[Dict[str, Any]] = []
    public_rows: List[Dict[str, Any]] = []
    for record in records:
        positions = groups[record.protein_id]
        expected_positions = set(range(1, len(record.sequence) + 1))
        observed_positions = set(positions)
        if observed_positions != expected_positions:
            missing = sorted(expected_positions.difference(observed_positions))
            extra = sorted(observed_positions.difference(expected_positions))
            _fail(
                path,
                "{} positions mismatch for {}; missing={}, extra={}".format(
                    ARCHIVE_MEMBERS[dataset], record.protein_id, missing, extra
                ),
            )
        for position in range(1, len(record.sequence) + 1):
            row = positions[position]
            ordered_rows.append(row)
            public_rows.append(
                {
                    "dataset": dataset,
                    "protein_id": record.protein_id,
                    "position": position,
                    **{column: row[column] for column in SCORE_COLUMNS},
                }
            )

    summary = _compute_summary(dataset, records, ordered_rows, "idp_edl_score")
    return public_rows, {"summary": summary, "rows": ordered_rows}


def _compute_summary(
    dataset: str,
    records: Sequence[DatasetRecord],
    rows: Sequence[Mapping[str, Any]],
    score_column: str,
) -> Dict[str, Any]:
    labels = [int(row["true_label"]) for row in rows]
    scores = [float(row[score_column]) for row in rows]
    evaluated_pairs = [
        (label, score) for label, score in zip(labels, scores) if label in (0, 1)
    ]
    metrics = compute_binary_metrics(labels, scores, threshold=THRESHOLD)
    positive = sum(label == 1 for label in labels)
    negative = sum(label == 0 for label in labels)
    if positive and evaluated_pairs:
        aupr: Optional[float] = float(
            average_precision_score(
                [label for label, _ in evaluated_pairs],
                [score for _, score in evaluated_pairs],
            )
        )
    else:
        aupr = None
    return {
        "dataset": dataset,
        "proteins": len(records),
        "rows": len(rows),
        "evaluated": len(evaluated_pairs),
        "unknown": len(rows) - len(evaluated_pairs),
        "positive": positive,
        "negative": negative,
        "accuracy": metrics.get("accuracy"),
        "auc": metrics.get("auc"),
        "aupr": aupr,
        "sn": metrics.get("sensitivity"),
        "sp": metrics.get("specificity"),
        "bacc": metrics.get("bacc"),
        "mcc": metrics.get("mcc"),
        "fmax": metrics.get("fmax"),
        "threshold": THRESHOLD,
        "tp": int(metrics.get("tp", 0) or 0),
        "tn": int(metrics.get("tn", 0) or 0),
        "fp": int(metrics.get("fp", 0) or 0),
        "fn": int(metrics.get("fn", 0) or 0),
    }


def _component_rows(
    dataset: str,
    records: Sequence[DatasetRecord],
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    predictor_names = (
        ("idp_edl_g", "idp_edl_g_score"),
        ("idp_edl_l", "idp_edl_l_score"),
        ("idp_edl_s", "idp_edl_s_score"),
        ("idp_edl", "idp_edl_score"),
    )
    for predictor, score_column in predictor_names:
        summary = _compute_summary(dataset, records, rows, score_column)
        result.append({"dataset": dataset, "predictor": predictor, **{key: summary[key] for key in SUMMARY_COLUMNS[1:]}})
    return result


def _metric_for_paper(summary: Mapping[str, Any], metric: str) -> Optional[float]:
    mapping = {"Sn": "sn", "Sp": "sp", "BACC": "bacc", "MCC": "mcc", "AUC": "auc"}
    value = summary[mapping[metric]]
    return None if value is None else float(value)


def _paper_rows(summaries: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        key = DATASET_KEYS[dataset]
        summary = summaries[dataset]
        for metric in ("Sn", "Sp", "BACC", "MCC", "AUC"):
            local_value = _metric_for_paper(summary, metric)
            paper_value = PAPER_VALUES[key][metric]
            delta = None if local_value is None else local_value - paper_value
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "paper_reported": paper_value,
                    "local_recomputed": local_value,
                    "delta_local_minus_paper": delta,
                    "paper_source": PAPER_SOURCES[key],
                    "status": (
                        "unavailable"
                        if local_value is None
                        else "reproduced_within_tolerance"
                        if abs(delta or 0.0) <= METRIC_TOLERANCE
                        else "not_reproduced"
                    ),
                }
            )
    return rows


def _output_metadata(path: Path, rows: Optional[int] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "filename": path.name,
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _clean_known_outputs(output: Path) -> None:
    names = [
        *("{}_residue_predictions.tsv.gz".format(DATASET_KEYS[dataset]) for dataset in DATASETS),
        *("{}_residue_predictions.tsv".format(DATASET_KEYS[dataset]) for dataset in DATASETS),
        "summary.tsv",
        "summary.json",
        "component_summary.tsv",
        "paper_comparison.tsv",
        "manifest.json",
    ]
    for name in names:
        candidate = output / name
        if candidate.is_file() or candidate.is_symlink():
            candidate.unlink()


def package_results(
    archive: Path,
    sl329_fasta: Path,
    mxd494_fasta: Path,
    disorder723_fasta: Path,
    output_dir: Path,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate the v2 archive and write a deterministic public package."""

    root = Path(project_root or ROOT).resolve()
    archive_path = _resolve(root, archive).resolve()
    fasta_paths = {
        "SL329": _resolve(root, sl329_fasta).resolve(),
        "MXD494": _resolve(root, mxd494_fasta).resolve(),
        "DISORDER723": _resolve(root, disorder723_fasta).resolve(),
    }
    output = _resolve(root, output_dir).resolve()

    records_by_dataset = {
        dataset: _read_dataset_fasta(fasta_paths[dataset], dataset) for dataset in DATASETS
    }
    rows_by_dataset, member_metadata, total_archive_members = _read_archive_rows(archive_path)

    validated: Dict[str, Dict[str, Any]] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    for dataset in DATASETS:
        public_rows, internal = _validate_dataset_rows(
            archive_path,
            dataset,
            records_by_dataset[dataset],
            rows_by_dataset[dataset],
        )
        validated[dataset] = {
            "public_rows": public_rows,
            "internal_rows": internal["rows"],
            "source_rows": len(rows_by_dataset[dataset]),
            "records": records_by_dataset[dataset],
        }
        summaries[dataset] = internal["summary"]

    # All validation completes before the target directory is touched.
    output.mkdir(parents=True, exist_ok=True)
    _clean_known_outputs(output)
    public_metadata: Dict[str, Dict[str, Any]] = {}
    for dataset in DATASETS:
        key = DATASET_KEYS[dataset]
        gzip_path = output / "{}_residue_predictions.tsv.gz".format(key)
        public_content = _tsv_bytes(
            validated[dataset]["public_rows"],
            PUBLIC_RESIDUE_COLUMNS,
            significant_digits=17,
        )
        _write_deterministic_gzip(gzip_path, public_content)
        public_metadata[dataset] = {
            "content_sha256": _sha256_bytes(public_content),
            "content_size_bytes": len(public_content),
            "compressed": _output_metadata(gzip_path, len(validated[dataset]["public_rows"])),
        }

    summary_path = output / "summary.tsv"
    _write_tsv(summary_path, [summaries[dataset] for dataset in DATASETS], SUMMARY_COLUMNS)
    summary_json_path = output / "summary.json"
    summary_json = {
        "package_version": PACKAGE_VERSION,
        "metadata": {
            "datasets": list(DATASETS),
            "threshold": THRESHOLD,
            "prediction_rule": "positive = idp_edl_score > 0.5",
            "evaluable_rule": "true_label in {0, 1}; -1 is unknown and excluded",
            "metric_rule": "all metrics are recomputed on evaluable residues using idp_edl_score",
            "confusion_matrix_fields": ["tp", "tn", "fp", "fn"],
        },
        "datasets": [summaries[dataset] for dataset in DATASETS],
    }
    _write_json(summary_json_path, summary_json)

    component_path = output / "component_summary.tsv"
    component_rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        component_rows.extend(
            _component_rows(dataset, validated[dataset]["records"], validated[dataset]["internal_rows"])
        )
    _write_tsv(component_path, component_rows, COMPONENT_SUMMARY_COLUMNS)

    paper_path = output / "paper_comparison.tsv"
    paper_rows = _paper_rows(summaries)
    _write_tsv(paper_path, paper_rows, PAPER_COMPARISON_COLUMNS)

    output_paths = {
        summary_path.name: _output_metadata(summary_path, len(DATASETS)),
        summary_json_path.name: _output_metadata(summary_json_path),
        component_path.name: _output_metadata(component_path, len(component_rows)),
        paper_path.name: _output_metadata(paper_path, len(paper_rows)),
    }
    for dataset in DATASETS:
        output_paths["{}_residue_predictions.tsv.gz".format(DATASET_KEYS[dataset])] = public_metadata[dataset]["compressed"]

    manifest: Dict[str, Any] = {
        "manifest_version": 2,
        "package_version": PACKAGE_VERSION,
        "archive": {
            "filename": archive_path.name,
            "sha256": _sha256(archive_path),
            "size_bytes": int(archive_path.stat().st_size),
            "member_count": total_archive_members,
            "source_member_count": len(member_metadata),
            "members": {ARCHIVE_MEMBERS[dataset]: member_metadata[dataset] for dataset in DATASETS},
        },
        "protocol": {
            "package_version": PACKAGE_VERSION,
            "archive_member_names": [ARCHIVE_MEMBERS[dataset] for dataset in DATASETS],
            "ignored_archive_metadata": ["directory entries", "__MACOSX entries", "AppleDouble ._* files"],
            "source_residue_columns": list(SOURCE_RESIDUE_COLUMNS),
            "public_residue_columns": list(PUBLIC_RESIDUE_COLUMNS),
            "fasta_record_format": "three lines: >header, sequence, Python literal labels list",
            "id_matching": "case-insensitive input IDs; public output uses local canonical ID",
            "position_base": 1,
            "require_complete_positions": True,
            "score_columns": list(SCORE_COLUMNS),
            "score_range": [0.0, 1.0],
            "threshold": THRESHOLD,
            "prediction_rule": "positive = idp_edl_score > 0.5",
            "evaluable_rule": "true_label in {0, 1}; -1 is unknown and excluded",
            "metric_rule": "compute_binary_metrics and average_precision_score on evaluable residues",
            "public_redaction": ["amino_acid", "true_label", "evaluable"],
            "deterministic_gzip": {"mtime": 0, "filename_header": "", "compresslevel": 9},
        },
        "inputs": {
            "archive": {"filename": archive_path.name, "sha256": _sha256(archive_path), "size_bytes": int(archive_path.stat().st_size)},
            "fasta": {
                dataset: _file_metadata(fasta_paths[dataset], root, len(records_by_dataset[dataset]))
                for dataset in DATASETS
            },
        },
        "datasets": {
            dataset: {
                "proteins": summaries[dataset]["proteins"],
                "source_member": ARCHIVE_MEMBERS[dataset],
                "source_rows": len(rows_by_dataset[dataset]),
                "public_output": public_metadata[dataset],
            }
            for dataset in DATASETS
        },
        # manifest.json is intentionally excluded from this map because a
        # file cannot contain a stable hash of its own final contents.
        "outputs": output_paths,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="complete residue prediction CSV zip")
    parser.add_argument("--sl329-fasta", type=Path, required=True, help="data/SL329_test.fasta")
    parser.add_argument("--mxd494-fasta", type=Path, required=True, help="data/MXD494_test.fasta")
    parser.add_argument("--disorder723-fasta", type=Path, required=True, help="data/DISORDER723_test.fasta")
    parser.add_argument("--output-dir", type=Path, required=True, help="public v2 package directory")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        manifest = package_results(
            archive=args.archive,
            sl329_fasta=args.sl329_fasta,
            mxd494_fasta=args.mxd494_fasta,
            disorder723_fasta=args.disorder723_fasta,
            output_dir=args.output_dir,
        )
    except (OSError, PackageValidationError, UnicodeError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(json.dumps({"package_version": manifest["package_version"], "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
