#!/usr/bin/env python3
"""Validate and package the DisorderUnetLM prediction archive.

The supplied prediction archive contains one NumPy probability array per
protein, with long proteins represented by ordered ``_P1``, ``_P2`` and so
on members.  This script joins those members to the local compact
header/sequence/labels datasets, validates the complete contract, computes
residue metrics, and writes a deterministic public package.

No model or paper comparison is inferred here.  The archive is treated as a
prediction artifact only; its lack of model, version, and runtime metadata is
reported explicitly in the generated manifest and README.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction.idp_edl.metrics import compute_binary_metrics  # noqa: E402


DATASETS = ("SL329", "MXD494", "DISORDER723")
DATASET_FILES = {dataset: dataset + "_test.fasta" for dataset in DATASETS}
EXPECTED_PROTEINS = {"SL329": 329, "MXD494": 494, "DISORDER723": 723}
THRESHOLD = 0.5
ROW_SUM_ATOL = 1.0e-5
ROW_SUM_RTOL = 1.0e-5
PACKAGE_VERSION = "disorderunetlm-predictions-v1"

RESIDUE_COLUMNS = ("dataset", "protein_id", "position", "prediction", "score")
SUMMARY_COLUMNS = (
    "dataset",
    "proteins",
    "source_residues",
    "predicted_residues",
    "excluded_label_minus1",
    "evaluated_residues",
    "positives",
    "negatives",
    "threshold",
    "tp",
    "tn",
    "fp",
    "fn",
    "accuracy",
    "sn",
    "sp",
    "bacc",
    "mcc",
    "auc",
    "aupr",
    "fmax",
)
PROTEIN_SUMMARY_COLUMNS = (
    "dataset",
    "protein_id",
    "sequence_length",
    "source_parts",
    "predicted_disordered_residues",
    "predicted_disordered_fraction",
    "score_mean",
    "score_std",
    "score_min",
    "score_max",
)
METRIC_NAMES = ("accuracy", "sn", "sp", "bacc", "mcc", "auc", "aupr", "fmax")
ARRAY_DTYPE = np.dtype("float32")
PROTEIN_PART_RE = re.compile(r"^(?P<protein>.+)_P(?P<part>[1-9][0-9]*)$")
SEQUENCE_RE = re.compile(r"^[A-Z]+$")


class PackagingError(ValueError):
    """Raised when the archive or source datasets violate the input contract."""


@dataclass(frozen=True)
class DatasetRecord:
    """One local three-line dataset record."""

    dataset: str
    protein_id: str
    header: str
    sequence: str
    labels: Tuple[int, ...]
    record_index: int


@dataclass(frozen=True)
class ArchivePredictions:
    """Validated class probabilities and their archive-member mapping."""

    scores: Dict[str, Dict[str, np.ndarray]]
    members: Dict[str, Dict[str, Tuple[str, ...]]]
    part_lengths: Dict[str, Dict[str, Tuple[int, ...]]]
    member_count: int


def _error(path: Union[Path, str], message: str) -> None:
    raise PackagingError("{}: {}".format(path, message))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        return path.name


def _file_metadata(path: Path, project_root: Path) -> Dict[str, Any]:
    return {
        "path": _display_path(path, project_root),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _require_file(path: Path) -> None:
    if not path.is_file():
        _error(path, "file does not exist")


def _parse_labels(raw: str, path: Path, line_number: int, protein_id: str) -> Tuple[int, ...]:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        _error(path, "line {} labels for {} are not a list literal: {}".format(line_number, protein_id, exc))
    if type(value) is not list:
        _error(path, "line {} labels for {} must be a list".format(line_number, protein_id))
    labels: List[int] = []
    for position, label in enumerate(value, 1):
        if type(label) is not int or label not in (-1, 0, 1):
            _error(
                path,
                "line {} label {} for {} is not one of -1, 0, 1".format(
                    line_number, position, protein_id
                ),
            )
        labels.append(int(label))
    return tuple(labels)


def parse_dataset(path: Union[Path, str], dataset: str) -> List[DatasetRecord]:
    """Parse one compact header/sequence/labels dataset deterministically."""

    path = Path(path).expanduser().resolve()
    dataset = str(dataset).upper()
    if dataset not in DATASETS:
        _error(path, "unsupported dataset {}".format(dataset))
    _require_file(path)
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _error(path, "could not read: {}".format(exc))
    lines = [(number, line.strip()) for number, line in enumerate(raw_lines, 1) if line.strip()]
    if not lines:
        _error(path, "contains no records")
    if len(lines) % 3:
        _error(path, "expected header/sequence/labels triples, found {} non-empty lines".format(len(lines)))

    records: List[DatasetRecord] = []
    seen: set = set()
    for record_index, offset in enumerate(range(0, len(lines), 3)):
        header_line, sequence_line, labels_line = lines[offset : offset + 3]
        header_number, raw_header = header_line
        sequence_number, raw_sequence = sequence_line
        labels_number, raw_labels = labels_line
        if not raw_header.startswith(">") or not raw_header[1:].strip():
            _error(path, "line {} is not a non-empty FASTA header".format(header_number))
        header = raw_header[1:].strip()
        protein_id = header.split()[0]
        if protein_id in seen:
            _error(path, "duplicate protein ID {}".format(protein_id))
        seen.add(protein_id)
        sequence = "".join(raw_sequence.split()).upper()
        if not sequence or SEQUENCE_RE.fullmatch(sequence) is None:
            _error(path, "line {} has an invalid sequence for {}".format(sequence_number, protein_id))
        labels = _parse_labels(raw_labels, path, labels_number, protein_id)
        if len(sequence) != len(labels):
            _error(
                path,
                "sequence/label length mismatch for {}: {} != {}".format(
                    protein_id, len(sequence), len(labels)
                ),
            )
        records.append(
            DatasetRecord(
                dataset=dataset,
                protein_id=protein_id,
                header=header,
                sequence=sequence,
                labels=labels,
                record_index=record_index,
            )
        )

    expected_count = EXPECTED_PROTEINS[dataset]
    if len(records) != expected_count:
        _error(path, "expected {} proteins, found {}".format(expected_count, len(records)))
    return records


def load_datasets(data_dir: Union[Path, str]) -> Dict[str, List[DatasetRecord]]:
    """Load and count all three required local datasets."""

    data_dir = Path(data_dir).expanduser().resolve()
    return {
        dataset: parse_dataset(data_dir / DATASET_FILES[dataset], dataset)
        for dataset in DATASETS
    }


def _read_probability_array(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    """Read one NPY member without enabling object deserialization."""

    try:
        raw = zf.read(member)
        loaded = np.load(io.BytesIO(raw), allow_pickle=False)
    except (OSError, ValueError, TypeError, EOFError) as exc:
        _error(member, "could not read NPY with allow_pickle=False: {}".format(exc))
    if not isinstance(loaded, np.ndarray):
        _error(member, "NPY payload is not an ndarray")
    array = np.asarray(loaded)
    if array.dtype != ARRAY_DTYPE:
        _error(member, "array dtype must be float32, got {}".format(array.dtype))
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 1:
        _error(member, "array shape must be [L, 2] with L >= 1, got {}".format(array.shape))
    if not np.all(np.isfinite(array)):
        _error(member, "array contains non-finite probabilities")
    if np.any(array < 0.0) or np.any(array > 1.0):
        _error(member, "probabilities must lie in [0, 1]")
    if not np.allclose(array.sum(axis=1), 1.0, atol=ROW_SUM_ATOL, rtol=ROW_SUM_RTOL):
        _error(member, "probability rows do not sum to 1 within tolerance")
    return array.copy()


def _parse_archive_member(
    member: str,
    expected_ids: Mapping[str, DatasetRecord],
) -> Tuple[str, Optional[int]]:
    """Return exact protein ID and optional part number for one ZIP member."""

    path = PurePosixPath(member)
    if len(path.parts) != 2 or path.parts[0] not in DATASETS:
        _error(member, "member must be DATASET/protein.npy under an allowed dataset directory")
    filename = path.parts[1]
    if not filename.endswith(".npy") or filename == ".npy":
        _error(member, "member must have a .npy suffix")
    stem = filename[:-4]
    if stem in expected_ids:
        return stem, None
    match = PROTEIN_PART_RE.fullmatch(stem)
    if match is None or match.group("protein") not in expected_ids:
        _error(member, "member names an unknown protein ID")
    return match.group("protein"), int(match.group("part"))


def load_archive(
    archive_path: Union[Path, str],
    records_by_dataset: Mapping[str, Sequence[DatasetRecord]],
) -> ArchivePredictions:
    """Validate every archive member and join multipart proteins in order."""

    archive_path = Path(archive_path).expanduser().resolve()
    _require_file(archive_path)
    expected_by_dataset: Dict[str, Dict[str, DatasetRecord]] = {
        dataset: {record.protein_id: record for record in records_by_dataset[dataset]}
        for dataset in DATASETS
    }
    members_by_dataset: Dict[str, Dict[str, List[Tuple[Optional[int], str]]]] = {
        dataset: {} for dataset in DATASETS
    }
    try:
        zf = zipfile.ZipFile(str(archive_path), "r")
    except (OSError, zipfile.BadZipFile) as exc:
        _error(archive_path, "not a readable ZIP archive: {}".format(exc))

    with zf:
        names = zf.namelist()
        if len(names) != len(set(names)):
            _error(archive_path, "contains duplicate ZIP member names")
        if not names:
            _error(archive_path, "contains no prediction members")
        for info in sorted(zf.infolist(), key=lambda item: item.filename):
            member = info.filename
            if info.is_dir():
                _error(member, "directory entries are not prediction arrays")
            path = PurePosixPath(member)
            if len(path.parts) != 2 or path.parts[0] not in DATASETS:
                _error(member, "unexpected dataset directory or unsafe ZIP path")
            dataset = path.parts[0]
            protein_id, part = _parse_archive_member(member, expected_by_dataset[dataset])
            entries = members_by_dataset[dataset].setdefault(protein_id, [])
            if any(existing_part == part for existing_part, _ in entries):
                _error(member, "duplicate unparted or part number for {}".format(protein_id))
            entries.append((part, member))

        scores: Dict[str, Dict[str, np.ndarray]] = {dataset: {} for dataset in DATASETS}
        ordered_members: Dict[str, Dict[str, Tuple[str, ...]]] = {dataset: {} for dataset in DATASETS}
        part_lengths: Dict[str, Dict[str, Tuple[int, ...]]] = {dataset: {} for dataset in DATASETS}
        for dataset in DATASETS:
            expected_ids = set(expected_by_dataset[dataset])
            observed_ids = set(members_by_dataset[dataset])
            missing = sorted(expected_ids - observed_ids)
            extra = sorted(observed_ids - expected_ids)
            if missing or extra:
                _error(
                    archive_path,
                    "{} protein mapping mismatch; missing={}, extra={}".format(
                        dataset, missing[:10], extra[:10]
                    ),
                )
            for record in records_by_dataset[dataset]:
                entries = members_by_dataset[dataset][record.protein_id]
                unparted = [name for part, name in entries if part is None]
                numbered = [(int(part), name) for part, name in entries if part is not None]
                if unparted and numbered:
                    _error(record.protein_id, "cannot mix an unparted member with _P parts")
                if unparted:
                    if len(unparted) != 1:
                        _error(record.protein_id, "expected one unparted member")
                    ordered_names = tuple(unparted)
                else:
                    part_numbers = sorted(part for part, _ in numbered)
                    expected_parts = list(range(1, len(part_numbers) + 1))
                    if part_numbers != expected_parts:
                        _error(
                            record.protein_id,
                            "parts must be contiguous and start at _P1; got {}".format(part_numbers),
                        )
                    names_by_part = {part: name for part, name in numbered}
                    ordered_names = tuple(names_by_part[part] for part in expected_parts)
                arrays = [_read_probability_array(zf, name) for name in ordered_names]
                concatenated = np.concatenate(arrays, axis=0)
                if concatenated.shape[0] != len(record.sequence) or concatenated.shape[0] != len(record.labels):
                    _error(
                        record.protein_id,
                        "concatenated prediction length {} does not equal sequence/labels length {}".format(
                            concatenated.shape[0], len(record.sequence)
                        ),
                    )
                scores[dataset][record.protein_id] = concatenated
                ordered_members[dataset][record.protein_id] = ordered_names
                part_lengths[dataset][record.protein_id] = tuple(int(array.shape[0]) for array in arrays)
    return ArchivePredictions(
        scores=scores,
        members=ordered_members,
        part_lengths=part_lengths,
        member_count=sum(
            len(names)
            for dataset_mapping in ordered_members.values()
            for names in dataset_mapping.values()
        ),
    )


def _metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, Any]:
    """Apply the project's binary metric semantics and add AUPR."""

    labels_array = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(scores, dtype=np.float64)
    result = compute_binary_metrics(labels_array, scores_array, threshold=THRESHOLD)
    valid = np.isin(labels_array, (0, 1))
    valid_labels = labels_array[valid]
    valid_scores = scores_array[valid]
    aupr: Optional[float]
    if valid_labels.size and np.unique(valid_labels).size == 2:
        aupr = float(average_precision_score(valid_labels, valid_scores))
    else:
        aupr = None
    result["aupr"] = aupr
    return result


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return "NA" if not np.isfinite(value) else "{:.9f}".format(value)
    return str(value)


def _tsv_bytes(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
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
        writer.writerow({column: _format_value(row.get(column)) for column in columns})
    return handle.getvalue().encode("utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


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


def _dataset_rows(
    dataset: str,
    records: Sequence[DatasetRecord],
    scores_by_id: Mapping[str, np.ndarray],
    members_by_id: Mapping[str, Tuple[str, ...]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    residue_rows: List[Dict[str, Any]] = []
    protein_rows: List[Dict[str, Any]] = []
    all_labels: List[int] = []
    all_scores: List[float] = []
    for record in records:
        scores = scores_by_id[record.protein_id][:, 1].astype(np.float64, copy=False)
        labels = list(record.labels)
        residue_rows.extend(
            {
                "dataset": dataset,
                "protein_id": record.protein_id,
                "position": position,
                "prediction": int(float(score) > THRESHOLD),
                "score": float(score),
            }
            for position, score in enumerate(scores, 1)
        )
        all_labels.extend(labels)
        all_scores.extend(float(score) for score in scores)
        predicted_disordered = int(np.count_nonzero(scores > THRESHOLD))
        protein_rows.append(
            {
                "dataset": dataset,
                "protein_id": record.protein_id,
                "sequence_length": len(record.sequence),
                "source_parts": len(members_by_id[record.protein_id]),
                "predicted_disordered_residues": predicted_disordered,
                "predicted_disordered_fraction": predicted_disordered / float(len(scores)),
                "score_mean": float(np.mean(scores)),
                "score_std": float(np.std(scores)),
                "score_min": float(np.min(scores)),
                "score_max": float(np.max(scores)),
            }
        )

    dataset_metrics = _metrics(all_labels, all_scores)
    summary = {
        "dataset": dataset,
        "proteins": len(records),
        "source_residues": sum(len(record.sequence) for record in records),
        "predicted_residues": len(all_scores),
        "excluded_label_minus1": sum(label == -1 for label in all_labels),
        "evaluated_residues": int(dataset_metrics["n_evaluated"]),
        "positives": sum(label == 1 for label in all_labels),
        "negatives": sum(label == 0 for label in all_labels),
        "threshold": THRESHOLD,
        "tp": dataset_metrics.get("tp"),
        "tn": dataset_metrics.get("tn"),
        "fp": dataset_metrics.get("fp"),
        "fn": dataset_metrics.get("fn"),
        "accuracy": dataset_metrics.get("accuracy"),
        "sn": dataset_metrics.get("sensitivity"),
        "sp": dataset_metrics.get("specificity"),
        "bacc": dataset_metrics.get("bacc"),
        "mcc": dataset_metrics.get("mcc"),
        "auc": dataset_metrics.get("auc"),
        "aupr": dataset_metrics.get("aupr"),
        "fmax": dataset_metrics.get("fmax"),
    }
    return residue_rows, summary, protein_rows


def _output_metadata(path: Path) -> Dict[str, Any]:
    return {"path": path.name, "sha256": _sha256(path), "size_bytes": int(path.stat().st_size)}


def _readme(
    summaries: Sequence[Mapping[str, Any]],
    archive_hash: str,
    source_metadata: Mapping[str, Mapping[str, Any]],
    archive: ArchivePredictions,
) -> str:
    lines = [
        "# DisorderUnetLM reproduced predictions",
        "",
        "This directory was generated by `scripts/package_disorderunetlm_predictions.py`.",
        "It contains the validated residue-level predictions from the supplied archive.",
        "",
        "## Contract",
        "",
        "- The second probability column is interpreted as the class-1 disorder score. The ZIP does not self-describe its class columns; this interpretation is supported by the paper's residue-disorder probability definition and `p > 0.5` decision rule.",
        "- Prediction is `score > 0.5`; a score equal to 0.5 is class 0.",
        "- Labels equal to -1 are retained only in source accounting and excluded from metrics.",
        "- AUPR is average precision and Fmax follows the existing project metric implementation.",
        "- NPY arrays were required to have shape `[L, 2]`, finite probabilities in `[0, 1]`, and row sums within tolerance of 1.",
        "",
        "## Files",
        "",
        "- `sl329_residue_predictions.tsv.gz`, `mxd494_residue_predictions.tsv.gz`, and `disorder723_residue_predictions.tsv.gz` are prediction-only residue tables.",
        "- `protein_summary.tsv.gz` contains prediction-only score and call statistics per protein; it contains no reference-label-derived fields.",
        "- `summary.tsv` and `summary.json` contain dataset-level metrics and confusion counts.",
        "- `manifest.json` records archive/source hashes and the exact archive-member mapping.",
        "",
        "## Dataset summary",
        "",
        "| Dataset | Proteins | Evaluated residues | Sn | Sp | BACC | MCC | AUC | AUPR | Fmax |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {dataset} | {proteins} | {evaluated_residues} | {sn} | {sp} | {bacc} | {mcc} | {auc} | {aupr} | {fmax} |".format(
                dataset=row["dataset"],
                proteins=row["proteins"],
                evaluated_residues=row["evaluated_residues"],
                sn=_format_value(row["sn"]),
                sp=_format_value(row["sp"]),
                bacc=_format_value(row["bacc"]),
                mcc=_format_value(row["mcc"]),
                auc=_format_value(row["auc"]),
                aupr=_format_value(row["aupr"]),
                fmax=_format_value(row["fmax"]),
            )
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "- Archive SHA-256: `{}`".format(archive_hash),
        ]
    )
    for dataset in DATASETS:
        lines.append(
            "- {} source SHA-256: `{}`".format(dataset, source_metadata[dataset]["sha256"])
        )
    multipart_details = []
    for dataset in DATASETS:
        for protein_id, members in archive.members[dataset].items():
            if len(members) > 1:
                lengths = archive.part_lengths[dataset][protein_id]
                multipart_details.append((dataset, protein_id, members, lengths))
    if multipart_details:
        lines.extend(["", "## Multipart predictions", ""])
        for dataset, protein_id, members, lengths in multipart_details:
            lines.append(
                "- `{}/{}`: {} ordered parts with residue lengths {}; concatenated length {}.".format(
                    dataset,
                    protein_id,
                    len(members),
                    ", ".join(str(length) for length in lengths),
                    sum(lengths),
                )
            )
        lines.append("- The paper states a 7168-residue model input limit.")
        dp00072 = next(
            (
                (members, lengths)
                for dataset, protein_id, members, lengths in multipart_details
                if dataset == "SL329" and protein_id == "DP00072"
            ),
            None,
        )
        if dp00072 is not None:
            members, lengths = dp00072
            lines.append(
                "- The supplied `SL329/DP00072` prediction has {} parts with lengths {} ({} total), which this package concatenates in archive-member order: {}. The ZIP does not document how those parts were inferred.".format(
                    len(members),
                    ", ".join(str(length) for length in lengths),
                    sum(lengths),
                    ", ".join("`{}`".format(PurePosixPath(member).stem) for member in members),
                )
            )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "The supplied archive contains predictions but no model, model-version, checkpoint, preprocessing, ensemble, dependency, device, runtime, seed, or class-column metadata. Those details cannot be reconstructed from this artifact, so this package is an evaluation of a supplied reproduction artifact rather than a claim of exact official reproduction.",
            "No paper-value comparison is included because the DisorderUnetLM paper reports no direct SL329, MXD494, or DISORDER723 results; its CAID-2 values must not be transferred to these datasets.",
            "",
        ]
    )
    return "\n".join(lines)


def package_predictions(
    archive_path: Union[Path, str],
    data_dir: Union[Path, str],
    output_dir: Union[Path, str],
    project_root: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Run the complete validation, evaluation, and deterministic packaging flow."""

    archive_path = Path(archive_path).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    project_root = (Path(project_root).expanduser().resolve() if project_root else ROOT.resolve())
    records_by_dataset = load_datasets(data_dir)
    archive = load_archive(archive_path, records_by_dataset)

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []
    all_protein_rows: List[Dict[str, Any]] = []
    output_paths: List[Path] = []
    for dataset in DATASETS:
        residue_rows, summary, protein_rows = _dataset_rows(
            dataset,
            records_by_dataset[dataset],
            archive.scores[dataset],
            archive.members[dataset],
        )
        summaries.append(summary)
        all_protein_rows.extend(protein_rows)
        residue_path = output_dir / (dataset.lower() + "_residue_predictions.tsv.gz")
        _write_deterministic_gzip(residue_path, _tsv_bytes(residue_rows, RESIDUE_COLUMNS))
        output_paths.append(residue_path)

    protein_path = output_dir / "protein_summary.tsv.gz"
    _write_deterministic_gzip(
        protein_path,
        _tsv_bytes(all_protein_rows, PROTEIN_SUMMARY_COLUMNS),
    )
    output_paths.append(protein_path)

    summary_tsv_path = output_dir / "summary.tsv"
    _write_text(summary_tsv_path, _tsv_bytes(summaries, SUMMARY_COLUMNS).decode("utf-8"))
    output_paths.append(summary_tsv_path)

    archive_metadata = {
        "filename": archive_path.name,
        "sha256": _sha256(archive_path),
        "size_bytes": int(archive_path.stat().st_size),
    }
    source_metadata = {
        dataset: {
            **_file_metadata(data_dir / DATASET_FILES[dataset], project_root),
            "proteins": len(records_by_dataset[dataset]),
            "source_residues": sum(len(record.sequence) for record in records_by_dataset[dataset]),
        }
        for dataset in DATASETS
    }
    script_metadata = _file_metadata(Path(__file__), project_root)
    caveat = (
        "The supplied archive contains predictions but no model, model-version, "
        "checkpoint, preprocessing, ensemble, dependency, device, runtime, seed, "
        "or class-column metadata."
    )
    class_caveat = (
        "Probability column 1 is interpreted as disorder based on the paper's residue-probability "
        "definition and p > 0.5 rule; the ZIP does not label its class columns."
    )
    paper_caveat = (
        "The DisorderUnetLM paper reports no direct SL329, MXD494, or DISORDER723 results; "
        "CAID-2 values are not comparable dataset results."
    )
    summary_json = {
        "metadata": {
            "package_version": PACKAGE_VERSION,
            "threshold": THRESHOLD,
            "score_definition": "probability_array[:, 1], interpreted as class-1 disorder score",
            "score_interpretation_evidence": "DisorderUnetLM paper defines yi as residue disorder probability and applies p > 0.5",
            "archive_class_column_metadata_present": False,
            "prediction_rule": "score > 0.5",
            "metric_label_policy": "labels 0 and 1 are evaluated; label -1 is excluded",
            "archive_sha256": archive_metadata["sha256"],
            "caveat": caveat,
            "paper_comparison": "none; the paper reports no direct results for these three datasets",
        },
        "datasets": summaries,
    }
    summary_json_path = output_dir / "summary.json"
    _write_json(summary_json_path, summary_json)
    output_paths.append(summary_json_path)

    readme_path = output_dir / "README.md"
    _write_text(
        readme_path,
        _readme(summaries, archive_metadata["sha256"], source_metadata, archive),
    )
    output_paths.append(readme_path)

    manifest: Dict[str, Any] = {
        "package": {"name": "DisorderUnetLM reproduced predictions", "version": PACKAGE_VERSION},
        "generation": {
            "command_template": (
                "python scripts/package_disorderunetlm_predictions.py "
                "--archive <prediction-archive.zip> --data-dir data "
                "--output-dir results/reproduction/disorderunetlm"
            ),
            "script": script_metadata,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
        "archive": archive_metadata,
        "sources": source_metadata,
        "protocol": {
            "datasets": list(DATASETS),
            "expected_proteins": EXPECTED_PROTEINS,
            "npy_dtype": "float32",
            "npy_shape": "[L, 2]",
            "probability_range": [0.0, 1.0],
            "row_sum_atol": ROW_SUM_ATOL,
            "row_sum_rtol": ROW_SUM_RTOL,
            "score_column": 1,
            "score_column_interpretation": "disorder class (evidence-supported; not self-described by ZIP)",
            "archive_class_column_metadata_present": False,
            "prediction_rule": "score > 0.5",
            "excluded_label": -1,
            "residue_columns": list(RESIDUE_COLUMNS),
            "summary_columns": list(SUMMARY_COLUMNS),
            "protein_summary_columns": list(PROTEIN_SUMMARY_COLUMNS),
        },
        "caveats": [caveat, class_caveat, paper_caveat],
        "datasets": {},
        "outputs": {},
    }
    for dataset in DATASETS:
        mapping = archive.members[dataset]
        manifest["datasets"][dataset] = {
            "proteins": len(records_by_dataset[dataset]),
            "archive_member_count": sum(len(names) for names in mapping.values()),
            "multipart_proteins": {
                protein_id: {
                    "members": list(names),
                    "part_lengths": list(archive.part_lengths[dataset][protein_id]),
                    "concatenated_length": sum(archive.part_lengths[dataset][protein_id]),
                }
                for protein_id, names in mapping.items()
                if len(names) > 1
            },
            "archive_members": {protein_id: list(names) for protein_id, names in mapping.items()},
            "summary": summaries[DATASETS.index(dataset)],
        }
    for path in output_paths:
        manifest["outputs"][path.name] = _output_metadata(path)
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    return {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "summary": summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        "--zip",
        dest="archive",
        required=True,
        type=Path,
        help="Supplied DisorderUnetLM .zip prediction archive.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Directory containing SL329_test.fasta, MXD494_test.fasta, and DISORDER723_test.fasta.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory in which the deterministic package will be written.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
        help="Root used to render relative provenance paths in manifest.json.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = package_predictions(
            archive_path=args.archive,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    except PackagingError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
