#!/usr/bin/env python3
"""Audit structure/function resources for P4 feasibility.

The goal is to decide whether structure-aware or function-aware extensions can
be tested with the local project assets without introducing obvious leakage.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DATASETS = (
    ("DM3000_Train", "train", Path("data/DM3000_Train.fasta")),
    ("DM1229_Validation", "validation", Path("data/DM1229_Validation.fasta")),
    ("SL329", "test", Path("data/SL329_test.fasta")),
    ("MXD494", "test", Path("data/MXD494_test.fasta")),
    ("DISORDER723", "test", Path("data/DISORDER723_test.fasta")),
)

PDB_CHAIN_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}[A-Za-z0-9]$")
PDB_VARIANT_RE = re.compile(r"^([0-9][A-Za-z0-9]{3})([-_].+)$")
PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
DISPROT_RE = re.compile(r"^DP\d+", re.IGNORECASE)
VALIDATION_RE = re.compile(r"^DM_validata\d+$", re.IGNORECASE)

STRUCTURE_KEYWORDS = (
    "alpha",
    "alphafold",
    "esmfold",
    "fold",
    "plddt",
    "pae",
    "structure",
    "pdb",
    "cif",
    "dssp",
    "sasa",
)
STRUCTURE_SUFFIXES = (
    ".pdb",
    ".ent",
    ".cif",
    ".mmcif",
    ".pdb.gz",
    ".ent.gz",
    ".cif.gz",
    ".mmcif.gz",
)
STRUCTURE_METADATA_SUFFIXES = (".json", ".npz")

FUNCTION_KEYWORDS = (
    "function",
    "binding",
    "morf",
    "linker",
    "disprot",
    "caid",
    "protein-binding",
    "dna-binding",
    "rna-binding",
    "flexible",
)
FUNCTION_SUFFIXES = (".tsv", ".csv", ".json", ".npz", ".txt", ".fasta", ".fa", ".pkl")

LITERATURE_TERMS = {
    "AlphaFold": re.compile(r"\bAlphaFold\b", re.IGNORECASE),
    "ESMFold": re.compile(r"\bESMFold\b", re.IGNORECASE),
    "pLDDT": re.compile(r"\bpLDDT\b", re.IGNORECASE),
    "PAE": re.compile(r"\bPAE\b|\bpredicted aligned error\b", re.IGNORECASE),
    "DSSP": re.compile(r"\bDSSP\b", re.IGNORECASE),
    "SASA": re.compile(r"\bSASA\b|\bsolvent accessible surface area\b", re.IGNORECASE),
    "function": re.compile(r"\bfunctions?\b|\bfunctional\b", re.IGNORECASE),
    "binding": re.compile(r"\bbinding\b|\bbind\b|\bbinds\b", re.IGNORECASE),
    "MoRF": re.compile(r"\bMoRFs?\b|\bmolecular recognition features?\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class FastaRecord:
    dataset: str
    split_role: str
    path: Path
    record_id: str
    header: str
    sequence: str
    labels: tuple[int, ...]


@dataclass(frozen=True)
class IdInfo:
    id_type: str
    pdb_id: str
    chain_or_fragment: str
    leakage_risk: str


@dataclass(frozen=True)
class ResourceFile:
    path: Path
    relative_path: str
    category: str
    size_bytes: int
    keyword_hits: tuple[str, ...]
    identifiers: frozenset[str]


def parse_labeled_fasta(path: Path, dataset: str, split_role: str) -> list[FastaRecord]:
    records: list[FastaRecord] = []
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
        labels_value = ast.literal_eval(label_text)
        if not isinstance(labels_value, list):
            raise ValueError(f"{path}: labels are not a list for {header}")
        labels = tuple(int(label) for label in labels_value)
        if any(label not in (-1, 0, 1) for label in labels):
            raise ValueError(f"{path}: labels must be -1, 0, or 1 for {header}")
        if len(sequence) != len(labels):
            raise ValueError(
                f"{path}: sequence/label length mismatch for {header}: "
                f"{len(sequence)} != {len(labels)}"
            )
        record_id = header.split()[0]
        records.append(
            FastaRecord(
                dataset=dataset,
                split_role=split_role,
                path=path,
                record_id=record_id,
                header=header,
                sequence=sequence,
                labels=labels,
            )
        )
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


def classify_id(record_id: str) -> IdInfo:
    if DISPROT_RE.match(record_id):
        return IdInfo("disprot", "", "", "medium: DisProt labels may include curated structural evidence")
    if VALIDATION_RE.match(record_id):
        return IdInfo("custom_validation", "", "", "unknown: no direct external database mapping in local ID")
    if PDB_CHAIN_RE.match(record_id):
        pdb_id = record_id[:4].upper()
        chain = record_id[4:]
        return IdInfo(
            "pdb_chain",
            pdb_id,
            chain,
            "high: direct solved-structure features can encode benchmark labels",
        )
    variant_match = PDB_VARIANT_RE.match(record_id)
    if variant_match:
        return IdInfo(
            "pdb_variant",
            variant_match.group(1).upper(),
            variant_match.group(2).lstrip("-_"),
            "high: PDB-derived fragment/domain ID can encode benchmark labels",
        )
    if PDB_ID_RE.match(record_id):
        return IdInfo(
            "pdb_id",
            record_id.upper(),
            "",
            "high: direct solved-structure features can encode benchmark labels",
        )
    return IdInfo("other", "", "", "unknown: mapping requires external accession resolution")


def label_counts(labels: tuple[int, ...]) -> dict[str, int]:
    counts = Counter(labels)
    return {
        "residues": len(labels),
        "known_residues": counts[0] + counts[1],
        "ordered_residues": counts[0],
        "disorder_residues": counts[1],
        "unknown_residues": counts[-1],
    }


def tsv_value(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.6f}"
    return str(value)


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: tsv_value(row.get(field, "")) for field in fieldnames})


def strip_known_suffixes(name: str) -> str:
    lower = name.lower()
    for suffix in sorted(STRUCTURE_SUFFIXES + STRUCTURE_METADATA_SUFFIXES + FUNCTION_SUFFIXES, key=len, reverse=True):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def identifiers_from_file_name(path: Path) -> frozenset[str]:
    stem = strip_known_suffixes(path.name).lower()
    tokens = {token for token in re.split(r"[^a-z0-9]+", stem) if token}
    tokens.add(stem)
    compact = re.sub(r"[^a-z0-9]", "", stem)
    if compact:
        tokens.add(compact)
    return frozenset(tokens)


def has_suffix(path: Path, suffixes: tuple[str, ...]) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in suffixes)


def skip_path(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part in {".git", "__pycache__", ".ipynb_checkpoints"} for part in parts)


def scan_structure_files(root: Path) -> list[ResourceFile]:
    resources: list[ResourceFile] = []
    for path in root.rglob("*"):
        if not path.is_file() or skip_path(path, root):
            continue
        path_lower = str(path.relative_to(root)).lower()
        keyword_hits = tuple(keyword for keyword in STRUCTURE_KEYWORDS if keyword in path_lower)
        is_structure = has_suffix(path, STRUCTURE_SUFFIXES)
        is_metadata = has_suffix(path, STRUCTURE_METADATA_SUFFIXES) and bool(keyword_hits)
        if not (is_structure or is_metadata):
            continue
        category = "structure_coordinate" if is_structure else "structure_metadata"
        resources.append(
            ResourceFile(
                path=path,
                relative_path=str(path.relative_to(root)),
                category=category,
                size_bytes=path.stat().st_size,
                keyword_hits=keyword_hits,
                identifiers=identifiers_from_file_name(path),
            )
        )
    return sorted(resources, key=lambda item: item.relative_path)


def scan_function_files(root: Path) -> list[ResourceFile]:
    resources: list[ResourceFile] = []
    excluded_roots = {"references", "reports", "scripts"}
    for path in root.rglob("*"):
        if not path.is_file() or skip_path(path, root):
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in excluded_roots:
            continue
        path_lower = str(relative).lower()
        keyword_hits = tuple(keyword for keyword in FUNCTION_KEYWORDS if keyword in path_lower)
        if not (keyword_hits and has_suffix(path, FUNCTION_SUFFIXES)):
            continue
        resources.append(
            ResourceFile(
                path=path,
                relative_path=str(relative),
                category="function_annotation_candidate",
                size_bytes=path.stat().st_size,
                keyword_hits=keyword_hits,
                identifiers=identifiers_from_file_name(path),
            )
        )
    return sorted(resources, key=lambda item: item.relative_path)


def build_resource_match_index(resources: list[ResourceFile]) -> dict[str, list[ResourceFile]]:
    index: dict[str, list[ResourceFile]] = defaultdict(list)
    for resource in resources:
        for identifier in resource.identifiers:
            index[identifier].append(resource)
    return index


def match_resources(record_id: str, id_info: IdInfo, match_index: dict[str, list[ResourceFile]]) -> list[ResourceFile]:
    keys = {record_id.lower()}
    if id_info.pdb_id:
        keys.add(id_info.pdb_id.lower())
    if id_info.pdb_id and id_info.chain_or_fragment and len(id_info.chain_or_fragment) == 1:
        keys.add(f"{id_info.pdb_id}{id_info.chain_or_fragment}".lower())
    matches: dict[str, ResourceFile] = {}
    for key in keys:
        for resource in match_index.get(key, []):
            matches[resource.relative_path] = resource
    return [matches[key] for key in sorted(matches)]


def summarize_ids(records: list[FastaRecord]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    mapping_rows: list[dict[str, object]] = []
    for record in records:
        info = classify_id(record.record_id)
        counts = label_counts(record.labels)
        key = (record.dataset, record.split_role, info.id_type)
        if key not in grouped:
            grouped[key] = {
                "dataset": record.dataset,
                "split_role": record.split_role,
                "id_type": info.id_type,
                "proteins": 0,
                "residues": 0,
                "known_residues": 0,
                "ordered_residues": 0,
                "disorder_residues": 0,
                "unknown_residues": 0,
                "unique_pdb_ids": set(),
            }
        row = grouped[key]
        row["proteins"] = int(row["proteins"]) + 1
        for field in ("residues", "known_residues", "ordered_residues", "disorder_residues", "unknown_residues"):
            row[field] = int(row[field]) + counts[field]
        if info.pdb_id:
            row["unique_pdb_ids"].add(info.pdb_id)

        mapping_rows.append(
            {
                "dataset": record.dataset,
                "split_role": record.split_role,
                "record_id": record.record_id,
                "id_type": info.id_type,
                "length": counts["residues"],
                "known_residues": counts["known_residues"],
                "ordered_residues": counts["ordered_residues"],
                "disorder_residues": counts["disorder_residues"],
                "unknown_residues": counts["unknown_residues"],
                "pdb_id": info.pdb_id or "NA",
                "chain_or_fragment": info.chain_or_fragment or "NA",
                "leakage_risk": info.leakage_risk,
            }
        )

    total_keys: dict[tuple[str, str], dict[str, object]] = {}
    for row in grouped.values():
        key = (str(row["dataset"]), str(row["split_role"]))
        if key not in total_keys:
            total_keys[key] = {
                "dataset": row["dataset"],
                "split_role": row["split_role"],
                "id_type": "ALL",
                "proteins": 0,
                "residues": 0,
                "known_residues": 0,
                "ordered_residues": 0,
                "disorder_residues": 0,
                "unknown_residues": 0,
                "unique_pdb_ids": set(),
            }
        total = total_keys[key]
        for field in ("proteins", "residues", "known_residues", "ordered_residues", "disorder_residues", "unknown_residues"):
            total[field] = int(total[field]) + int(row[field])
        total["unique_pdb_ids"].update(row["unique_pdb_ids"])

    summary_rows: list[dict[str, object]] = []
    for row in list(total_keys.values()) + list(grouped.values()):
        known = int(row["known_residues"])
        proteins = int(row["proteins"])
        pdb_count = len(row["unique_pdb_ids"])
        summary_rows.append(
            {
                **{key: value for key, value in row.items() if key != "unique_pdb_ids"},
                "disorder_fraction_known": (int(row["disorder_residues"]) / known) if known else math.nan,
                "unknown_fraction_all": (int(row["unknown_residues"]) / int(row["residues"])) if int(row["residues"]) else math.nan,
                "unique_pdb_ids": pdb_count,
                "pdb_ids_per_protein": (pdb_count / proteins) if proteins else math.nan,
            }
        )
    return sorted(summary_rows, key=lambda item: (str(item["dataset"]), str(item["id_type"]))), mapping_rows


def add_structure_matches(
    mapping_rows: list[dict[str, object]],
    structure_resources: list[ResourceFile],
) -> list[dict[str, object]]:
    match_index = build_resource_match_index(structure_resources)
    enriched_rows: list[dict[str, object]] = []
    for row in mapping_rows:
        info = IdInfo(
            id_type=str(row["id_type"]),
            pdb_id="" if row["pdb_id"] == "NA" else str(row["pdb_id"]),
            chain_or_fragment="" if row["chain_or_fragment"] == "NA" else str(row["chain_or_fragment"]),
            leakage_risk=str(row["leakage_risk"]),
        )
        matches = match_resources(str(row["record_id"]), info, match_index)
        enriched_rows.append(
            {
                **row,
                "local_structure_match_count": len(matches),
                "local_structure_matches": ";".join(match.relative_path for match in matches[:10]) if matches else "NA",
            }
        )
    return enriched_rows


def summarize_structure_coverage(mapping_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for row in mapping_rows:
        key = (str(row["dataset"]), str(row["split_role"]))
        if key not in grouped:
            grouped[key] = {
                "dataset": row["dataset"],
                "split_role": row["split_role"],
                "proteins": 0,
                "pdb_mappable_proteins": 0,
                "unique_pdb_ids": set(),
                "local_structure_matched_proteins": 0,
                "high_leakage_risk_proteins": 0,
            }
        group = grouped[key]
        group["proteins"] = int(group["proteins"]) + 1
        if row["pdb_id"] != "NA":
            group["pdb_mappable_proteins"] = int(group["pdb_mappable_proteins"]) + 1
            group["unique_pdb_ids"].add(row["pdb_id"])
        if int(row["local_structure_match_count"]) > 0:
            group["local_structure_matched_proteins"] = int(group["local_structure_matched_proteins"]) + 1
        if str(row["leakage_risk"]).startswith("high"):
            group["high_leakage_risk_proteins"] = int(group["high_leakage_risk_proteins"]) + 1

    rows: list[dict[str, object]] = []
    for group in grouped.values():
        proteins = int(group["proteins"])
        rows.append(
            {
                "dataset": group["dataset"],
                "split_role": group["split_role"],
                "proteins": proteins,
                "pdb_mappable_proteins": group["pdb_mappable_proteins"],
                "pdb_mappable_fraction": int(group["pdb_mappable_proteins"]) / proteins if proteins else math.nan,
                "unique_pdb_ids": len(group["unique_pdb_ids"]),
                "local_structure_matched_proteins": group["local_structure_matched_proteins"],
                "local_structure_coverage": int(group["local_structure_matched_proteins"]) / proteins if proteins else math.nan,
                "high_leakage_risk_proteins": group["high_leakage_risk_proteins"],
                "high_leakage_risk_fraction": int(group["high_leakage_risk_proteins"]) / proteins if proteins else math.nan,
            }
        )
    return sorted(rows, key=lambda item: str(item["dataset"]))


def summarize_resource_files(
    resources: list[ResourceFile],
    mapping_rows: list[dict[str, object]],
    resource_scope: str,
) -> list[dict[str, object]]:
    id_match_counts: Counter[str] = Counter()
    pdb_match_counts: Counter[str] = Counter()
    for row in mapping_rows:
        record_id = str(row["record_id"]).lower()
        pdb_id = "" if row["pdb_id"] == "NA" else str(row["pdb_id"]).lower()
        for resource in resources:
            if record_id in resource.identifiers:
                id_match_counts[resource.relative_path] += 1
            if pdb_id and pdb_id in resource.identifiers:
                pdb_match_counts[resource.relative_path] += 1

    if not resources:
        return [
            {
                "resource_scope": resource_scope,
                "resource_category": "none_detected",
                "path": "NA",
                "file_name": "NA",
                "size_bytes": 0,
                "keyword_hits": "NA",
                "direct_record_id_match_count": 0,
                "pdb_id_match_count": 0,
                "notes": "No candidate local resource file detected by extension/name scan.",
            }
        ]

    rows: list[dict[str, object]] = []
    for resource in resources:
        rows.append(
            {
                "resource_scope": resource_scope,
                "resource_category": resource.category,
                "path": resource.relative_path,
                "file_name": resource.path.name,
                "size_bytes": resource.size_bytes,
                "keyword_hits": ",".join(resource.keyword_hits) if resource.keyword_hits else "NA",
                "direct_record_id_match_count": id_match_counts[resource.relative_path],
                "pdb_id_match_count": pdb_match_counts[resource.relative_path],
                "notes": "candidate file; inspect semantics before using as feature",
            }
        )
    return rows


def scan_literature_terms(root: Path) -> list[dict[str, object]]:
    reference_paths = sorted((root / "references" / "pdf_texts").glob("*.txt"))
    reference_paths.extend(sorted((root / "references").glob("*.xml")))
    rows: list[dict[str, object]] = []
    for path in reference_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        counts = {term: len(pattern.findall(text)) for term, pattern in LITERATURE_TERMS.items()}
        total = sum(counts.values())
        rows.append(
            {
                "source_file": str(path.relative_to(root)),
                **counts,
                "total_hits": total,
            }
        )
    return rows


def make_decision_rows(structure_coverage_rows: list[dict[str, object]], function_resources: list[ResourceFile]) -> list[dict[str, object]]:
    max_structure_coverage = max(float(row["local_structure_coverage"]) for row in structure_coverage_rows)
    max_leakage_fraction = max(float(row["high_leakage_risk_fraction"]) for row in structure_coverage_rows)
    has_function_labels = len(function_resources) > 0

    structure_decision = (
        "no_go_for_training"
        if max_structure_coverage == 0.0
        else "pilot_only_with_leakage_controls"
    )
    function_decision = "pilot_only_after_label_collection" if not has_function_labels else "audit_labels_then_pilot"

    return [
        {
            "extension": "structure_features",
            "current_status": "no local structure feature files" if max_structure_coverage == 0.0 else "candidate local files detected",
            "coverage_signal": f"max local structure coverage={max_structure_coverage:.6f}",
            "leakage_signal": f"max high-risk PDB-derived ID fraction={max_leakage_fraction:.6f}",
            "decision": structure_decision,
            "recommended_action": "collect UniProt-mapped AlphaFold/ESMFold confidence features first; avoid using solved PDB missing residues as direct inputs",
        },
        {
            "extension": "function_auxiliary_heads",
            "current_status": "no local function-label tables" if not has_function_labels else "candidate local function files detected",
            "coverage_signal": f"candidate local function resources={len(function_resources)}",
            "leakage_signal": "function labels require source/date/split audit before multi-task training",
            "decision": function_decision,
            "recommended_action": "collect DisProt/CAID binding-linker/MoRF labels and keep them out of test-tuned threshold/model selection",
        },
    ]


def render_report(
    root: Path,
    id_summary_rows: list[dict[str, object]],
    structure_coverage_rows: list[dict[str, object]],
    structure_resources: list[ResourceFile],
    function_resources: list[ResourceFile],
    literature_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
) -> str:
    all_rows = [row for row in id_summary_rows if row["id_type"] == "ALL"]
    id_rows = [row for row in id_summary_rows if row["id_type"] != "ALL"]

    def md_table(rows: list[dict[str, object]], columns: list[str]) -> str:
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(tsv_value(row.get(column, "")) for column in columns) + " |")
        return "\n".join(lines)

    term_totals = {
        term: sum(int(row[term]) for row in literature_rows)
        for term in LITERATURE_TERMS
    }
    top_literature = sorted(literature_rows, key=lambda row: int(row["total_hits"]), reverse=True)[:8]
    has_structure = len(structure_resources) > 0
    has_function = len(function_resources) > 0

    report = f"""# P4 Structure/Function Extension Audit

Date: 2026-07-31

## Scope

P4 was executed as a local feasibility and leakage-risk audit before any structure-aware or function-aware model training.

Inputs:

- `data/DM3000_Train.fasta`
- `data/DM1229_Validation.fasta`
- `data/SL329_test.fasta`
- `data/MXD494_test.fasta`
- `data/DISORDER723_test.fasta`
- local `references/` text/XML files for literature support only

Generated assets:

- `scripts/audit_structure_function_resources.py`
- `results/structure/P4_DATASET_ID_TYPE_SUMMARY.tsv`
- `results/structure/P4_POTENTIAL_PDB_MAPPING.tsv`
- `results/structure/P4_STRUCTURE_COVERAGE_BY_DATASET.tsv`
- `results/structure/P4_STRUCTURE_RESOURCE_AUDIT.tsv`
- `results/structure/P4_FUNCTION_RESOURCE_AUDIT.tsv`
- `results/structure/P4_LITERATURE_TERM_COUNTS.tsv`
- `results/structure/P4_DECISION_MATRIX.tsv`

## Dataset ID Audit

Overall dataset sizes:

{md_table(all_rows, ["dataset", "split_role", "proteins", "residues", "known_residues", "disorder_residues", "unknown_residues", "disorder_fraction_known"])}

ID-type composition:

{md_table(id_rows, ["dataset", "id_type", "proteins", "unique_pdb_ids", "residues", "unknown_residues", "disorder_fraction_known"])}

Structure mapping/leakage signals:

{md_table(structure_coverage_rows, ["dataset", "proteins", "pdb_mappable_proteins", "pdb_mappable_fraction", "unique_pdb_ids", "local_structure_matched_proteins", "local_structure_coverage", "high_leakage_risk_fraction"])}

Interpretation:

- DM3000, MXD494, and DISORDER723 contain many PDB-chain or PDB-fragment style IDs. These are useful for tracing benchmark provenance, but they make experimentally solved structure features risky.
- SL329 uses DisProt-style IDs, so structure use would require a separate accession/mapping step and a dated source audit.
- DM1229 validation uses project-local IDs and has no direct external mapping in the FASTA headers.

## Local Structure Resource Audit

Candidate local structure resources detected: {len(structure_resources)}.

Decision from local scan: {"structure files were detected and require semantic inspection before training" if has_structure else "no local PDB/mmCIF/AlphaFold/ESMFold/pLDDT/PAE resource files were detected"}.

Because local structure coverage is currently zero, P4 should not proceed directly to structure-enhanced training. A structure-aware model would first need a controlled resource acquisition step.

## Local Function Resource Audit

Candidate local function-label resources detected outside `references/`, `reports/`, and `scripts/`: {len(function_resources)}.

Decision from local scan: {"candidate files require label-source inspection before multi-task training" if has_function else "no local DisProt/CAID/MoRF/binding/linker label table was detected"}.

This means function auxiliary heads are not ready for training from local files alone. The literature supports the biological relevance of disordered function prediction, but labels must be collected and split-audited first.

## Literature Support Signals

Term totals across local reference text/XML files:

{md_table([{**term_totals}], list(LITERATURE_TERMS.keys()))}

Top reference files by P4-relevant term hits:

{md_table(top_literature, ["source_file", "AlphaFold", "ESMFold", "pLDDT", "PAE", "DSSP", "SASA", "function", "binding", "MoRF", "total_hits"])}

Interpretation:

- The local references strongly support function-aware IDR prediction, especially binding, MoRF, and flexible-linker directions.
- The current local literature set contains much weaker support for direct AlphaFold/pLDDT/PAE feature use in this project than for PLM plus function-aware modeling.

## P4 Decision

{md_table(decision_rows, ["extension", "current_status", "coverage_signal", "leakage_signal", "decision", "recommended_action"])}

Main decision:

1. Do not train a structure-enhanced main model yet, because local structure feature coverage is zero.
2. Do not use solved PDB missing-residue or coordinate-derived features as ordinary inputs for DM3000/MXD494/DISORDER723 without a leakage-control design.
3. Treat AlphaFold/ESMFold confidence features as a future optional branch, preferably through UniProt mapping and sequence-alignment verification.
4. Treat function prediction as a more publication-relevant P4 extension, but only after collecting explicit DisProt/CAID/MoRF/binding/linker labels.

## Recommended Next P4 Actions

P4.1 Lock the current main model as the sequence-only calibrated model: `RegionAwareTCN + aux 3-seed ensemble + Platt`.

P4.2 Build an accession-mapping table for all records:

- PDB-chain IDs: map PDB code/chain to UniProt where possible.
- DisProt IDs: map DisProt entry to UniProt accession and evidence date.
- Project-local validation IDs: recover source accession if available from upstream metadata.

P4.3 If structure is still desired, collect AlphaFold/ESMFold pLDDT only for UniProt-mappable proteins, then report coverage and run a sequence-only versus sequence+pLDDT ablation on the same covered subset.

P4.4 If function extension is prioritized, collect residue-level labels for protein-binding IDRs, DNA/RNA-binding IDRs, MoRFs, and flexible linkers, then train auxiliary heads without changing the disorder-threshold selection protocol.

P4.5 For a high-level journal manuscript, keep structure/function as optional extension evidence unless coverage becomes high and leakage-controlled. The current strongest manuscript core remains: low-leakage NR25 evaluation, region-aware modeling, calibration, and uncertainty/error enrichment.
"""
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root")
    parser.add_argument("--out-dir", type=Path, default=Path("results/structure"), help="Output table directory")
    parser.add_argument("--report", type=Path, default=Path("reports/P4_STRUCTURE_FUNCTION_AUDIT.md"), help="Markdown report path")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    report_path = args.report if args.report.is_absolute() else root / args.report

    records: list[FastaRecord] = []
    for dataset, split_role, relative_path in DATASETS:
        records.extend(parse_labeled_fasta(root / relative_path, dataset, split_role))

    id_summary_rows, mapping_rows = summarize_ids(records)
    structure_resources = scan_structure_files(root)
    function_resources = scan_function_files(root)
    enriched_mapping_rows = add_structure_matches(mapping_rows, structure_resources)
    structure_coverage_rows = summarize_structure_coverage(enriched_mapping_rows)
    literature_rows = scan_literature_terms(root)
    decision_rows = make_decision_rows(structure_coverage_rows, function_resources)

    id_summary_fields = [
        "dataset",
        "split_role",
        "id_type",
        "proteins",
        "residues",
        "known_residues",
        "ordered_residues",
        "disorder_residues",
        "unknown_residues",
        "disorder_fraction_known",
        "unknown_fraction_all",
        "unique_pdb_ids",
        "pdb_ids_per_protein",
    ]
    mapping_fields = [
        "dataset",
        "split_role",
        "record_id",
        "id_type",
        "length",
        "known_residues",
        "ordered_residues",
        "disorder_residues",
        "unknown_residues",
        "pdb_id",
        "chain_or_fragment",
        "leakage_risk",
        "local_structure_match_count",
        "local_structure_matches",
    ]
    structure_coverage_fields = [
        "dataset",
        "split_role",
        "proteins",
        "pdb_mappable_proteins",
        "pdb_mappable_fraction",
        "unique_pdb_ids",
        "local_structure_matched_proteins",
        "local_structure_coverage",
        "high_leakage_risk_proteins",
        "high_leakage_risk_fraction",
    ]
    resource_fields = [
        "resource_scope",
        "resource_category",
        "path",
        "file_name",
        "size_bytes",
        "keyword_hits",
        "direct_record_id_match_count",
        "pdb_id_match_count",
        "notes",
    ]
    literature_fields = ["source_file", *LITERATURE_TERMS.keys(), "total_hits"]
    decision_fields = [
        "extension",
        "current_status",
        "coverage_signal",
        "leakage_signal",
        "decision",
        "recommended_action",
    ]

    write_tsv(out_dir / "P4_DATASET_ID_TYPE_SUMMARY.tsv", id_summary_rows, id_summary_fields)
    write_tsv(out_dir / "P4_POTENTIAL_PDB_MAPPING.tsv", enriched_mapping_rows, mapping_fields)
    write_tsv(out_dir / "P4_STRUCTURE_COVERAGE_BY_DATASET.tsv", structure_coverage_rows, structure_coverage_fields)
    write_tsv(
        out_dir / "P4_STRUCTURE_RESOURCE_AUDIT.tsv",
        summarize_resource_files(structure_resources, enriched_mapping_rows, "local_structure"),
        resource_fields,
    )
    write_tsv(
        out_dir / "P4_FUNCTION_RESOURCE_AUDIT.tsv",
        summarize_resource_files(function_resources, enriched_mapping_rows, "local_function"),
        resource_fields,
    )
    write_tsv(out_dir / "P4_LITERATURE_TERM_COUNTS.tsv", literature_rows, literature_fields)
    write_tsv(out_dir / "P4_DECISION_MATRIX.tsv", decision_rows, decision_fields)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(
            root=root,
            id_summary_rows=id_summary_rows,
            structure_coverage_rows=structure_coverage_rows,
            structure_resources=structure_resources,
            function_resources=function_resources,
            literature_rows=literature_rows,
            decision_rows=decision_rows,
        ),
        encoding="utf-8",
    )

    print(f"records={len(records)}")
    print(f"structure_resources={len(structure_resources)}")
    print(f"function_resources={len(function_resources)}")
    print(f"tables={out_dir}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
