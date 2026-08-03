#!/usr/bin/env python3
"""Build DM3000 training sets filtered against each test set at 25% identity."""

from __future__ import annotations

import argparse
import ast
import csv
import shutil
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path("/data8T/IDPs_DM3000Train")
DATA = ROOT / "data"
MMSEQS = Path("/data8T/IDP_function/tools/mmseqs2/bin/mmseqs")

TRAIN_FASTA = DATA / "DM3000_Train.fasta"
TEST_FASTAS = {
    "SL329": DATA / "SL329_test.fasta",
    "MXD494": DATA / "MXD494_test.fasta",
    "DISORDER723": DATA / "DISORDER723_test.fasta",
}


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


def write_sequence_fasta(records: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(f">{record['id']}\n")
            handle.write(f"{record['sequence']}\n")


def write_labeled_fasta(records: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(f">{record['id']}\n")
            handle.write(f"{record['sequence']}\n")
            handle.write(f"{record['labels']}\n")


def identity_percent(value: str) -> float:
    number = float(value)
    return number * 100.0 if number <= 1.0 else number


def summarize(records: list[dict[str, object]]) -> dict[str, int]:
    counts: Counter[int] = Counter()
    residues = 0
    for record in records:
        labels = record["labels"]
        sequence = record["sequence"]
        assert isinstance(labels, list)
        assert isinstance(sequence, str)
        residues += len(sequence)
        counts.update(labels)
    return {
        "records": len(records),
        "residues": residues,
        "label_1_disordered": counts[1],
        "label_0_ordered": counts[0],
        "label_-1_unknown": counts[-1],
    }


def run_mmseqs(query_fasta: Path, target_fasta: Path, hits_path: Path, tmp_dir: Path, threads: int) -> None:
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    hits_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(MMSEQS),
        "easy-search",
        str(query_fasta),
        str(target_fasta),
        str(hits_path),
        str(tmp_dir),
        "--search-type",
        "1",
        "--min-seq-id",
        "0.25",
        "-s",
        "7.5",
        "--max-seqs",
        "100000",
        "--format-output",
        "query,target,pident,fident,nident,alnlen,qlen,tlen,qcov,tcov,evalue,bits",
        "--threads",
        str(threads),
        "-v",
        "2",
    ]
    subprocess.run(command, check=True)


def collect_removed_ids(hits_path: Path, threshold: float) -> tuple[set[str], dict[str, tuple[float, str]]]:
    removed: set[str] = set()
    best_hit: dict[str, tuple[float, str]] = {}
    with hits_path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 12:
                continue
            query_id, target_id = row[0], row[1]
            pident = identity_percent(row[2])
            if pident <= threshold:
                continue
            removed.add(target_id)
            previous = best_hit.get(target_id)
            if previous is None or pident > previous[0]:
                best_hit[target_id] = (pident, query_id)
    return removed, best_hit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=25.0)
    args = parser.parse_args()

    if not MMSEQS.exists():
        raise FileNotFoundError(MMSEQS)

    out_dir = DATA / "nr25_by_test"
    seq_dir = out_dir / "_sequence_fastas"
    hits_dir = out_dir / "mmseqs_hits"
    removed_dir = out_dir / "removed_ids"
    tmp_root = out_dir / "_tmp"
    for directory in (seq_dir, hits_dir, removed_dir, tmp_root):
        directory.mkdir(parents=True, exist_ok=True)

    train_records = parse_labeled_fasta(TRAIN_FASTA)
    train_by_id = {str(record["id"]): record for record in train_records}
    if len(train_by_id) != len(train_records):
        raise ValueError("DM3000 training set contains duplicate IDs")

    train_seq_fasta = seq_dir / "DM3000_Train.sequences.fasta"
    write_sequence_fasta(train_records, train_seq_fasta)

    summary_rows: list[dict[str, object]] = []
    train_summary = summarize(train_records)

    for test_name, test_path in TEST_FASTAS.items():
        test_records = parse_labeled_fasta(test_path)
        test_seq_fasta = seq_dir / f"{test_name}_test.sequences.fasta"
        write_sequence_fasta(test_records, test_seq_fasta)

        hits_path = hits_dir / f"{test_name}_vs_DM3000_minseqid25.m8"
        run_mmseqs(
            query_fasta=test_seq_fasta,
            target_fasta=train_seq_fasta,
            hits_path=hits_path,
            tmp_dir=tmp_root / test_name,
            threads=args.threads,
        )

        removed_ids, best_hit = collect_removed_ids(hits_path, args.threshold)
        kept_records = [record for record in train_records if str(record["id"]) not in removed_ids]

        out_fasta = out_dir / f"DM3000_Train_nr25_vs_{test_name}.fasta"
        write_labeled_fasta(kept_records, out_fasta)

        removed_path = removed_dir / f"DM3000_removed_for_{test_name}.tsv"
        with removed_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("train_id\tbest_pident\tmatched_test_id\n")
            for train_id in sorted(removed_ids):
                pident, query_id = best_hit[train_id]
                handle.write(f"{train_id}\t{pident:.6f}\t{query_id}\n")

        kept_summary = summarize(kept_records)
        test_summary = summarize(test_records)
        summary_rows.append(
            {
                "test_set": test_name,
                "test_file": str(test_path),
                "output_train_file": str(out_fasta),
                "criterion": f"MMseqs2 pident > {args.threshold:.1f}%",
                "original_train_records": train_summary["records"],
                "removed_train_records": len(removed_ids),
                "kept_train_records": kept_summary["records"],
                "kept_train_residues": kept_summary["residues"],
                "kept_label_1_disordered": kept_summary["label_1_disordered"],
                "kept_label_0_ordered": kept_summary["label_0_ordered"],
                "kept_label_-1_unknown": kept_summary["label_-1_unknown"],
                "test_records": test_summary["records"],
                "test_residues": test_summary["residues"],
                "hits_file": str(hits_path),
                "removed_ids_file": str(removed_path),
            }
        )

    summary_path = out_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    readme_path = out_dir / "README.md"
    readme_path.write_text(
        "# DM3000 NR25 Training Sets\n\n"
        "These files were generated from `data/DM3000_Train.fasta` by searching each "
        "test set against DM3000 with MMseqs2 and removing any DM3000 training "
        "sequence with an alignment sequence identity (`pident`) greater than 25% "
        "to at least one sequence in the corresponding test set.\n\n"
        "MMseqs2 command core: `easy-search <test_sequences> <DM3000_sequences> "
        "<hits.m8> <tmp> --min-seq-id 0.25 -s 7.5 --max-seqs 100000`.\n\n"
        "Output training FASTA files preserve the original three-line labeled FASTA "
        "format: header, sequence, label list.\n",
        encoding="utf-8",
    )

    print(summary_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        print(handle.read(), end="")


if __name__ == "__main__":
    main()
