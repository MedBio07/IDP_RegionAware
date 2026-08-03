#!/usr/bin/env python3
"""Extract and cache per-residue ESM2 embeddings for local FASTA datasets."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_disorder_predictions import parse_labeled_fasta
from models.features import embedding_path, sequence_sha1


MODEL_SPECS = {
    "esm2_t6_8M_UR50D": ("esm2_t6_8M_UR50D", 6),
    "esm2_t12_35M_UR50D": ("esm2_t12_35M_UR50D", 12),
    "esm2_t30_150M_UR50D": ("esm2_t30_150M_UR50D", 30),
    "esm2_t33_650M_UR50D": ("esm2_t33_650M_UR50D", 33),
    "esm2_t36_3B_UR50D": ("esm2_t36_3B_UR50D", 36),
    "esm2_t48_15B_UR50D": ("esm2_t48_15B_UR50D", 48),
}

DEFAULT_FASTAS = (
    ("DM3000_Train", Path("data/DM3000_Train.fasta")),
    ("DM1229_Validation", Path("data/DM1229_Validation.fasta")),
    ("SL329", Path("data/SL329_test.fasta")),
    ("MXD494", Path("data/MXD494_test.fasta")),
    ("DISORDER723", Path("data/DISORDER723_test.fasta")),
    ("DM3000_Train_nr25_vs_SL329", Path("data/nr25_by_test/DM3000_Train_nr25_vs_SL329.fasta")),
    ("DM3000_Train_nr25_vs_MXD494", Path("data/nr25_by_test/DM3000_Train_nr25_vs_MXD494.fasta")),
    ("DM3000_Train_nr25_vs_DISORDER723", Path("data/nr25_by_test/DM3000_Train_nr25_vs_DISORDER723.fasta")),
)


def parse_fasta_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("FASTA must be DATASET=PATH")
    name, path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("dataset name is empty")
    return name.strip(), Path(path.strip())


def load_unique_sequences(fastas: list[tuple[str, Path]]) -> dict[str, dict[str, object]]:
    sequences: dict[str, dict[str, object]] = {}
    for dataset, path in fastas:
        for record in parse_labeled_fasta(path):
            sequence = str(record["sequence"]).upper()
            digest = sequence_sha1(sequence)
            entry = sequences.setdefault(
                digest,
                {"sequence": sequence, "length": len(sequence), "datasets": set(), "ids": []},
            )
            entry["datasets"].add(dataset)
            entry["ids"].append(str(record["id"]))
    return sequences


def sanitize_sequence(sequence: str, alphabet: object) -> str:
    token_to_idx = getattr(alphabet, "tok_to_idx")
    fallback = "X" if "X" in token_to_idx else "A"
    return "".join(aa if aa in token_to_idx and len(aa) == 1 else fallback for aa in sequence.upper())


def chunk_sequence(sequence: str, max_residues: int, overlap: int) -> list[tuple[int, str]]:
    if max_residues <= 0:
        raise ValueError("max_residues must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_residues:
        raise ValueError("overlap must be smaller than max_residues")
    if len(sequence) <= max_residues:
        return [(0, sequence)]
    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(sequence):
        end = min(len(sequence), start + max_residues)
        chunks.append((start, sequence[start:end]))
        if end == len(sequence):
            break
        start = end - overlap
    return chunks


def make_batches(chunks: list[tuple[str, int, str]], max_batch_tokens: int) -> list[list[tuple[str, int, str]]]:
    batches: list[list[tuple[str, int, str]]] = []
    batch: list[tuple[str, int, str]] = []
    token_count = 0
    for item in chunks:
        length = len(item[2]) + 2
        if batch and token_count + length > max_batch_tokens:
            batches.append(batch)
            batch = []
            token_count = 0
        batch.append(item)
        token_count += length
    if batch:
        batches.append(batch)
    return batches


def load_model(model_name: str, device: str) -> tuple[object, object, int]:
    import esm
    import torch

    if model_name not in MODEL_SPECS:
        raise ValueError(f"unsupported model name: {model_name}")
    loader_name, layer = MODEL_SPECS[model_name]
    loader = getattr(esm.pretrained, loader_name)
    model, alphabet = loader()
    model.eval()
    model.to(torch.device(device))
    return model, alphabet, layer


def cache_valid(path: Path, expected_length: int) -> bool:
    if not path.exists():
        return False
    try:
        import numpy as np

        array = np.load(path, mmap_mode="r")
    except Exception:
        return False
    return array.ndim == 2 and array.shape[0] == expected_length


def write_manifest(cache_dir: Path, sequences: dict[str, dict[str, object]], model_name: str, layer: int) -> None:
    with (cache_dir / "manifest.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        fieldnames = [
            "sequence_sha1",
            "file_name",
            "sequence_length",
            "model_name",
            "layer",
            "dtype",
            "datasets",
            "example_ids",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for digest, entry in sorted(sequences.items(), key=lambda item: (int(item[1]["length"]), item[0])):
            writer.writerow(
                {
                    "sequence_sha1": digest,
                    "file_name": f"{digest}.npy",
                    "sequence_length": entry["length"],
                    "model_name": model_name,
                    "layer": layer,
                    "dtype": "float16",
                    "datasets": ",".join(sorted(entry["datasets"])),
                    "example_ids": "|".join(entry["ids"][:5]),
                }
            )


def extract(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    fastas = args.fasta if args.fasta else list(DEFAULT_FASTAS)
    sequences = load_unique_sequences(fastas)
    model, alphabet, layer = load_model(args.model_name, args.device)
    batch_converter = alphabet.get_batch_converter()
    cache_dir = args.cache_root / f"{args.model_name}_layer{layer}_fp16"
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        digest
        for digest, entry in sorted(sequences.items(), key=lambda item: (int(item[1]["length"]), item[0]))
        if not cache_valid(embedding_path(cache_dir, str(entry["sequence"])), int(entry["length"]))
    ]
    started = time.perf_counter()
    if missing:
        chunks: list[tuple[str, int, str]] = []
        expected_chunks: dict[str, int] = {}
        for digest in missing:
            sequence = sanitize_sequence(str(sequences[digest]["sequence"]), alphabet)
            per_sequence_chunks = chunk_sequence(sequence, args.max_residues, args.overlap)
            expected_chunks[digest] = len(per_sequence_chunks)
            chunks.extend((digest, start, chunk) for start, chunk in per_sequence_chunks)
        batches = make_batches(chunks, args.max_batch_tokens)
        accum: dict[str, np.ndarray] = {}
        counts: dict[str, np.ndarray] = {}
        generated = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(batches, start=1):
                labels = [f"{index}:{digest}:{start}" for index, (digest, start, _) in enumerate(batch)]
                batch_sequences = [sequence for _, _, sequence in batch]
                _, _, tokens = batch_converter(list(zip(labels, batch_sequences)))
                tokens = tokens.to(args.device)
                output = model(tokens, repr_layers=[layer], return_contacts=False)
                reps = output["representations"][layer].detach().cpu().numpy()
                for row_index, (digest, start, sequence) in enumerate(batch):
                    length = len(sequence)
                    piece = reps[row_index, 1 : length + 1].astype(np.float32, copy=False)
                    total_length = int(sequences[digest]["length"])
                    if digest not in accum:
                        accum[digest] = np.zeros((total_length, piece.shape[1]), dtype=np.float32)
                        counts[digest] = np.zeros(total_length, dtype=np.float32)
                    end = start + length
                    accum[digest][start:end] += piece
                    counts[digest][start:end] += 1.0
                    expected_chunks[digest] -= 1
                    if expected_chunks[digest] == 0:
                        if not np.all(counts[digest] > 0):
                            raise ValueError(f"chunk coverage gap for sequence {digest}")
                        embedding = accum[digest] / counts[digest][:, None]
                        np.save(embedding_path(cache_dir, str(sequences[digest]["sequence"])), embedding.astype(np.float16))
                        del accum[digest]
                        del counts[digest]
                        generated += 1
                        if args.progress_every and generated % args.progress_every == 0:
                            print(
                                f"Generated {generated}/{len(missing)} embeddings; "
                                f"batch {batch_index}/{len(batches)}",
                                flush=True,
                            )
        if accum:
            raise RuntimeError(f"unfinished sequences: {len(accum)}")
    write_manifest(cache_dir, sequences, args.model_name, layer)
    elapsed = time.perf_counter() - started
    print(
        f"cache_dir={cache_dir}\n"
        f"unique_sequences={len(sequences)}\n"
        f"generated={len(missing)}\n"
        f"seconds={elapsed:.3f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", action="append", type=parse_fasta_arg, help="Dataset FASTA as NAME=PATH.")
    parser.add_argument("--model-name", default="esm2_t12_35M_UR50D", choices=sorted(MODEL_SPECS))
    parser.add_argument("--cache-root", type=Path, default=Path("data/features/esm2_embeddings"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-residues", type=int, default=1022)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract(args)


if __name__ == "__main__":
    main()
