"""Feature construction utilities for residue-level disorder baselines."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np


AA_ORDER = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: index for index, aa in enumerate(AA_ORDER)}


def sequence_sha1(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()


def embedding_path(embedding_dir: Path, sequence: str) -> Path:
    return embedding_dir / f"{sequence_sha1(sequence)}.npy"


def aa_one_hot(sequence: str) -> np.ndarray:
    matrix = np.zeros((len(sequence), len(AA_ORDER) + 1), dtype=np.float32)
    for index, aa in enumerate(sequence.upper()):
        aa_index = AA_INDEX.get(aa)
        if aa_index is None:
            matrix[index, -1] = 1.0
        else:
            matrix[index, aa_index] = 1.0
    return matrix


def position_features(length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("sequence length must be positive")
    if length == 1:
        relative = np.zeros(1, dtype=np.float32)
    else:
        relative = np.arange(length, dtype=np.float32) / float(length - 1)
    n_distance = relative
    c_distance = 1.0 - relative
    terminal_distance = np.minimum(n_distance, c_distance)
    log_length = np.full(length, math.log1p(length) / 10.0, dtype=np.float32)
    return np.column_stack([relative, n_distance, c_distance, terminal_distance, log_length]).astype(np.float32)


def load_embedding(embedding_dir: Path, sequence: str) -> np.ndarray:
    path = embedding_path(embedding_dir, sequence)
    if not path.exists():
        raise FileNotFoundError(f"missing embedding for sequence SHA1 {sequence_sha1(sequence)}: {path}")
    embedding = np.load(path).astype(np.float32, copy=False)
    if embedding.ndim != 2 or embedding.shape[0] != len(sequence):
        raise ValueError(f"embedding shape mismatch for {path}: {embedding.shape} vs sequence length {len(sequence)}")
    return embedding


def feature_matrix(
    sequence: str,
    feature_names: list[str],
    embedding_dir: Path | None = None,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for name in feature_names:
        if name == "onehot":
            parts.append(aa_one_hot(sequence))
        elif name == "position":
            parts.append(position_features(len(sequence)))
        elif name == "esm":
            if embedding_dir is None:
                raise ValueError("embedding_dir is required for esm features")
            parts.append(load_embedding(embedding_dir, sequence))
        else:
            raise ValueError(f"unsupported feature name: {name}")
    if not parts:
        raise ValueError("at least one feature must be requested")
    if len(parts) == 1:
        return parts[0].astype(np.float32, copy=False)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def parse_feature_list(text: str) -> list[str]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names:
        raise ValueError("feature list is empty")
    supported = {"onehot", "position", "esm"}
    unsupported = sorted(set(names) - supported)
    if unsupported:
        raise ValueError(f"unsupported features: {', '.join(unsupported)}")
    return names
