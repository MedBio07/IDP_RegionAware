#!/usr/bin/env python3
"""Predict residue-level disorder probabilities with a trained sequence model."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from evaluate_disorder_predictions import parse_labeled_fasta
from models.features import feature_matrix
from models.sequence_models import AuxiliaryTCN, GenericTCN, RegionAwareTCN


def metadata_features(value: object) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def build_model(metadata: dict[str, object]) -> torch.nn.Module:
    model_type = str(metadata.get("model_type", "RegionAwareTCN"))
    kwargs = {
        "input_dim": int(metadata["input_dim"]),
        "hidden_dim": int(metadata["hidden_dim"]),
        "layers": int(metadata["layers"]),
        "dropout": float(metadata["dropout"]),
    }
    if model_type == "RegionAwareTCN":
        return RegionAwareTCN(**kwargs)
    if model_type == "GenericTCN":
        return GenericTCN(**kwargs)
    if model_type == "AuxiliaryTCN":
        return AuxiliaryTCN(**kwargs)
    raise ValueError(f"unsupported model type in checkpoint: {model_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    checkpoint = torch.load(args.model, map_location=device)
    metadata = checkpoint["metadata"]
    feature_names = metadata_features(metadata["features"])
    embedding_dir_text = metadata.get("embedding_dir")
    embedding_dir = Path(embedding_dir_text) if embedding_dir_text else None
    model = build_model(metadata).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    records = parse_labeled_fasta(args.fasta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "scores"])
        with torch.no_grad():
            for record in records:
                matrix = feature_matrix(str(record["sequence"]), feature_names, embedding_dir)
                x = torch.from_numpy(matrix).unsqueeze(0).to(device)
                mask = torch.ones((1, matrix.shape[0]), dtype=torch.float32, device=device)
                output = model(x, mask)
                scores = torch.sigmoid(output["disorder_logits"]).squeeze(0).detach().cpu().numpy()
                writer.writerow([record["id"], "[" + ", ".join(f"{float(score):.8g}" for score in scores) + "]"])


if __name__ == "__main__":
    main()
