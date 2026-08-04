#!/usr/bin/env python3
"""Train a frozen-feature sequence model for residue-level disorder prediction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from torch.nn import functional as F

from annotate_disorder_regions import is_terminal_segment, iter_disorder_segments
from evaluate_disorder_predictions import parse_labeled_fasta, roc_auc
from models.features import feature_matrix, parse_feature_list
from models.sequence_models import AuxiliaryTCN, GenericTCN, RegionAdapterMoETCN, RegionAwareTCN


AUXILIARY_NAMES = ("sdr", "ldr", "terminal_idr", "internal_idr")


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return math.nan
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    tp = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / positives


def fmax(labels: list[int], scores: list[float]) -> tuple[float, float]:
    positives = sum(labels)
    if positives == 0:
        return math.nan, math.nan
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    tp = 0
    fp = 0
    best_f1 = 0.0
    best_threshold = 0.5
    i = 0
    while i < len(order):
        threshold = scores[order[i]]
        j = i
        while j < len(order) and scores[order[j]] == threshold:
            if labels[order[j]] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / positives
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        i = j
    return best_f1, best_threshold


def threshold_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        predicted = score >= threshold
        if label == 1 and predicted:
            tp += 1
        elif label == 1:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    sn = tp / (tp + fn) if (tp + fn) else math.nan
    sp = tn / (tn + fp) if (tn + fp) else math.nan
    bacc = (sn + sp) / 2.0 if math.isfinite(sn) and math.isfinite(sp) else math.nan
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else math.nan
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "sn": sn, "sp": sp, "bacc": bacc, "mcc": mcc}


def auxiliary_targets(labels: list[int]) -> np.ndarray:
    targets = np.zeros((len(labels), len(AUXILIARY_NAMES)), dtype=np.float32)
    for start, end, segment_length in iter_disorder_segments(labels):
        length_index = 0 if segment_length < 30 else 1
        location_index = 2 if is_terminal_segment(start, end, len(labels)) else 3
        targets[start - 1 : end, length_index] = 1.0
        targets[start - 1 : end, location_index] = 1.0
    return targets


def load_examples(
    fasta: Path,
    feature_names: list[str],
    embedding_dir: Path | None,
    cache_features: bool,
) -> list[dict[str, object]]:
    records = parse_labeled_fasta(fasta)
    examples: list[dict[str, object]] = []
    for record in records:
        sequence = str(record["sequence"])
        labels = np.asarray(record["labels"], dtype=np.float32)
        known_mask = labels != -1
        aux = auxiliary_targets([int(label) for label in record["labels"]])
        example = {
            "id": str(record["id"]),
            "sequence": sequence,
            "labels": np.where(known_mask, labels, 0.0).astype(np.float32),
            "known_mask": known_mask.astype(np.float32),
            "auxiliary": aux,
            "length": len(sequence),
        }
        if cache_features:
            example["features"] = feature_matrix(sequence, feature_names, embedding_dir)
        examples.append(example)
    return examples


def feature_for_example(example: dict[str, object], feature_names: list[str], embedding_dir: Path | None) -> np.ndarray:
    if "features" in example:
        return example["features"]
    return feature_matrix(str(example["sequence"]), feature_names, embedding_dir)


def make_batches(
    examples: list[dict[str, object]],
    max_tokens: int,
    max_proteins: int,
    shuffle: bool,
    rng: random.Random,
) -> list[list[dict[str, object]]]:
    items = list(examples)
    if shuffle:
        rng.shuffle(items)
    items.sort(key=lambda item: int(item["length"]))
    batches: list[list[dict[str, object]]] = []
    batch: list[dict[str, object]] = []
    max_len = 0
    for item in items:
        length = int(item["length"])
        prospective_max = max(max_len, length)
        prospective_size = len(batch) + 1
        if batch and (prospective_max * prospective_size > max_tokens or prospective_size > max_proteins):
            batches.append(batch)
            batch = []
            max_len = 0
        batch.append(item)
        max_len = max(max_len, length)
    if batch:
        batches.append(batch)
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate_batch(
    batch: list[dict[str, object]],
    feature_names: list[str],
    embedding_dir: Path | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    max_len = max(int(item["length"]) for item in batch)
    first_features = feature_for_example(batch[0], feature_names, embedding_dir)
    input_dim = first_features.shape[1]
    x = np.zeros((len(batch), max_len, input_dim), dtype=np.float32)
    labels = np.zeros((len(batch), max_len), dtype=np.float32)
    known_mask = np.zeros((len(batch), max_len), dtype=np.float32)
    pad_mask = np.zeros((len(batch), max_len), dtype=np.float32)
    aux = np.zeros((len(batch), max_len, len(AUXILIARY_NAMES)), dtype=np.float32)
    for row, item in enumerate(batch):
        features = first_features if row == 0 else feature_for_example(item, feature_names, embedding_dir)
        length = int(item["length"])
        x[row, :length] = features
        labels[row, :length] = item["labels"]
        known_mask[row, :length] = item["known_mask"]
        pad_mask[row, :length] = 1.0
        aux[row, :length] = item["auxiliary"]
    return {
        "features": torch.from_numpy(x).to(device),
        "labels": torch.from_numpy(labels).to(device),
        "known_mask": torch.from_numpy(known_mask).to(device),
        "pad_mask": torch.from_numpy(pad_mask).to(device),
        "auxiliary": torch.from_numpy(aux).to(device),
    }


def compute_positive_weights(examples: list[dict[str, object]]) -> tuple[float, np.ndarray]:
    positives = negatives = 0.0
    aux_pos = np.zeros(len(AUXILIARY_NAMES), dtype=np.float64)
    aux_neg = np.zeros(len(AUXILIARY_NAMES), dtype=np.float64)
    for item in examples:
        mask = item["known_mask"].astype(bool)
        labels = item["labels"][mask]
        positives += float(np.sum(labels == 1))
        negatives += float(np.sum(labels == 0))
        aux = item["auxiliary"][mask]
        aux_pos += np.sum(aux == 1, axis=0)
        aux_neg += np.sum(aux == 0, axis=0)
    disorder_weight = negatives / positives if positives else 1.0
    aux_weights = np.divide(aux_neg, aux_pos, out=np.ones_like(aux_neg), where=aux_pos > 0)
    return min(disorder_weight, 25.0), np.minimum(aux_weights, 25.0).astype(np.float32)


def masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    return torch.sum(loss * mask) / torch.clamp(torch.sum(mask), min=1.0)


def masked_disorder_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
    loss_type: str,
    focal_gamma: float,
    asym_gamma_pos: float,
    asym_gamma_neg: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    if loss_type == "bce":
        loss = bce
    elif loss_type == "focal":
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        loss = bce * torch.pow(torch.clamp(1.0 - p_t, min=0.0, max=1.0), focal_gamma)
    elif loss_type == "asymmetric":
        probabilities = torch.sigmoid(logits)
        positive_factor = torch.pow(torch.clamp(1.0 - probabilities, min=0.0, max=1.0), asym_gamma_pos)
        negative_factor = torch.pow(torch.clamp(probabilities, min=0.0, max=1.0), asym_gamma_neg)
        factor = positive_factor * targets + negative_factor * (1.0 - targets)
        loss = bce * factor
    else:
        raise ValueError(f"unsupported loss type: {loss_type}")
    return torch.sum(loss * mask) / torch.clamp(torch.sum(mask), min=1.0)


def masked_region_gate_loss(
    gate_weights: torch.Tensor,
    region_targets: torch.Tensor,
    known_mask: torch.Tensor,
) -> torch.Tensor:
    target_sum = torch.sum(region_targets, dim=-1, keepdim=True)
    gate_mask = known_mask * (target_sum.squeeze(-1) > 0).float()
    normalized_targets = region_targets / torch.clamp(target_sum, min=1.0)
    log_gate = torch.log(torch.clamp(gate_weights, min=1.0e-8))
    loss = -torch.sum(normalized_targets * log_gate, dim=-1)
    return torch.sum(loss * gate_mask) / torch.clamp(torch.sum(gate_mask), min=1.0)


def build_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
    adapter_dim: int,
    gate_temperature: float,
) -> torch.nn.Module:
    if model_type in ("RegionAwareTCN", "region_aware_tcn"):
        return RegionAwareTCN(input_dim=input_dim, hidden_dim=hidden_dim, layers=layers, dropout=dropout)
    if model_type in ("GenericTCN", "generic_tcn"):
        return GenericTCN(input_dim=input_dim, hidden_dim=hidden_dim, layers=layers, dropout=dropout)
    if model_type in ("AuxiliaryTCN", "auxiliary_tcn"):
        return AuxiliaryTCN(input_dim=input_dim, hidden_dim=hidden_dim, layers=layers, dropout=dropout)
    if model_type in ("RegionAdapterMoETCN", "region_adapter_moe_tcn"):
        return RegionAdapterMoETCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
            adapter_dim=adapter_dim,
            gate_temperature=gate_temperature,
        )
    raise ValueError(f"unsupported model type: {model_type}")


def canonical_model_type(model_type: str) -> str:
    if model_type in ("RegionAwareTCN", "region_aware_tcn"):
        return "RegionAwareTCN"
    if model_type in ("GenericTCN", "generic_tcn"):
        return "GenericTCN"
    if model_type in ("AuxiliaryTCN", "auxiliary_tcn"):
        return "AuxiliaryTCN"
    if model_type in ("RegionAdapterMoETCN", "region_adapter_moe_tcn"):
        return "RegionAdapterMoETCN"
    raise ValueError(f"unsupported model type: {model_type}")


def initialize_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> list[str]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()
    loaded_keys: list[str] = []
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key].copy_(value)
            loaded_keys.append(key)
    if isinstance(model, RegionAdapterMoETCN):
        expert_weight = source_state.get("expert_head.weight")
        expert_bias = source_state.get("expert_head.bias")
        if expert_weight is not None:
            for index in range(min(4, expert_weight.shape[0])):
                key = f"expert_heads.{index}.weight"
                if key in target_state and target_state[key].shape == expert_weight[index : index + 1].shape:
                    target_state[key].copy_(expert_weight[index : index + 1])
                    loaded_keys.append(key)
        if expert_bias is not None:
            for index in range(min(4, expert_bias.shape[0])):
                key = f"expert_heads.{index}.bias"
                if key in target_state and target_state[key].shape == expert_bias[index : index + 1].shape:
                    target_state[key].copy_(expert_bias[index : index + 1])
                    loaded_keys.append(key)
    model.load_state_dict(target_state)
    return loaded_keys


def freeze_shared_backbone_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable_prefixes = ("region_adapters.", "expert_heads.", "gate_head.")
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        parameter.requires_grad = name.startswith(trainable_prefixes)
        if parameter.requires_grad:
            trainable += parameter.numel()
    return total, trainable


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return total, trainable


def train_epoch(
    model: torch.nn.Module,
    examples: list[dict[str, object]],
    feature_names: list[str],
    embedding_dir: Path | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_tokens: int,
    max_proteins: int,
    disorder_pos_weight: float,
    aux_pos_weight: np.ndarray,
    aux_loss_weight: float,
    loss_type: str,
    focal_gamma: float,
    asym_gamma_pos: float,
    asym_gamma_neg: float,
    gate_loss_weight: float,
    rng: random.Random,
) -> dict[str, float]:
    model.train()
    batches = make_batches(examples, max_tokens, max_proteins, shuffle=True, rng=rng)
    disorder_weight = torch.tensor(disorder_pos_weight, dtype=torch.float32, device=device)
    aux_weight = torch.tensor(aux_pos_weight, dtype=torch.float32, device=device)
    total_loss = 0.0
    total_main = 0.0
    total_aux = 0.0
    total_gate = 0.0
    total_known = 0.0
    for batch in batches:
        tensors = collate_batch(batch, feature_names, embedding_dir, device)
        output = model(tensors["features"], tensors["pad_mask"])
        main_loss = masked_disorder_loss(
            output["disorder_logits"],
            tensors["labels"],
            tensors["known_mask"],
            disorder_weight,
            loss_type,
            focal_gamma,
            asym_gamma_pos,
            asym_gamma_neg,
        )
        aux_loss = output["disorder_logits"].new_tensor(0.0)
        if aux_loss_weight > 0.0:
            if "auxiliary_logits" not in output:
                raise ValueError("auxiliary loss requires a model with auxiliary_logits")
            aux_mask = tensors["known_mask"].unsqueeze(-1)
            aux_loss = masked_bce(
                output["auxiliary_logits"],
                tensors["auxiliary"],
                aux_mask,
                aux_weight,
            )
        gate_loss = output["disorder_logits"].new_tensor(0.0)
        if gate_loss_weight > 0.0:
            if "gate_weights" not in output:
                raise ValueError("gate loss requires a model with gate_weights")
            gate_loss = masked_region_gate_loss(
                output["gate_weights"],
                tensors["auxiliary"],
                tensors["known_mask"],
            )
        loss = main_loss + aux_loss_weight * aux_loss + gate_loss_weight * gate_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        known = float(torch.sum(tensors["known_mask"]).detach().cpu())
        total_loss += float(loss.detach().cpu()) * known
        total_main += float(main_loss.detach().cpu()) * known
        total_aux += float(aux_loss.detach().cpu()) * known
        total_gate += float(gate_loss.detach().cpu()) * known
        total_known += known
    return {
        "train_loss": total_loss / total_known,
        "train_main_loss": total_main / total_known,
        "train_aux_loss": total_aux / total_known,
        "train_gate_loss": total_gate / total_known,
        "train_batches": float(len(batches)),
    }


def predict_examples(
    model: torch.nn.Module,
    examples: list[dict[str, object]],
    feature_names: list[str],
    embedding_dir: Path | None,
    device: torch.device,
    max_tokens: int,
    max_proteins: int,
) -> dict[str, list[float]]:
    model.eval()
    predictions: dict[str, list[float]] = {}
    rng = random.Random(0)
    batches = make_batches(examples, max_tokens, max_proteins, shuffle=False, rng=rng)
    with torch.no_grad():
        for batch in batches:
            tensors = collate_batch(batch, feature_names, embedding_dir, device)
            output = model(tensors["features"], tensors["pad_mask"])
            scores = torch.sigmoid(output["disorder_logits"]).detach().cpu().numpy()
            for row, item in enumerate(batch):
                length = int(item["length"])
                predictions[str(item["id"])] = [float(value) for value in scores[row, :length]]
    return predictions


def collect_known_labels_scores(
    examples: list[dict[str, object]],
    predictions: dict[str, list[float]],
) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    scores: list[float] = []
    for item in examples:
        item_scores = predictions[str(item["id"])]
        mask = item["known_mask"].astype(bool)
        item_labels = item["labels"].astype(np.int8)
        for label, score, keep in zip(item_labels, item_scores, mask):
            if keep:
                labels.append(int(label))
                scores.append(float(score))
    return labels, scores


def evaluate_predictions(examples: list[dict[str, object]], predictions: dict[str, list[float]]) -> dict[str, float | int]:
    labels, scores = collect_known_labels_scores(examples, predictions)
    fmax_value, threshold = fmax(labels, scores)
    threshold_row = threshold_metrics(labels, scores, threshold)
    return {
        "validation_residues": len(labels),
        "validation_positives": sum(labels),
        "validation_negatives": len(labels) - sum(labels),
        "threshold": threshold,
        "validation_auc": roc_auc(labels, scores),
        "validation_aupr": average_precision(labels, scores),
        "validation_fmax": fmax_value,
        "validation_sn": threshold_row["sn"],
        "validation_sp": threshold_row["sp"],
        "validation_bacc": threshold_row["bacc"],
        "validation_mcc": threshold_row["mcc"],
    }


def write_prediction_tsv(path: Path, predictions: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "scores"])
        for protein_id, scores in predictions.items():
            writer.writerow([protein_id, "[" + ", ".join(f"{score:.8g}" for score in scores) + "]"])


def write_metrics(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "train_set",
        "validation_set",
        "features",
        "model_type",
        "train_residues",
        "train_positives",
        "train_negatives",
        "validation_residues",
        "validation_positives",
        "validation_negatives",
        "threshold",
        "validation_auc",
        "validation_aupr",
        "validation_fmax",
        "validation_sn",
        "validation_sp",
        "validation_bacc",
        "validation_mcc",
        "fit_seconds",
        "best_epoch",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def write_epoch_log(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_main_loss",
        "train_aux_loss",
        "train_gate_loss",
        "train_batches",
        "validation_auc",
        "validation_aupr",
        "validation_fmax",
        "validation_mcc",
        "validation_bacc",
        "threshold",
        "epoch_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def format_value(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.6f}"
    return value


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--features", default="esm,position,onehot")
    parser.add_argument(
        "--model-type",
        default="RegionAwareTCN",
        choices=(
            "RegionAwareTCN",
            "region_aware_tcn",
            "GenericTCN",
            "generic_tcn",
            "AuxiliaryTCN",
            "auxiliary_tcn",
            "RegionAdapterMoETCN",
            "region_adapter_moe_tcn",
        ),
    )
    parser.add_argument("--embedding-dir", type=Path)
    parser.add_argument("--model-out", required=True, type=Path)
    parser.add_argument("--metrics-out", required=True, type=Path)
    parser.add_argument("--epoch-log-out", required=True, type=Path)
    parser.add_argument("--validation-predictions-out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--adapter-dim", type=int, default=32)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--init-from-checkpoint", type=Path)
    parser.add_argument("--freeze-shared-backbone", action="store_true")
    parser.add_argument("--select-initial-checkpoint", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-loss-weight", type=float, default=0.20)
    parser.add_argument("--gate-loss-weight", type=float, default=0.0)
    parser.add_argument("--loss-type", choices=("bce", "focal", "asymmetric"), default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--asym-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asym-gamma-neg", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--max-proteins", type=int, default=32)
    parser.add_argument("--no-cache-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    feature_names = parse_feature_list(args.features)
    if "esm" in feature_names and args.embedding_dir is None:
        raise ValueError("--embedding-dir is required when using esm features")
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    cache_features = not args.no_cache_features
    train_examples = load_examples(args.train, feature_names, args.embedding_dir, cache_features)
    validation_examples = load_examples(args.validation, feature_names, args.embedding_dir, cache_features)
    input_dim = feature_for_example(train_examples[0], feature_names, args.embedding_dir).shape[1]
    disorder_pos_weight, aux_pos_weight = compute_positive_weights(train_examples)
    train_labels = np.concatenate([item["labels"][item["known_mask"].astype(bool)] for item in train_examples])
    model_type = canonical_model_type(args.model_type)

    model = build_model(
        model_type=model_type,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
        adapter_dim=args.adapter_dim,
        gate_temperature=args.gate_temperature,
    ).to(device)
    loaded_init_keys: list[str] = []
    if args.init_from_checkpoint:
        loaded_init_keys = initialize_model_from_checkpoint(model, args.init_from_checkpoint, device)
    total_parameters, trainable_parameters = count_trainable_parameters(model)
    if args.freeze_shared_backbone:
        total_parameters, trainable_parameters = freeze_shared_backbone_parameters(model)
    trainable_parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameter_list:
        raise ValueError("no trainable parameters remain after freezing")
    optimizer = torch.optim.AdamW(trainable_parameter_list, lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)

    best_auc = -1.0
    best_metadata: dict[str, object] | None = None
    epoch_rows: list[dict[str, object]] = []

    def checkpoint_metadata(epoch: int, val_metrics: dict[str, float | int]) -> dict[str, object]:
        return {
            "experiment_id": args.experiment_id,
            "train": str(args.train),
            "validation": str(args.validation),
            "features": feature_names,
            "embedding_dir": str(args.embedding_dir) if args.embedding_dir else None,
            "model_type": model_type,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "dropout": args.dropout,
            "adapter_dim": args.adapter_dim,
            "gate_temperature": args.gate_temperature,
            "init_from_checkpoint": str(args.init_from_checkpoint) if args.init_from_checkpoint else None,
            "loaded_init_keys": len(loaded_init_keys),
            "freeze_shared_backbone": args.freeze_shared_backbone,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "auxiliary_names": list(AUXILIARY_NAMES),
            "aux_loss_weight": args.aux_loss_weight,
            "gate_loss_weight": args.gate_loss_weight,
            "loss_type": args.loss_type,
            "focal_gamma": args.focal_gamma,
            "asym_gamma_pos": args.asym_gamma_pos,
            "asym_gamma_neg": args.asym_gamma_neg,
            "disorder_pos_weight": disorder_pos_weight,
            "aux_pos_weight": aux_pos_weight.tolist(),
            "threshold": val_metrics["threshold"],
            "validation_auc": val_metrics["validation_auc"],
            "validation_aupr": val_metrics["validation_aupr"],
            "validation_fmax": val_metrics["validation_fmax"],
            "validation_mcc": val_metrics["validation_mcc"],
            "best_epoch": epoch,
            "seed": args.seed,
        }

    if args.select_initial_checkpoint:
        val_predictions = predict_examples(
            model=model,
            examples=validation_examples,
            feature_names=feature_names,
            embedding_dir=args.embedding_dir,
            device=device,
            max_tokens=args.max_tokens,
            max_proteins=args.max_proteins,
        )
        val_metrics = evaluate_predictions(validation_examples, val_predictions)
        best_auc = float(val_metrics["validation_auc"])
        best_metadata = checkpoint_metadata(0, val_metrics)
        save_checkpoint(args.model_out, model, best_metadata)
        epoch_row = {
            "epoch": 0,
            "train_loss": math.nan,
            "train_main_loss": math.nan,
            "train_aux_loss": math.nan,
            "train_gate_loss": math.nan,
            "train_batches": 0.0,
            **val_metrics,
            "epoch_seconds": 0.0,
        }
        epoch_rows.append(epoch_row)
        print(json.dumps({key: format_value(value) for key, value in epoch_row.items()}, ensure_ascii=False), flush=True)

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_row = train_epoch(
            model=model,
            examples=train_examples,
            feature_names=feature_names,
            embedding_dir=args.embedding_dir,
            optimizer=optimizer,
            device=device,
            max_tokens=args.max_tokens,
            max_proteins=args.max_proteins,
            disorder_pos_weight=disorder_pos_weight,
            aux_pos_weight=aux_pos_weight,
            aux_loss_weight=args.aux_loss_weight,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            asym_gamma_pos=args.asym_gamma_pos,
            asym_gamma_neg=args.asym_gamma_neg,
            gate_loss_weight=args.gate_loss_weight,
            rng=rng,
        )
        val_predictions = predict_examples(
            model=model,
            examples=validation_examples,
            feature_names=feature_names,
            embedding_dir=args.embedding_dir,
            device=device,
            max_tokens=args.max_tokens,
            max_proteins=args.max_proteins,
        )
        val_metrics = evaluate_predictions(validation_examples, val_predictions)
        epoch_row = {
            "epoch": epoch,
            **train_row,
            **val_metrics,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        epoch_rows.append(epoch_row)
        print(json.dumps({key: format_value(value) for key, value in epoch_row.items()}, ensure_ascii=False), flush=True)
        if float(val_metrics["validation_auc"]) > best_auc:
            best_auc = float(val_metrics["validation_auc"])
            best_metadata = checkpoint_metadata(epoch, val_metrics)
            save_checkpoint(args.model_out, model, best_metadata)

    if best_metadata is None:
        raise RuntimeError("no checkpoint was selected")
    checkpoint = torch.load(args.model_out, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_predictions = predict_examples(
        model=model,
        examples=validation_examples,
        feature_names=feature_names,
        embedding_dir=args.embedding_dir,
        device=device,
        max_tokens=args.max_tokens,
        max_proteins=args.max_proteins,
    )
    final_metrics = evaluate_predictions(validation_examples, final_predictions)
    metrics_row = {
        "experiment_id": args.experiment_id,
        "train_set": str(args.train),
        "validation_set": str(args.validation),
        "features": ",".join(feature_names),
        "model_type": model_type,
        "train_residues": int(len(train_labels)),
        "train_positives": int(np.sum(train_labels == 1)),
        "train_negatives": int(np.sum(train_labels == 0)),
        **final_metrics,
        "fit_seconds": time.perf_counter() - started,
        "best_epoch": checkpoint["metadata"]["best_epoch"],
    }
    write_epoch_log(args.epoch_log_out, epoch_rows)
    write_metrics(args.metrics_out, metrics_row)
    if args.validation_predictions_out:
        write_prediction_tsv(args.validation_predictions_out, final_predictions)
    checkpoint["metadata"].update(metrics_row)
    checkpoint["metadata"]["features"] = feature_names
    checkpoint["metadata"]["features_text"] = ",".join(feature_names)
    save_checkpoint(args.model_out, model, checkpoint["metadata"])
    print(json.dumps({key: format_value(value) for key, value in metrics_row.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
