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
from models.sequence_models import (
    AuxiliaryTCN,
    FactorizedRegionAdapterMoETCN,
    GenericTCN,
    RegionAdapterMoETCN,
    RegionAwareTCN,
)


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


def _contiguous_true_runs(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return half-open runs of true values from one residue mask."""
    if mask.ndim != 1:
        raise ValueError("a residue mask must be one-dimensional")
    values = mask.detach().to(device="cpu", dtype=torch.bool).tolist()
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def sample_internal_ordered_pairs(
    labels: torch.Tensor,
    auxiliary: torch.Tensor,
    known_mask: torch.Tensor,
    max_per_segment: int,
    rng: random.Random,
) -> list[tuple[int, list[int], list[int]]]:
    """Sample balanced residue pools for each internal disorder segment.

    The returned tuples contain ``(batch_index, internal_positive_indices,
    ordered_negative_indices)``.  Both pools are sampled independently up to
    ``max_per_segment``; zero means no cap.  The loss caller averages each
    segment before averaging segments, so long segments and proteins cannot
    dominate merely by contributing more residues.  Terminal disorder is
    intentionally absent from the negative pool.
    """
    if max_per_segment < 0:
        raise ValueError("max_per_segment must be non-negative")
    if labels.ndim != 2 or known_mask.shape != labels.shape:
        raise ValueError("labels and known_mask must have shape [batch, length]")
    if auxiliary.ndim != 3 or auxiliary.shape[:2] != labels.shape or auxiliary.shape[-1] < 4:
        raise ValueError("auxiliary must have shape [batch, length, >=4]")

    known = known_mask > 0.5
    internal_positive = (labels > 0.5) & (auxiliary[..., 3] > 0.5) & known
    ordered_negative = (labels <= 0.5) & known
    sampled: list[tuple[int, list[int], list[int]]] = []
    for batch_index in range(labels.shape[0]):
        negative_candidates = [
            int(index) for index in torch.where(ordered_negative[batch_index])[0].detach().cpu().tolist()
        ]
        if not negative_candidates:
            continue
        for start, end in _contiguous_true_runs(internal_positive[batch_index]):
            positive_candidates = list(range(start, end))
            if max_per_segment > 0:
                if len(positive_candidates) > max_per_segment:
                    positive_candidates = rng.sample(positive_candidates, max_per_segment)
                if len(negative_candidates) > max_per_segment:
                    negatives = rng.sample(negative_candidates, max_per_segment)
                else:
                    negatives = list(negative_candidates)
            else:
                negatives = list(negative_candidates)
            positive_candidates.sort()
            negatives.sort()
            if positive_candidates and negatives:
                sampled.append((batch_index, positive_candidates, negatives))
    return sampled


def masked_internal_ordered_pairwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    auxiliary: torch.Tensor,
    known_mask: torch.Tensor,
    margin: float,
    max_per_segment: int,
    rng: random.Random,
) -> tuple[torch.Tensor, int, int]:
    """Rank internal disorder logits above ordered logits within each protein.

    A segment contributes the mean of all sampled positive-negative softplus
    pairs, and all eligible internal segments contribute equally.  The margin
    shifts the softplus target; sampling only selects indices and does not
    detach the selected logits.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")
    sampled = sample_internal_ordered_pairs(
        labels=labels,
        auxiliary=auxiliary,
        known_mask=known_mask,
        max_per_segment=max_per_segment,
        rng=rng,
    )
    segment_losses: list[torch.Tensor] = []
    pair_count = 0
    for batch_index, positive_indices, negative_indices in sampled:
        positive_index = torch.tensor(positive_indices, dtype=torch.long, device=logits.device)
        negative_index = torch.tensor(negative_indices, dtype=torch.long, device=logits.device)
        positive_logits = logits[batch_index].index_select(0, positive_index)
        negative_logits = logits[batch_index].index_select(0, negative_index)
        pair_losses = F.softplus(
            logits.new_tensor(margin)
            - positive_logits.unsqueeze(-1)
            + negative_logits.unsqueeze(0)
        )
        segment_losses.append(torch.mean(pair_losses))
        pair_count += len(positive_indices) * len(negative_indices)
    if not segment_losses:
        return logits.sum() * 0.0, 0, 0
    return torch.mean(torch.stack(segment_losses)), pair_count, len(segment_losses)


def masked_teacher_logit_huber_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    auxiliary: torch.Tensor,
    known_mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Protect ordered and terminal-positive logits against a frozen teacher.

    Internal-positive residues are deliberately excluded so the new ranking
    objective can change the model where the motivation predicts the largest
    regional weakness.  ``teacher_logits`` is detached defensively; callers
    should still run the teacher in ``eval``/``no_grad`` mode.
    """
    if delta <= 0:
        raise ValueError("Huber delta must be positive")
    if student_logits.shape != teacher_logits.shape or student_logits.shape != labels.shape:
        raise ValueError("student_logits, teacher_logits, and labels must have matching shapes")
    if auxiliary.ndim != 3 or auxiliary.shape[:2] != labels.shape or auxiliary.shape[-1] < 4:
        raise ValueError("auxiliary must have shape [batch, length, >=4]")
    if known_mask.shape != labels.shape:
        raise ValueError("known_mask must match labels")
    ordered = labels <= 0.5
    terminal_positive = (labels > 0.5) & (auxiliary[..., 2] > 0.5)
    protection_mask = (ordered | terminal_positive) & (known_mask > 0.5)
    huber = F.huber_loss(
        student_logits,
        teacher_logits.detach(),
        reduction="none",
        delta=delta,
    )
    mask = protection_mask.to(dtype=huber.dtype)
    return torch.sum(huber * mask) / torch.clamp(torch.sum(mask), min=1.0)


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
    if model_type in ("FactorizedRegionAdapterMoETCN", "factorized_region_adapter_moe_tcn"):
        return FactorizedRegionAdapterMoETCN(
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
    if model_type in ("FactorizedRegionAdapterMoETCN", "factorized_region_adapter_moe_tcn"):
        return "FactorizedRegionAdapterMoETCN"
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
    if isinstance(model, FactorizedRegionAdapterMoETCN):
        legacy_gate_weight = source_state.get("gate_head.weight")
        legacy_gate_bias = source_state.get("gate_head.bias")
        if legacy_gate_weight is not None and legacy_gate_weight.ndim == 2 and legacy_gate_weight.shape[0] >= 4:
            for target_key, start, end in (
                ("length_gate_head.weight", 0, 2),
                ("location_gate_head.weight", 2, 4),
            ):
                value = legacy_gate_weight[start:end]
                if target_key in target_state and target_state[target_key].shape == value.shape:
                    target_state[target_key].copy_(value)
                    loaded_keys.append(target_key)
        if legacy_gate_bias is not None and legacy_gate_bias.ndim == 1 and legacy_gate_bias.shape[0] >= 4:
            for target_key, start, end in (
                ("length_gate_head.bias", 0, 2),
                ("location_gate_head.bias", 2, 4),
            ):
                value = legacy_gate_bias[start:end]
                if target_key in target_state and target_state[target_key].shape == value.shape:
                    target_state[target_key].copy_(value)
                    loaded_keys.append(target_key)
    model.load_state_dict(target_state)
    return loaded_keys


def last_block_parameter_prefixes(model: torch.nn.Module) -> tuple[str, str]:
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0 or not hasattr(model, "final_norm"):
        raise ValueError("--unfreeze-last-block requires a model with TCN blocks and final_norm")
    return (f"blocks.{len(blocks) - 1}.", "final_norm.")


def freeze_shared_backbone_parameters(
    model: torch.nn.Module,
    unfreeze_last_block: bool = False,
) -> tuple[int, int]:
    if isinstance(model, FactorizedRegionAdapterMoETCN):
        trainable_prefixes = (
            "region_adapters.",
            "expert_heads.",
            "length_gate_head.",
            "location_gate_head.",
        )
    else:
        trainable_prefixes = ("region_adapters.", "expert_heads.", "gate_head.")
    if unfreeze_last_block:
        trainable_prefixes += last_block_parameter_prefixes(model)
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        parameter.requires_grad = name.startswith(trainable_prefixes)
        if parameter.requires_grad:
            trainable += parameter.numel()
    return total, trainable


def build_optimizer_parameter_groups(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    last_block_learning_rate: float | None = None,
) -> list[dict[str, object]]:
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named_parameters:
        return []
    if last_block_learning_rate is None:
        return [{"params": [parameter for _, parameter in named_parameters]}]

    last_prefixes = last_block_parameter_prefixes(model)
    last_parameters = [
        parameter for name, parameter in named_parameters if name.startswith(last_prefixes)
    ]
    adapter_expert_gate_parameters = [
        parameter for name, parameter in named_parameters if not name.startswith(last_prefixes)
    ]
    if not last_parameters:
        raise ValueError("--unfreeze-last-block found no trainable last block or final_norm parameters")
    if not adapter_expert_gate_parameters:
        raise ValueError("--unfreeze-last-block requires trainable adapter/expert/gate parameters")
    return [
        {
            "name": "adapter_expert_gate",
            "params": adapter_expert_gate_parameters,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
        {
            "name": "last_block_final_norm",
            "params": last_parameters,
            "lr": last_block_learning_rate,
            "weight_decay": weight_decay,
        },
    ]


def validate_last_block_options(
    freeze_shared_backbone: bool,
    unfreeze_last_block: bool,
    last_block_learning_rate: float | None,
) -> None:
    if unfreeze_last_block and not freeze_shared_backbone:
        raise ValueError("--unfreeze-last-block requires --freeze-shared-backbone")
    if last_block_learning_rate is not None and not unfreeze_last_block:
        raise ValueError("--last-block-learning-rate requires --unfreeze-last-block")
    if unfreeze_last_block and last_block_learning_rate is None:
        raise ValueError("--unfreeze-last-block requires --last-block-learning-rate")
    if last_block_learning_rate is not None and last_block_learning_rate <= 0:
        raise ValueError("--last-block-learning-rate must be positive")


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
    pairwise_rank_weight: float = 0.0,
    teacher_protect_weight: float = 0.0,
    pairwise_margin: float = 0.2,
    pairwise_max_per_segment: int = 16,
    teacher: torch.nn.Module | None = None,
    main_loss_weight: float = 1.0,
) -> dict[str, float]:
    if main_loss_weight < 0 or pairwise_rank_weight < 0 or teacher_protect_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if pairwise_margin < 0:
        raise ValueError("pairwise margin must be non-negative")
    if pairwise_max_per_segment < 0:
        raise ValueError("pairwise_max_per_segment must be non-negative")
    if teacher_protect_weight > 0 and teacher is None:
        raise ValueError("teacher_protect_weight requires a frozen teacher model")
    model.train()
    batches = make_batches(examples, max_tokens, max_proteins, shuffle=True, rng=rng)
    disorder_weight = torch.tensor(disorder_pos_weight, dtype=torch.float32, device=device)
    aux_weight = torch.tensor(aux_pos_weight, dtype=torch.float32, device=device)
    total_loss = 0.0
    total_main = 0.0
    total_aux = 0.0
    total_gate = 0.0
    total_pairwise_rank = 0.0
    total_teacher_protect = 0.0
    total_pairwise_pairs = 0.0
    total_pairwise_segments = 0.0
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
        pairwise_rank_loss = output["disorder_logits"].new_tensor(0.0)
        pairwise_pairs = 0
        pairwise_segments = 0
        if pairwise_rank_weight > 0.0:
            pairwise_rank_loss, pairwise_pairs, pairwise_segments = masked_internal_ordered_pairwise_ranking_loss(
                logits=output["disorder_logits"],
                labels=tensors["labels"],
                auxiliary=tensors["auxiliary"],
                known_mask=tensors["known_mask"],
                margin=pairwise_margin,
                max_per_segment=pairwise_max_per_segment,
                rng=rng,
            )
        teacher_protect_loss = output["disorder_logits"].new_tensor(0.0)
        if teacher_protect_weight > 0.0:
            assert teacher is not None
            teacher.eval()
            with torch.no_grad():
                teacher_output = teacher(tensors["features"], tensors["pad_mask"])
            teacher_protect_loss = masked_teacher_logit_huber_loss(
                student_logits=output["disorder_logits"],
                teacher_logits=teacher_output["disorder_logits"],
                labels=tensors["labels"],
                auxiliary=tensors["auxiliary"],
                known_mask=tensors["known_mask"],
            )
        loss = (
            main_loss_weight * main_loss
            + aux_loss_weight * aux_loss
            + gate_loss_weight * gate_loss
            + pairwise_rank_weight * pairwise_rank_loss
            + teacher_protect_weight * teacher_protect_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        known = float(torch.sum(tensors["known_mask"]).detach().cpu())
        total_loss += float(loss.detach().cpu()) * known
        total_main += float(main_loss.detach().cpu()) * known
        total_aux += float(aux_loss.detach().cpu()) * known
        total_gate += float(gate_loss.detach().cpu()) * known
        total_pairwise_rank += float(pairwise_rank_loss.detach().cpu()) * known
        total_teacher_protect += float(teacher_protect_loss.detach().cpu()) * known
        total_pairwise_pairs += float(pairwise_pairs)
        total_pairwise_segments += float(pairwise_segments)
        total_known += known
    return {
        "train_loss": total_loss / total_known,
        "train_main_loss": total_main / total_known,
        "train_aux_loss": total_aux / total_known,
        "train_gate_loss": total_gate / total_known,
        "train_pairwise_rank_loss": total_pairwise_rank / total_known,
        "train_teacher_protect_loss": total_teacher_protect / total_known,
        "train_pairwise_pairs": total_pairwise_pairs,
        "train_pairwise_segments": total_pairwise_segments,
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
        "train_pairwise_rank_loss",
        "train_teacher_protect_loss",
        "train_pairwise_pairs",
        "train_pairwise_segments",
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
            "FactorizedRegionAdapterMoETCN",
            "factorized_region_adapter_moe_tcn",
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
    parser.add_argument("--unfreeze-last-block", action="store_true")
    parser.add_argument("--select-initial-checkpoint", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--last-block-learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--main-loss-weight", type=float, default=1.0)
    parser.add_argument("--aux-loss-weight", type=float, default=0.20)
    parser.add_argument("--gate-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--pairwise-rank-weight",
        "--rank-loss-weight",
        "--rank-weight",
        dest="pairwise_rank_weight",
        type=float,
        default=0.0,
        help="weight for Internal-positive versus ordered-negative margin ranking (default: 0)",
    )
    parser.add_argument(
        "--teacher-protect-weight",
        "--protect-loss-weight",
        "--protect-weight",
        dest="teacher_protect_weight",
        type=float,
        default=0.0,
        help="weight for frozen-init-teacher logit Huber protection (default: 0)",
    )
    parser.add_argument(
        "--pairwise-margin",
        "--rank-margin",
        dest="pairwise_margin",
        type=float,
        default=0.2,
        help="required Internal-over-ordered logit margin",
    )
    parser.add_argument(
        "--pairwise-max-per-segment",
        "--rank-max-per-segment",
        "--max-rank-per-segment",
        dest="pairwise_max_per_segment",
        type=int,
        default=16,
        help="maximum sampled positives and ordered negatives per Internal segment; 0 means uncapped",
    )
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
    validate_last_block_options(
        freeze_shared_backbone=args.freeze_shared_backbone,
        unfreeze_last_block=args.unfreeze_last_block,
        last_block_learning_rate=args.last_block_learning_rate,
    )
    if args.main_loss_weight < 0 or args.pairwise_rank_weight < 0 or args.teacher_protect_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if args.pairwise_margin < 0:
        raise ValueError("--pairwise-margin must be non-negative")
    if args.pairwise_max_per_segment < 0:
        raise ValueError("--pairwise-max-per-segment must be non-negative")
    if args.teacher_protect_weight > 0 and args.init_from_checkpoint is None:
        raise ValueError("--teacher-protect-weight requires --init-from-checkpoint")
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
    teacher: torch.nn.Module | None = None
    teacher_loaded_init_keys: list[str] = []
    if args.teacher_protect_weight > 0.0:
        teacher = build_model(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            dropout=args.dropout,
            adapter_dim=args.adapter_dim,
            gate_temperature=args.gate_temperature,
        ).to(device)
        assert args.init_from_checkpoint is not None
        teacher_loaded_init_keys = initialize_model_from_checkpoint(teacher, args.init_from_checkpoint, device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
    total_parameters, trainable_parameters = count_trainable_parameters(model)
    if args.freeze_shared_backbone:
        total_parameters, trainable_parameters = freeze_shared_backbone_parameters(
            model,
            unfreeze_last_block=args.unfreeze_last_block,
        )
    trainable_parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameter_list:
        raise ValueError("no trainable parameters remain after freezing")
    if args.unfreeze_last_block:
        optimizer_parameter_groups = build_optimizer_parameter_groups(
            model=model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            last_block_learning_rate=args.last_block_learning_rate,
        )
        optimizer = torch.optim.AdamW(optimizer_parameter_groups)
        optimizer_group_metadata = [
            {
                "name": str(group["name"]),
                "learning_rate": float(group["lr"]),
                "parameter_count": int(sum(parameter.numel() for parameter in group["params"])),
            }
            for group in optimizer_parameter_groups
        ]
    else:
        optimizer = torch.optim.AdamW(trainable_parameter_list, lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer_group_metadata = [
            {
                "name": "all_trainable",
                "learning_rate": args.learning_rate,
                "parameter_count": trainable_parameters,
            }
        ]
    rng = random.Random(args.seed)

    last_block_prefixes = last_block_parameter_prefixes(model) if args.unfreeze_last_block else ()

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
            "gate_factorization": (
                "length_location"
                if model_type == "FactorizedRegionAdapterMoETCN"
                else "four_way"
            ),
            "length_gate_names": ["sdr", "ldr"] if model_type == "FactorizedRegionAdapterMoETCN" else None,
            "location_gate_names": (
                ["terminal_idr", "internal_idr"]
                if model_type == "FactorizedRegionAdapterMoETCN"
                else None
            ),
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "dropout": args.dropout,
            "adapter_dim": args.adapter_dim,
            "gate_temperature": args.gate_temperature,
            "init_from_checkpoint": str(args.init_from_checkpoint) if args.init_from_checkpoint else None,
            "loaded_init_keys": len(loaded_init_keys),
            "freeze_shared_backbone": args.freeze_shared_backbone,
            "unfreeze_last_block": args.unfreeze_last_block,
            "last_block_learning_rate": args.last_block_learning_rate,
            "last_block_parameter_prefixes": list(last_block_prefixes),
            "optimizer_parameter_groups": optimizer_group_metadata,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "auxiliary_names": list(AUXILIARY_NAMES),
            "main_loss_weight": args.main_loss_weight,
            "aux_loss_weight": args.aux_loss_weight,
            "gate_loss_weight": args.gate_loss_weight,
            "pairwise_rank_weight": args.pairwise_rank_weight,
            "teacher_protect_weight": args.teacher_protect_weight,
            "pairwise_margin": args.pairwise_margin,
            "pairwise_max_per_segment": args.pairwise_max_per_segment,
            "pairwise_positive_region": "internal_idr",
            "pairwise_negative_region": "ordered_only",
            "pairwise_sampling": "equal_weight_per_internal_disorder_segment",
            "teacher_protection_regions": ["ordered", "terminal_idr"],
            "teacher_unprotected_regions": ["internal_idr"],
            "teacher_checkpoint": str(args.init_from_checkpoint)
            if args.teacher_protect_weight > 0.0 and args.init_from_checkpoint
            else None,
            "teacher_loaded_init_keys": len(teacher_loaded_init_keys),
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
            "train_pairwise_rank_loss": math.nan,
            "train_teacher_protect_loss": math.nan,
            "train_pairwise_pairs": 0.0,
            "train_pairwise_segments": 0.0,
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
            pairwise_rank_weight=args.pairwise_rank_weight,
            teacher_protect_weight=args.teacher_protect_weight,
            pairwise_margin=args.pairwise_margin,
            pairwise_max_per_segment=args.pairwise_max_per_segment,
            teacher=teacher,
            main_loss_weight=args.main_loss_weight,
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
