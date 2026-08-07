"""Residue-level inference with explicit EOS/truncation accounting."""

from typing import Dict, List, Sequence, Tuple

import torch

from .data import LabeledExample, TruncationStats, normalize_for_prott5, truncation_stats


def prepare_token_input(tokenizer, sequence: str, max_length: int = 1024):
    normalized = normalize_for_prott5(sequence)
    stats = truncation_stats(len(normalized), max_length)
    text = " ".join(normalized)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        padding=False,
        truncation=True,
    )
    token_length = int(inputs["attention_mask"].sum().item())
    expected_token_length = stats.retained_length + stats.eos_tokens
    if token_length != expected_token_length:
        raise ValueError(
            "ProtT5 tokenizer is not residue-aligned: expected {} tokens, got {}".format(
                expected_token_length, token_length
            )
        )
    return inputs, normalized, stats


def predict_example(
    model,
    tokenizer,
    device: torch.device,
    example: LabeledExample,
    max_length: int = 1024,
) -> Tuple[List[Dict[str, object]], TruncationStats]:
    return predict_batch(model, tokenizer, device, [example], max_length)[0]


def predict_batch(
    model,
    tokenizer,
    device: torch.device,
    examples: Sequence[LabeledExample],
    max_length: int = 1024,
) -> List[Tuple[List[Dict[str, object]], TruncationStats]]:
    """Predict a padded batch using the author's notebook evaluation shape."""

    if not examples:
        return []
    normalized = [normalize_for_prott5(example.sequence) for example in examples]
    stats = [truncation_stats(len(sequence), max_length) for sequence in normalized]
    inputs = tokenizer(
        [" ".join(sequence) for sequence in normalized],
        return_tensors="pt",
        max_length=max_length,
        padding=True,
        truncation=True,
    )
    token_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    for index, (token_length, item_stats) in enumerate(zip(token_lengths, stats)):
        expected = item_stats.retained_length + item_stats.eos_tokens
        if int(token_length) != expected:
            raise ValueError(
                "ProtT5 tokenizer is not residue-aligned for batch item {}: expected {} tokens, got {}".format(
                    index, expected, int(token_length)
                )
            )

    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        probabilities = torch.softmax(logits, dim=-1)[..., 1].detach().cpu()
        predictions = logits.argmax(dim=-1).detach().cpu()

    output = []
    for batch_index, (example, model_sequence, item_stats) in enumerate(zip(examples, normalized, stats)):
        rows = []
        for position in range(item_stats.retained_length):
            label = example.labels[position] if position < len(example.labels) else ""
            rows.append(
                {
                    "dataset": example.dataset,
                    "protein_id": example.identifier,
                    "header": example.sequence_header,
                    "position": position + 1,
                    "aa": example.sequence[position],
                    "model_aa": model_sequence[position],
                    "label": label,
                    "included_in_metrics": int(label in ("0", "1")),
                    "prediction": int(predictions[batch_index, position].item()),
                    "score": float(probabilities[batch_index, position].item()),
                    "max_length": item_stats.max_length,
                    "truncated_residues": item_stats.truncated_residues,
                }
            )
        output.append((rows, item_stats))
    return output


def dry_run_example(
    example: LabeledExample, max_length: int = 1024
) -> Tuple[List[Dict[str, object]], TruncationStats]:
    """Create auditable residue rows without loading the 1.2B ProtT5 model."""

    normalized = normalize_for_prott5(example.sequence)
    stats = truncation_stats(len(normalized), max_length)
    rows = []
    for position in range(stats.retained_length):
        label = example.labels[position] if position < len(example.labels) else ""
        rows.append(
            {
                "dataset": example.dataset,
                "protein_id": example.identifier,
                "header": example.sequence_header,
                "position": position + 1,
                "aa": example.sequence[position],
                "model_aa": normalized[position],
                "label": label,
                "included_in_metrics": int(label in ("0", "1")),
                "prediction": "",
                "score": "",
                "max_length": stats.max_length,
                "truncated_residues": stats.truncated_residues,
            }
        )
    return rows, stats
