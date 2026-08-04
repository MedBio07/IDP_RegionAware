#!/usr/bin/env python3
"""Analyze RegionAdapterMoETCN gate specialization by disorder-region strata."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from annotate_disorder_regions import disorder_content_bin, is_terminal_segment, iter_disorder_segments, length_bin
from evaluate_disorder_predictions import parse_labeled_fasta
from models.features import feature_matrix
from predict_sequence_disorder_model import build_model, metadata_features


EXPERTS = ("sdr", "ldr", "terminal_idr", "internal_idr")
DATASETS = {
    "DM1229_Validation": ROOT / "data/DM1229_Validation.fasta",
    "SL329": ROOT / "data/SL329_test.fasta",
    "MXD494": ROOT / "data/MXD494_test.fasta",
    "DISORDER723": ROOT / "data/DISORDER723_test.fasta",
}


@dataclass
class GateAccumulator:
    residues: int = 0
    positives: int = 0
    gate_sum: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    argmax_counts: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.int64))
    entropy_sum: float = 0.0
    target_gate_sum: float = 0.0
    target_gate_count: int = 0

    def add(self, gate: np.ndarray, label: int, target_indices: tuple[int, ...]) -> None:
        self.residues += 1
        self.positives += int(label == 1)
        self.gate_sum += gate
        self.argmax_counts[int(np.argmax(gate))] += 1
        clipped = np.clip(gate, 1.0e-8, 1.0)
        self.entropy_sum += float(-np.sum(clipped * np.log(clipped)) / math.log(len(EXPERTS)))
        for index in target_indices:
            self.target_gate_sum += float(gate[index])
            self.target_gate_count += 1

    def row(self, dataset: str, stratum_group: str, stratum: str) -> dict[str, object]:
        row: dict[str, object] = {
            "dataset": dataset,
            "stratum_group": stratum_group,
            "stratum": stratum,
            "residues": self.residues,
            "positives": self.positives,
            "positive_fraction": self.positives / self.residues if self.residues else math.nan,
            "mean_gate_entropy": self.entropy_sum / self.residues if self.residues else math.nan,
            "mean_target_gate_weight": (
                self.target_gate_sum / self.target_gate_count if self.target_gate_count else math.nan
            ),
        }
        for index, expert in enumerate(EXPERTS):
            row[f"mean_gate_{expert}"] = self.gate_sum[index] / self.residues if self.residues else math.nan
            row[f"argmax_fraction_{expert}"] = self.argmax_counts[index] / self.residues if self.residues else math.nan
        return row


def format_value(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.6f}"
    return value


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fieldnames})


def load_models(paths: list[Path], device: torch.device) -> tuple[list[torch.nn.Module], list[str], Path | None]:
    models: list[torch.nn.Module] = []
    feature_names: list[str] | None = None
    embedding_dir: Path | None = None
    for path in paths:
        checkpoint = torch.load(path, map_location=device)
        metadata = checkpoint["metadata"]
        model = build_model(metadata).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        current_features = metadata_features(metadata["features"])
        current_embedding_dir = Path(str(metadata["embedding_dir"])) if metadata.get("embedding_dir") else None
        if feature_names is None:
            feature_names = current_features
            embedding_dir = current_embedding_dir
        elif feature_names != current_features or embedding_dir != current_embedding_dir:
            raise ValueError("all checkpoints must use the same feature set and embedding directory")
        models.append(model)
    if feature_names is None:
        raise ValueError("at least one checkpoint is required")
    return models, feature_names, embedding_dir


def residue_region_maps(labels: list[int], sequence_length: int) -> tuple[dict[int, str], dict[int, str]]:
    length_type_by_pos: dict[int, str] = {}
    location_by_pos: dict[int, str] = {}
    for start, end, segment_length in iter_disorder_segments(labels):
        length_type = "sdr" if segment_length < 30 else "ldr"
        location = "terminal_idr" if is_terminal_segment(start, end, sequence_length) else "internal_idr"
        for position in range(start, end + 1):
            length_type_by_pos[position] = length_type
            location_by_pos[position] = location
    return length_type_by_pos, location_by_pos


def target_indices(label: int, length_type: str, location: str) -> tuple[int, ...]:
    if label != 1:
        return ()
    indices: list[int] = []
    if length_type == "sdr":
        indices.append(0)
    elif length_type == "ldr":
        indices.append(1)
    if location == "terminal_idr":
        indices.append(2)
    elif location == "internal_idr":
        indices.append(3)
    return tuple(indices)


def add_group(
    accumulators: dict[tuple[str, str], GateAccumulator],
    group: str,
    stratum: str,
    gate: np.ndarray,
    label: int,
    targets: tuple[int, ...],
) -> None:
    accumulators[(group, stratum)].add(gate, label, targets)


def analyze_dataset(
    dataset: str,
    fasta: Path,
    models: list[torch.nn.Module],
    feature_names: list[str],
    embedding_dir: Path | None,
    device: torch.device,
) -> list[dict[str, object]]:
    records = parse_labeled_fasta(fasta)
    accumulators: dict[tuple[str, str], GateAccumulator] = defaultdict(GateAccumulator)

    with torch.no_grad():
        for record in records:
            sequence = str(record["sequence"])
            labels = [int(value) for value in record["labels"]]
            features = feature_matrix(sequence, feature_names, embedding_dir)
            x = torch.from_numpy(features).unsqueeze(0).to(device)
            mask = torch.ones((1, features.shape[0]), dtype=torch.float32, device=device)
            gates = []
            for model in models:
                output = model(x, mask)
                if "gate_weights" not in output:
                    raise ValueError("checkpoint model does not expose gate_weights")
                gates.append(output["gate_weights"].squeeze(0).detach().cpu().numpy())
            mean_gates = np.mean(np.stack(gates, axis=0), axis=0)

            sequence_length = len(sequence)
            known = sum(label in (0, 1) for label in labels)
            disordered = sum(label == 1 for label in labels)
            protein_length_bin = length_bin(sequence_length)
            protein_disorder_bin = disorder_content_bin(disordered, known)
            length_type_by_pos, location_by_pos = residue_region_maps(labels, sequence_length)

            for position, (label, gate) in enumerate(zip(labels, mean_gates), start=1):
                if label == -1:
                    continue
                length_type = length_type_by_pos.get(position, "ordered")
                location = location_by_pos.get(position, "ordered")
                targets = target_indices(label, length_type, location)

                add_group(accumulators, "overall", "all_known", gate, label, targets)
                add_group(accumulators, "label", "disordered" if label == 1 else "ordered", gate, label, targets)
                add_group(accumulators, "protein_length_bin", protein_length_bin, gate, label, targets)
                add_group(accumulators, "protein_disorder_content_bin", protein_disorder_bin, gate, label, targets)
                if label == 1:
                    add_group(accumulators, "positive_region_length_type", length_type.upper(), gate, label, targets)
                    add_group(
                        accumulators,
                        "positive_region_location",
                        "terminal" if location == "terminal_idr" else "internal",
                        gate,
                        label,
                        targets,
                    )
                    add_group(
                        accumulators,
                        "positive_region_combined",
                        f"{length_type.upper()}_{'terminal' if location == 'terminal_idr' else 'internal'}",
                        gate,
                        label,
                        targets,
                    )

    rows = [
        accumulator.row(dataset, group, stratum)
        for (group, stratum), accumulator in sorted(accumulators.items())
        if accumulator.residues > 0
    ]
    return rows


def row_lookup(rows: list[dict[str, object]], dataset: str, group: str, stratum: str) -> dict[str, object]:
    matches = [
        row
        for row in rows
        if row["dataset"] == dataset and row["stratum_group"] == group and row["stratum"] == stratum
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {dataset}/{group}/{stratum}; found {len(matches)}")
    return matches[0]


def write_focus_table(path: Path, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    focus_specs = [
        ("DM1229_Validation", "label", "ordered", "Validation ordered"),
        ("DM1229_Validation", "positive_region_length_type", "SDR", "Validation SDR"),
        ("DM1229_Validation", "positive_region_length_type", "LDR", "Validation LDR"),
        ("DM1229_Validation", "positive_region_location", "terminal", "Validation terminal IDR"),
        ("DM1229_Validation", "positive_region_location", "internal", "Validation internal IDR"),
        ("SL329", "positive_region_location", "internal", "SL329 internal IDR"),
        ("MXD494", "positive_region_location", "internal", "MXD494 internal IDR"),
        ("DISORDER723", "positive_region_location", "internal", "DISORDER723 internal IDR"),
        ("DISORDER723", "positive_region_length_type", "LDR", "DISORDER723 LDR"),
    ]
    focus: list[dict[str, object]] = []
    for dataset, group, stratum, label in focus_specs:
        row = row_lookup(rows, dataset, group, stratum)
        focus.append({"display_stratum": label, **row})
    fields = [
        "display_stratum",
        "dataset",
        "stratum_group",
        "stratum",
        "residues",
        "positives",
        "mean_gate_sdr",
        "mean_gate_ldr",
        "mean_gate_terminal_idr",
        "mean_gate_internal_idr",
        "mean_target_gate_weight",
        "mean_gate_entropy",
        "argmax_fraction_sdr",
        "argmax_fraction_ldr",
        "argmax_fraction_terminal_idr",
        "argmax_fraction_internal_idr",
    ]
    write_rows(path, focus, fields)
    return focus


def plot_gate_specialization(path_pdf: Path, path_png: Path, focus_rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["display_stratum"]) for row in focus_rows]
    matrix = np.asarray([[float(row[f"mean_gate_{expert}"]) for expert in EXPERTS] for row in focus_rows])
    target = [float(row["mean_target_gate_weight"]) if row["mean_target_gate_weight"] != "NA" else math.nan for row in focus_rows]
    entropy = [float(row["mean_gate_entropy"]) for row in focus_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4), gridspec_kw={"width_ratios": [1.45, 1.0]})
    im = axes[0].imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.55, float(np.max(matrix))))
    axes[0].set_xticks(np.arange(len(EXPERTS)))
    axes[0].set_xticklabels(["SDR", "LDR", "Terminal", "Internal"], rotation=30, ha="right")
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].set_title("Mean MoE gate weights")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes[0].text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("Gate weight")

    y = np.arange(len(labels))
    axes[1].barh(y - 0.18, target, height=0.34, label="Target gate weight", color="#2563eb")
    axes[1].barh(y + 0.18, entropy, height=0.34, label="Gate entropy", color="#16a34a")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlim(0.0, 1.0)
    axes[1].invert_yaxis()
    axes[1].set_title("Alignment and sharpness")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("Value")
    fig.tight_layout()
    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_pdf)
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, object]], focus_rows: list[dict[str, object]]) -> None:
    validation_ordered = row_lookup(rows, "DM1229_Validation", "label", "ordered")
    validation_sdr = row_lookup(rows, "DM1229_Validation", "positive_region_length_type", "SDR")
    validation_terminal = row_lookup(rows, "DM1229_Validation", "positive_region_location", "terminal")
    validation_internal = row_lookup(rows, "DM1229_Validation", "positive_region_location", "internal")
    sl329_internal = row_lookup(rows, "SL329", "positive_region_location", "internal")
    mxd494_internal = row_lookup(rows, "MXD494", "positive_region_location", "internal")
    disorder_internal = row_lookup(rows, "DISORDER723", "positive_region_location", "internal")
    disorder_ldr = row_lookup(rows, "DISORDER723", "positive_region_length_type", "LDR")

    def f(row: dict[str, object], key: str) -> float:
        return float(row[key])

    text = f"""# P5.7 RegionAdapterMoETCN Gate Mechanism Analysis

Date: 2026-08-04

## Scope

This analysis uses the P4.7/P4.8 warm-start three-seed `RegionAdapterMoETCN` ensemble and summarizes the mean residue-level MoE gate weights over DM1229 validation, SL329, MXD494 and DISORDER723. Gate experts correspond to SDR, LDR, terminal IDR and internal IDR adapters. Results are descriptive mechanism evidence, not a causal intervention.

## Key Observations

- The learned gate is not a clean one-hot biological region classifier. Ordered validation residues have the highest mean internal-IDR gate component (`mean_gate_internal_idr={f(validation_ordered, 'mean_gate_internal_idr'):.6f}`), so the internal expert cannot be interpreted directly as an internal-IDR detector.
- The clearest positive alignment is on DM1229 validation SDR and terminal-IDR residues: SDR residues have `mean_gate_sdr={f(validation_sdr, 'mean_gate_sdr'):.6f}` and terminal-IDR residues have `mean_gate_terminal_idr={f(validation_terminal, 'mean_gate_terminal_idr'):.6f}`.
- Validation internal-IDR residues still have above-random target gate mass (`mean_target_gate_weight={f(validation_internal, 'mean_target_gate_weight'):.6f}` versus a four-expert random baseline of 0.25), but their largest mean gate component is SDR rather than internal.
- External internal-IDR routing is heterogeneous: target gate mass is `{f(sl329_internal, 'mean_target_gate_weight'):.6f}` on SL329, `{f(mxd494_internal, 'mean_target_gate_weight'):.6f}` on MXD494 and `{f(disorder_internal, 'mean_target_gate_weight'):.6f}` on DISORDER723.
- DISORDER723 LDR residues increase the LDR expert contribution (`mean_gate_ldr={f(disorder_ldr, 'mean_gate_ldr'):.6f}`), but the gate remains distributed rather than sharply specialized.

## Manuscript Interpretation

The gate analysis supports only a cautious auxiliary mechanism claim: warm-start adapters learn region-shifted routing preferences, especially for validation SDR and terminal-IDR residues, but the gate should not be described as a direct biological region classifier. The primary evidence for P4.8 should remain empirical: P4.8 preserves aggregate benchmark/NR25 performance and gives its largest practical gain on DISORDER723 internal IDRs.

## Generated Files

- `results/p5_7/P5_7_REGION_ADAPTER_MOE_GATE_SUMMARY.tsv`
- `results/p5_7/P5_7_REGION_ADAPTER_MOE_GATE_FOCUS.tsv`
- `figures/p5_7/P5_7_REGION_ADAPTER_MOE_GATE_SPECIALIZATION.pdf`
- `figures/p5_7/P5_7_REGION_ADAPTER_MOE_GATE_SPECIALIZATION.png`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=Path,
        nargs="+",
        default=[
            ROOT / "models/p4_7_region_adapter_moe_tcn_esm2_t33_gate002_warm_seed1.pt",
            ROOT / "models/p4_7_region_adapter_moe_tcn_esm2_t33_gate002_warm_seed2.pt",
            ROOT / "models/p4_7_region_adapter_moe_tcn_esm2_t33_gate002_warm_seed3.pt",
        ],
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    models, feature_names, embedding_dir = load_models(args.models, device)

    rows: list[dict[str, object]] = []
    for dataset, fasta in DATASETS.items():
        rows.extend(analyze_dataset(dataset, fasta, models, feature_names, embedding_dir, device))

    fields = [
        "dataset",
        "stratum_group",
        "stratum",
        "residues",
        "positives",
        "positive_fraction",
        "mean_gate_sdr",
        "mean_gate_ldr",
        "mean_gate_terminal_idr",
        "mean_gate_internal_idr",
        "mean_gate_entropy",
        "mean_target_gate_weight",
        "argmax_fraction_sdr",
        "argmax_fraction_ldr",
        "argmax_fraction_terminal_idr",
        "argmax_fraction_internal_idr",
    ]
    result_dir = ROOT / "results/p5_7"
    figure_dir = ROOT / "figures/p5_7"
    write_rows(result_dir / "P5_7_REGION_ADAPTER_MOE_GATE_SUMMARY.tsv", rows, fields)
    focus_rows = write_focus_table(result_dir / "P5_7_REGION_ADAPTER_MOE_GATE_FOCUS.tsv", rows)
    plot_gate_specialization(
        figure_dir / "P5_7_REGION_ADAPTER_MOE_GATE_SPECIALIZATION.pdf",
        figure_dir / "P5_7_REGION_ADAPTER_MOE_GATE_SPECIALIZATION.png",
        focus_rows,
    )
    write_report(ROOT / "reports/P5_7_REGION_ADAPTER_MOE_GATE_MECHANISM.md", rows, focus_rows)


if __name__ == "__main__":
    main()
