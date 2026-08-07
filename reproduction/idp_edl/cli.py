"""Command-line entry point for local IDP-EDL inference/evaluation."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from .data import (
    DATASET_ORDER,
    canonical_dataset,
    default_sample_paths,
    parse_author_samples,
    parse_sequence_fasta,
    single_sequence,
    truncate_labels,
)
from .inference import dry_run_example, predict_batch
from .metrics import compute_binary_metrics
from .official import ModelLoadError, load_idp_edl


RESIDUE_COLUMNS = (
    "dataset",
    "protein_id",
    "header",
    "position",
    "aa",
    "model_aa",
    "label",
    "included_in_metrics",
    "prediction",
    "score",
    "max_length",
    "truncated_residues",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(value: Path, root: Path) -> Path:
    return value if value.is_absolute() else root / value


def _dtype(value: str):
    if value in ("auto", "float32"):
        return torch.float32
    if value == "float16":
        raise ValueError(
            "--dtype float16 is disabled for official IDP-EDL inference: "
            "IDPModel.forward creates an FP32 h0; use --dtype float32. "
            "FP16 remains available for checkpoint conversion/storage."
        )
    raise ValueError("unsupported dtype: {}".format(value))


def _examples_from_args(args, root: Path):
    provided = sum(value is not None for value in (args.sequence, args.fasta)) + int(args.smoke)
    if provided > 1:
        raise ValueError("choose only one of --sequence, --fasta, and --smoke")
    if args.sequence is not None:
        return single_sequence(args.sequence)
    if args.fasta is not None:
        return parse_sequence_fasta(_resolve(args.fasta, root))
    if args.smoke:
        return single_sequence("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "smoke_1")

    selected = canonical_dataset(args.dataset)
    sequence_path, label_path = default_sample_paths(root)
    if args.sample_sequences is not None:
        sequence_path = _resolve(args.sample_sequences, root)
    if args.sample_labels is not None:
        label_path = _resolve(args.sample_labels, root)
    names = DATASET_ORDER if selected == "all" else (selected,)
    return parse_author_samples(sequence_path, label_path, names)


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "{:.9f}".format(value)
    return value


def _summarize(dataset: str, examples, rows, scores, max_length: int, status: str) -> Dict[str, object]:
    original_residues = sum(len(example.sequence) for example in examples)
    retained_residues = sum(
        min(len(example.sequence), max_length - 1) for example in examples
    )
    truncated_residues = original_residues - retained_residues
    labels = []
    score_values = []
    for example, example_scores in scores:
        retained_labels, _ = truncate_labels(example.labels, len(example.sequence), max_length) if example.labels else ("", None)
        labels.extend(int(value) for value in retained_labels)
        if example_scores is not None:
            score_values.extend(example_scores)
    metric_values = {}
    if labels and score_values and len(labels) == len(score_values):
        metric_values = compute_binary_metrics(labels, score_values)
    elif labels and score_values:
        raise ValueError(
            "label/score alignment failed for {}: {} labels versus {} scores".format(
                dataset, len(labels), len(score_values)
            )
        )
    else:
        metric_values = {
            "n_total": len(labels),
            "n_evaluated": sum(value in (0, 1) for value in labels),
            "n_excluded_label_2": sum(value == 2 for value in labels),
            "threshold": 0.5,
            "accuracy": None,
            "auc": None,
            "sensitivity": None,
            "specificity": None,
            "bacc": None,
            "mcc": None,
            "fmax": None,
        }
    result = {
        "dataset": dataset,
        "status": status,
        "proteins": len(examples),
        "original_residues": original_residues,
        "retained_residues": retained_residues,
        "truncated_residues": truncated_residues,
        "label_0_retained": sum(value == 0 for value in labels),
        "label_1_retained": sum(value == 1 for value in labels),
        "label_2_excluded": sum(value == 2 for value in labels),
        "evaluated_residues": sum(value in (0, 1) for value in labels),
    }
    result.update({key: value for key, value in metric_values.items() if key not in ("n_total", "n_evaluated", "n_excluded_label_2")})
    return result


def write_outputs(output_dir: Path, rows: List[Dict[str, object]], summaries: List[Dict[str, object]], metadata: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    residue_path = output_dir / "idp_edl_residue_predictions.tsv"
    with residue_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESIDUE_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key, "")) for key in RESIDUE_COLUMNS})

    summary_columns = [
        "dataset", "status", "proteins", "original_residues", "retained_residues", "truncated_residues",
        "label_0_retained", "label_1_retained", "label_2_excluded", "evaluated_residues",
        "accuracy", "auc", "sensitivity", "specificity", "bacc", "mcc", "fmax", "threshold",
    ]
    summary_path = output_dir / "idp_edl_summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: _format_value(summary.get(key, "")) for key in summary_columns})

    with (output_dir / "idp_edl_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "datasets": summaries}, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", help="all, mxd494, sl329, or disorder723")
    parser.add_argument("--sample-sequences", type=Path, help="override author sample_sequences.fasta")
    parser.add_argument("--sample-labels", type=Path, help="override author sample_labels.fasta")
    parser.add_argument("--sequence", help="single sequence without labels")
    parser.add_argument("--fasta", type=Path, help="sequence-only FASTA without labels")
    parser.add_argument("--smoke", action="store_true", help="run one short built-in sequence")
    parser.add_argument("--dry-run", action="store_true", help="parse/truncate/write outputs without ProtT5")
    parser.add_argument("--model-dir", type=Path, help="local encoder or full T5 HF directory")
    parser.add_argument("--weights-dir", type=Path, default=Path("external/IDP-EDL/model"))
    parser.add_argument("--external-root", type=Path, default=Path("external/IDP-EDL"))
    parser.add_argument("--output-dir", type=Path, default=Path("reproduction/idp_edl/outputs"))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="evaluation batch size; 4 matches the author's public notebook",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float32", "auto", "float16"),
        default="float32",
        help="runtime dtype; official IDP-EDL inference is FP32 (auto is an FP32 alias; FP16 is rejected)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = _root()
    if args.max_length < 2:
        print("ERROR: --max-length must be at least 2")
        return 2
    if args.batch_size < 1:
        print("ERROR: --batch-size must be at least 1")
        return 2
    try:
        examples = _examples_from_args(args, root)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is unavailable")
        device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

        bundle = None
        status = "dry_run" if args.dry_run else "predicted"
        if not args.dry_run:
            model_dir = args.model_dir
            if model_dir is None:
                converted = root / "external/ProtTrans/weights/prot_t5_xl_uniref50-enc-fp16"
                full = root / "external/ProtTrans/weights/prot_t5_xl_uniref50"
                model_dir = converted if converted.is_dir() else full
            bundle = load_idp_edl(
                _resolve(model_dir, root),
                _resolve(args.weights_dir, root),
                _resolve(args.external_root, root),
                device=device,
                dtype=_dtype(args.dtype),
            )

        rows = []
        score_groups = defaultdict(list)
        grouped_examples = defaultdict(list)
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            if args.dry_run:
                batch_results = [dry_run_example(example, args.max_length) for example in batch]
            else:
                batch_results = predict_batch(
                    bundle.model, bundle.tokenizer, bundle.device, batch, args.max_length
                )
            for example, (example_rows, _) in zip(batch, batch_results):
                grouped_examples[example.dataset].append(example)
                example_scores = (
                    [row["score"] for row in example_rows]
                    if not args.dry_run and example.labels
                    else None
                )
                rows.extend(example_rows)
                score_groups[example.dataset].append((example, example_scores))

        summaries = [
            _summarize(dataset, grouped_examples[dataset], rows, score_groups[dataset], args.max_length, status)
            for dataset in sorted(grouped_examples)
        ]
        metadata = {
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "eos_tokens": 1,
            "label_2_policy": "retained in residue TSV and excluded from metrics",
            "model_loaded": not args.dry_run,
            "device": str(device),
            "source": "official IDP-EDL sample pair" if not (args.sequence or args.fasta or args.smoke) else "user/smoke input",
        }
        write_outputs(_resolve(args.output_dir, root), rows, summaries, metadata)
        print("wrote {} residue rows and {} dataset summaries".format(len(rows), len(summaries)))
        if bundle is not None:
            print(json.dumps(bundle.reports, indent=2, sort_keys=True))
        return 0
    except (ValueError, FileNotFoundError, ModelLoadError, RuntimeError) as exc:
        print("ERROR: {}".format(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
