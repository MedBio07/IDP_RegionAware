import csv
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from reproduction.idp_edl.metrics import compute_binary_metrics
from scripts.package_idp_edl_reproduction import (
    PAPER_VALUES,
    PUBLIC_RESIDUE_COLUMNS,
    SOURCE_RESIDUE_COLUMNS,
    SOURCE_SUMMARY_COLUMNS,
    package_results,
)


class PackageResultsEndToEndTest(unittest.TestCase):
    def _write_author_pair(self, root):
        sequence_path = root / "sample_sequences.fasta"
        label_path = root / "sample_labels.fasta"
        records = [
            ("sl_a", "sl329", "ACDE", "0121"),
            ("sl_b", "sl329", "FGH", "001"),
            ("sl_long", "sl329", "A" * 1025, "0" * 1025),
            ("mx_a", "mxd494", "ACDE", "0110"),
            ("mx_b", "mxd494", "FGH", "001"),
            ("dis_a", "disorder723", "ACDE", "0101"),
            ("dis_b", "disorder723", "FGH", "001"),
        ]
        with sequence_path.open("w", encoding="utf-8", newline="\n") as sequences:
            for identifier, dataset, sequence, _ in records:
                sequences.write(">%s %s\n%s\n" % (identifier, dataset, sequence))
        with label_path.open("w", encoding="utf-8", newline="\n") as labels:
            for identifier, dataset, _, label in records:
                labels.write(">%s %s\n%s\n" % (identifier, dataset, label))
        return sequence_path, label_path, records

    def _write_source_outputs(self, root, records):
        by_dataset = {}
        for identifier, dataset, sequence, labels in records:
            by_dataset.setdefault(dataset, []).append((identifier, sequence, labels))

        scores_by_protein = {
            "sl_a": [0.2, 0.8, 0.7, 0.6],
            "sl_b": [0.1, 0.2, 0.8],
            "sl_long": [0.1] * 1023,
            "mx_a": [0.2, 0.8, 0.7, 0.6],
            "mx_b": [0.1, 0.2, 0.8],
            "dis_a": [0.2, 0.8, 0.3, 0.9],
            "dis_b": [0.1, 0.2, 0.8],
        }
        outputs_root = root / "reproduction" / "idp_edl" / "outputs"
        for dataset, proteins in by_dataset.items():
            directory = outputs_root / (dataset + "_fp32_batch4")
            directory.mkdir(parents=True, exist_ok=True)
            rows = []
            for identifier, sequence, labels in proteins:
                retained = min(len(sequence), 1023)
                truncated = len(sequence) - retained
                scores = scores_by_protein[identifier]
                self.assertEqual(len(scores), retained)
                for position in range(1, retained + 1):
                    score = scores[position - 1]
                    label = labels[position - 1]
                    rows.append(
                        {
                            "dataset": dataset,
                            "protein_id": identifier,
                            "header": identifier + " " + dataset,
                            "position": position,
                            "aa": sequence[position - 1],
                            "model_aa": sequence[position - 1],
                            "label": label,
                            "included_in_metrics": int(label in "01"),
                            "prediction": int(score > 0.5),
                            "score": score,
                            "max_length": 1024,
                            "truncated_residues": truncated,
                        }
                    )

            residue_path = directory / "idp_edl_residue_predictions.tsv"
            with residue_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(SOURCE_RESIDUE_COLUMNS),
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

            labels_for_metrics = [int(row["label"]) for row in rows]
            scores_for_metrics = [float(row["score"]) for row in rows]
            metrics = compute_binary_metrics(labels_for_metrics, scores_for_metrics)
            summary = {
                "dataset": dataset,
                "status": "predicted",
                "proteins": len(proteins),
                "original_residues": sum(len(sequence) for _, sequence, _ in proteins),
                "retained_residues": len(rows),
                "truncated_residues": sum(max(0, len(sequence) - 1023) for _, sequence, _ in proteins),
                "label_0_retained": labels_for_metrics.count(0),
                "label_1_retained": labels_for_metrics.count(1),
                "label_2_excluded": labels_for_metrics.count(2),
                "evaluated_residues": sum(label in (0, 1) for label in labels_for_metrics),
                "accuracy": metrics["accuracy"],
                "auc": metrics["auc"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "bacc": metrics["bacc"],
                "mcc": metrics["mcc"],
                "fmax": metrics["fmax"],
                "threshold": 0.5,
            }
            summary_tsv_path = directory / "idp_edl_summary.tsv"
            with summary_tsv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(SOURCE_SUMMARY_COLUMNS),
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(summary)
            summary_json = {
                "metadata": {
                    "batch_size": 4,
                    "device": "cpu",
                    "eos_tokens": 1,
                    "label_2_policy": "retained in residue TSV and excluded from metrics",
                    "max_length": 1024,
                    "model_loaded": True,
                    "source": "synthetic author sample pair",
                },
                "datasets": [{**summary, "tp": metrics["tp"], "tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"]}],
            }
            with (directory / "idp_edl_summary.json").open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(summary_json, handle, indent=2, sort_keys=True)
                handle.write("\n")
        return outputs_root

    def test_pack_validates_and_publishes_redacted_deterministic_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence_path, label_path, records = self._write_author_pair(root)
            outputs_root = self._write_source_outputs(root, records)
            output_one = root / "package_one"
            output_two = root / "package_two"

            manifest_one = package_results(
                outputs_root=outputs_root,
                output_dir=output_one,
                sample_sequences=sequence_path,
                sample_labels=label_path,
                project_root=root,
            )
            package_results(
                outputs_root=outputs_root,
                output_dir=output_two,
                sample_sequences=sequence_path,
                sample_labels=label_path,
                project_root=root,
            )

            self.assertEqual(
                (output_one / "sl329_residue_predictions.tsv.gz").read_bytes(),
                (output_two / "sl329_residue_predictions.tsv.gz").read_bytes(),
            )
            self.assertEqual(
                (output_one / "summary.tsv").read_bytes(),
                (output_two / "summary.tsv").read_bytes(),
            )

            self.assertFalse((output_one / "sl329_residue_predictions.tsv").exists())
            with gzip.open(
                output_one / "sl329_residue_predictions.tsv.gz",
                "rt",
                encoding="utf-8",
                newline="",
            ) as handle:
                public_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(tuple(public_rows[0]), PUBLIC_RESIDUE_COLUMNS)
            self.assertNotIn("aa", public_rows[0])
            self.assertNotIn("model_aa", public_rows[0])
            self.assertNotIn("label", public_rows[0])
            self.assertNotIn("header", public_rows[0])
            self.assertEqual(len(public_rows), 4 + 3 + 1023)
            with gzip.open(output_one / "sl329_residue_predictions.tsv.gz", "rt", encoding="utf-8", newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle, delimiter="\t")), public_rows)

            with (output_one / "summary.tsv").open("r", encoding="utf-8", newline="") as handle:
                summary = {row["dataset"]: row for row in csv.DictReader(handle, delimiter="\t")}
            self.assertEqual(set(summary), {"sl329", "mxd494", "disorder723"})
            self.assertEqual(summary["sl329"]["proteins"], "3")
            self.assertEqual(summary["sl329"]["truncated_residues"], "2")
            self.assertEqual(summary["sl329"]["label_2_excluded"], "1")
            self.assertAlmostEqual(float(summary["sl329"]["auc"]), 1.0)
            self.assertAlmostEqual(float(summary["sl329"]["aupr"]), 1.0)

            manifest_data = json.loads((output_one / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest_data["protocol"]["labels_sequence_intentionally_omitted"])
            self.assertTrue(manifest_data["protocol"]["aggregate_label_counts_included"])
            self.assertTrue(manifest_data["protocol"]["full_source_tables_not_copied"])
            self.assertEqual(
                manifest_data["protocol"]["public_residue_columns"], list(PUBLIC_RESIDUE_COLUMNS)
            )
            source_path = outputs_root / "sl329_fp32_batch4" / "idp_edl_residue_predictions.tsv"
            self.assertEqual(
                manifest_data["datasets"]["sl329"]["source"]["residue_tsv"]["sha256"],
                hashlib.sha256(source_path.read_bytes()).hexdigest(),
            )
            with gzip.open(output_one / "sl329_residue_predictions.tsv.gz", "rb") as handle:
                public_content = handle.read()
            self.assertEqual(
                manifest_data["datasets"]["sl329"]["public"]["content"]["sha256"],
                hashlib.sha256(public_content).hexdigest(),
            )
            self.assertEqual(
                manifest_data["datasets"]["sl329"]["public"]["compressed"]["sha256"],
                hashlib.sha256((output_one / "sl329_residue_predictions.tsv.gz").read_bytes()).hexdigest(),
            )

            with (output_one / "paper_comparison.tsv").open("r", encoding="utf-8", newline="") as handle:
                comparison = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(comparison), 15)
            sl_auc = next(row for row in comparison if row["dataset"] == "SL329" and row["metric"] == "AUC")
            self.assertAlmostEqual(
                float(sl_auc["delta_local_minus_paper"]),
                float(sl_auc["local_recomputed"]) - PAPER_VALUES["sl329"]["AUC"],
                places=8,
            )


if __name__ == "__main__":
    unittest.main()
