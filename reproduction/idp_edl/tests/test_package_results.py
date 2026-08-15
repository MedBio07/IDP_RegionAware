import csv
import gzip
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.package_idp_edl_reproduction import (
    ARCHIVE_MEMBERS,
    COMPONENT_SUMMARY_COLUMNS,
    DATASETS,
    PACKAGE_VERSION,
    PUBLIC_RESIDUE_COLUMNS,
    SOURCE_RESIDUE_COLUMNS,
    SUMMARY_COLUMNS,
    PackageValidationError,
    package_results,
)


class PackageResultsV2Test(unittest.TestCase):
    RECORDS = {
        "SL329": [("Alpha", "ACDE", [0, 1, -1, 0]), ("Beta", "FG", [1, 0])],
        "MXD494": [("MxOne", "HIJ", [1, 0, 1]), ("MxTwo", "KL", [0, -1])],
        "DISORDER723": [("DisOne", "MNP", [0, 1, 0]), ("DisTwo", "QRST", [-1, 1, 0, 1])],
    }
    FINAL_SCORES = {
        "Alpha": [0.1, 0.9, 0.7, 0.4],
        "Beta": [0.8, 0.2],
        "MxOne": [0.9, 0.1, 0.8],
        "MxTwo": [0.2, 0.6],
        "DisOne": [0.2, 0.8, 0.3],
        "DisTwo": [0.6, 0.1, 0.9, 0.8],
    }

    def _write_fastas(self, root):
        paths = {}
        for dataset in DATASETS:
            path = root / "{}_test.fasta".format(dataset)
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for protein_id, sequence, labels in self.RECORDS[dataset]:
                    handle.write(">{}\n{}\n{}\n".format(protein_id, sequence, labels))
            paths[dataset] = path
        return paths

    def _source_rows(self, dataset):
        rows = []
        for record_index, (protein_id, sequence, labels) in enumerate(self.RECORDS[dataset]):
            final_scores = self.FINAL_SCORES[protein_id]
            for position, (amino_acid, label, final_score) in enumerate(
                zip(sequence, labels, final_scores), 1
            ):
                rows.append(
                    {
                        "protein_id": protein_id.lower() if record_index == 0 else protein_id,
                        "residue_index": str(position),
                        "amino_acid": amino_acid,
                        "true_label": str(label),
                        "evaluable": str(label in (0, 1)).lower(),
                        "idp_edl_g_score": "{:.3f}".format(min(1.0, final_score + 0.05)),
                        "idp_edl_l_score": "{:.3f}".format(max(0.0, final_score - 0.05)),
                        "idp_edl_s_score": "{:.3f}".format(final_score),
                        "idp_edl_score": "{:.3f}".format(final_score),
                    }
                )
        return rows

    def _write_archive(self, root, mutate=None):
        archive_path = root / "idp_edl_predictions.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("IDP-EDL predictions/", "")
            for dataset in DATASETS:
                rows = self._source_rows(dataset)
                if mutate is not None:
                    mutate(dataset, rows)
                handle = io.StringIO(newline="")
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(SOURCE_RESIDUE_COLUMNS),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(reversed(rows))
                archive.writestr(
                    "IDP-EDL predictions/{}".format(ARCHIVE_MEMBERS[dataset]),
                    handle.getvalue(),
                )
                archive.writestr(
                    "__MACOSX/IDP-EDL predictions/._{}".format(ARCHIVE_MEMBERS[dataset]),
                    b"macOS metadata",
                )
        return archive_path

    def _package_kwargs(self, root, archive, fastas, output):
        return {
            "archive": archive,
            "sl329_fasta": fastas["SL329"],
            "mxd494_fasta": fastas["MXD494"],
            "disorder723_fasta": fastas["DISORDER723"],
            "output_dir": output,
            "project_root": root,
        }

    def test_complete_zip_validation_and_deterministic_public_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fastas = self._write_fastas(root)
            archive = self._write_archive(root)
            output_one = root / "package_one"
            output_two = root / "package_two"

            manifest_one = package_results(**self._package_kwargs(root, archive, fastas, output_one))
            manifest_two = package_results(**self._package_kwargs(root, archive, fastas, output_two))

            expected_files = {
                "sl329_residue_predictions.tsv.gz",
                "mxd494_residue_predictions.tsv.gz",
                "disorder723_residue_predictions.tsv.gz",
                "summary.tsv",
                "summary.json",
                "component_summary.tsv",
                "paper_comparison.tsv",
                "manifest.json",
            }
            self.assertEqual({path.name for path in output_one.iterdir()}, expected_files)
            for name in expected_files:
                self.assertEqual((output_one / name).read_bytes(), (output_two / name).read_bytes())
            self.assertEqual(manifest_one, manifest_two)

            with gzip.open(output_one / "sl329_residue_predictions.tsv.gz", "rt", encoding="utf-8", newline="") as handle:
                public_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(tuple(public_rows[0]), PUBLIC_RESIDUE_COLUMNS)
            self.assertEqual(len(public_rows), 6)
            self.assertEqual(public_rows[0]["protein_id"], "Alpha")
            self.assertNotIn("amino_acid", public_rows[0])
            self.assertNotIn("true_label", public_rows[0])
            self.assertNotIn("evaluable", public_rows[0])
            self.assertNotIn("prediction", public_rows[0])

            with (output_one / "summary.tsv").open("r", encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(tuple(summary_rows[0]), SUMMARY_COLUMNS)
            sl_summary = next(row for row in summary_rows if row["dataset"] == "SL329")
            self.assertEqual(
                {key: sl_summary[key] for key in ("proteins", "rows", "evaluated", "unknown", "positive", "negative", "tp", "tn", "fp", "fn")},
                {"proteins": "2", "rows": "6", "evaluated": "5", "unknown": "1", "positive": "2", "negative": "3", "tp": "2", "tn": "3", "fp": "0", "fn": "0"},
            )
            self.assertEqual(float(sl_summary["auc"]), 1.0)
            self.assertEqual(float(sl_summary["aupr"]), 1.0)

            with (output_one / "component_summary.tsv").open("r", encoding="utf-8", newline="") as handle:
                component_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(tuple(component_rows[0]), COMPONENT_SUMMARY_COLUMNS)
            self.assertEqual(len(component_rows), 12)
            final_component = next(
                row for row in component_rows if row["dataset"] == "SL329" and row["predictor"] == "idp_edl"
            )
            self.assertEqual(final_component["auc"], sl_summary["auc"])

            summary_json = json.loads((output_one / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_json["package_version"], PACKAGE_VERSION)
            self.assertEqual(len(summary_json["datasets"]), 3)

            manifest = json.loads((output_one / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["package_version"], "v2")
            self.assertEqual(manifest["protocol"]["source_residue_columns"], list(SOURCE_RESIDUE_COLUMNS))
            self.assertEqual(manifest["protocol"]["public_residue_columns"], list(PUBLIC_RESIDUE_COLUMNS))
            self.assertEqual(manifest["archive"]["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
            with zipfile.ZipFile(archive) as source_zip:
                for dataset in DATASETS:
                    source_name = "IDP-EDL predictions/{}".format(ARCHIVE_MEMBERS[dataset])
                    info = source_zip.getinfo(source_name)
                    member = manifest["archive"]["members"][ARCHIVE_MEMBERS[dataset]]
                    self.assertEqual(member["crc"], info.CRC)
                    self.assertEqual(member["size"], info.file_size)
                    self.assertEqual(member["filename"], ARCHIVE_MEMBERS[dataset])
            self.assertEqual(manifest["archive"]["member_count"], 7)
            self.assertEqual(manifest["archive"]["source_member_count"], 3)
            public_name = "sl329_residue_predictions.tsv.gz"
            self.assertEqual(
                manifest["outputs"][public_name]["sha256"],
                hashlib.sha256((output_one / public_name).read_bytes()).hexdigest(),
            )

    def test_label_mismatch_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fastas = self._write_fastas(root)

            def mutate(dataset, rows):
                if dataset == "SL329":
                    rows[0]["true_label"] = "1"

            archive = self._write_archive(root, mutate)
            output = root / "rejected_label"
            with self.assertRaises(PackageValidationError):
                package_results(**self._package_kwargs(root, archive, fastas, output))
            self.assertFalse(output.exists())

    def test_position_mismatch_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fastas = self._write_fastas(root)

            def mutate(dataset, rows):
                if dataset == "MXD494":
                    del rows[-1]

            archive = self._write_archive(root, mutate)
            output = root / "rejected_position"
            with self.assertRaises(PackageValidationError):
                package_results(**self._package_kwargs(root, archive, fastas, output))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
