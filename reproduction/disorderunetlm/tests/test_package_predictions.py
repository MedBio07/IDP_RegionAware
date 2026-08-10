from __future__ import annotations

import csv
import gzip
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Set

import numpy as np

from scripts.package_disorderunetlm_predictions import (
    DATASETS,
    PackagingError,
    package_predictions,
)


def _npy_bytes(array: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.save(handle, array)
    return handle.getvalue()


class DisorderUnetLMPackagerEndToEndTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> Path:
        data_dir = root / "data"
        data_dir.mkdir()
        archive_path = root / "predictions.zip"
        archive_members: Dict[str, bytes] = {}

        for dataset in DATASETS:
            count = {"SL329": 329, "MXD494": 494, "DISORDER723": 723}[dataset]
            ids = ["sl_multipart" if dataset == "SL329" else dataset.lower() + "_0000"]
            ids.extend(dataset.lower() + "_{:04d}".format(index) for index in range(1, count))
            records = []
            for protein_id in ids:
                if dataset == "SL329" and protein_id == "sl_multipart":
                    sequence = "ACDE"
                    labels = [0, 1, -1, 1]
                    arrays = {
                        "{}_P1.npy".format(protein_id): np.asarray(
                            [[0.9, 0.1], [0.1, 0.9]], dtype=np.float32
                        ),
                        "{}_P2.npy".format(protein_id): np.asarray(
                            [[0.5, 0.5], [0.2, 0.8]], dtype=np.float32
                        ),
                    }
                else:
                    sequence = "AC"
                    labels = [0, 1]
                    arrays = {
                        "{}.npy".format(protein_id): np.asarray(
                            [[0.9, 0.1], [0.1, 0.9]], dtype=np.float32
                        )
                    }
                records.append((protein_id, sequence, labels))
                for filename, array in arrays.items():
                    archive_members[dataset + "/" + filename] = _npy_bytes(array)

            dataset_path = data_dir / (dataset + "_test.fasta")
            with dataset_path.open("w", encoding="utf-8", newline="\n") as handle:
                for protein_id, sequence, labels in records:
                    handle.write(">{} description\n{}\n{}\n".format(protein_id, sequence, labels))

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(archive_members):
                archive.writestr(name, archive_members[name])
        return archive_path

    def _rewrite_archive(
        self,
        source: Path,
        target: Path,
        replace: Optional[Dict[str, bytes]] = None,
        remove: Optional[Set[str]] = None,
        add: Optional[Dict[str, bytes]] = None,
    ) -> None:
        replace = replace or {}
        remove = remove or set()
        add = add or {}
        with zipfile.ZipFile(source, "r") as source_zip, zipfile.ZipFile(
            target, "w", compression=zipfile.ZIP_DEFLATED
        ) as target_zip:
            for info in source_zip.infolist():
                if info.filename in remove:
                    continue
                target_zip.writestr(info.filename, replace.get(info.filename, source_zip.read(info.filename)))
            for name in sorted(add):
                target_zip.writestr(name, add[name])

    def test_end_to_end_multipart_outputs_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self._write_fixture(root)
            output_one = root / "package_one"
            output_two = root / "package_two"

            package_predictions(archive, root / "data", output_one, project_root=root)
            package_predictions(archive, root / "data", output_two, project_root=root)

            expected_files = {
                "SL329": "sl329_residue_predictions.tsv.gz",
                "MXD494": "mxd494_residue_predictions.tsv.gz",
                "DISORDER723": "disorder723_residue_predictions.tsv.gz",
            }
            expected_output_names = set(expected_files.values()) | {
                "protein_summary.tsv.gz",
                "summary.tsv",
                "summary.json",
                "manifest.json",
                "README.md",
            }
            self.assertEqual({path.name for path in output_one.iterdir()}, expected_output_names)
            self.assertEqual(
                {path.name for path in output_two.iterdir()},
                expected_output_names,
            )
            for name in expected_output_names:
                self.assertEqual((output_one / name).read_bytes(), (output_two / name).read_bytes(), name)

            with gzip.open(output_one / expected_files["SL329"], "rt", encoding="utf-8", newline="") as handle:
                residue_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(residue_rows), 4 + 328 * 2)
            self.assertEqual(
                [(row["position"], row["score"]) for row in residue_rows[:4]],
                [("1", "0.100000001"), ("2", "0.899999976"), ("3", "0.500000000"), ("4", "0.800000012")],
            )
            self.assertEqual(residue_rows[0]["prediction"], "0")
            self.assertEqual(residue_rows[1]["prediction"], "1")
            self.assertEqual(residue_rows[2]["prediction"], "0")
            self.assertEqual(tuple(residue_rows[0]), ("dataset", "protein_id", "position", "prediction", "score"))

            with (output_one / "summary.tsv").open("r", encoding="utf-8", newline="") as handle:
                summaries = {row["dataset"]: row for row in csv.DictReader(handle, delimiter="\t")}
            self.assertEqual(set(summaries), set(DATASETS))
            self.assertEqual(summaries["SL329"]["proteins"], "329")
            self.assertEqual(summaries["SL329"]["excluded_label_minus1"], "1")
            self.assertEqual(summaries["SL329"]["evaluated_residues"], "659")
            self.assertEqual(summaries["SL329"]["tp"], "330")
            self.assertEqual(summaries["SL329"]["tn"], "329")
            self.assertEqual(summaries["SL329"]["fp"], "0")
            self.assertEqual(summaries["SL329"]["fn"], "0")
            self.assertEqual(summaries["SL329"]["auc"], "1.000000000")
            self.assertEqual(summaries["SL329"]["aupr"], "1.000000000")
            self.assertEqual(summaries["SL329"]["fmax"], "1.000000000")

            with gzip.open(output_one / "protein_summary.tsv.gz", "rt", encoding="utf-8", newline="") as handle:
                protein_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(protein_rows), 329 + 494 + 723)
            self.assertEqual(protein_rows[0]["protein_id"], "sl_multipart")
            self.assertEqual(protein_rows[0]["sequence_length"], "4")
            self.assertEqual(protein_rows[0]["source_parts"], "2")
            self.assertEqual(protein_rows[0]["predicted_disordered_residues"], "2")
            self.assertEqual(protein_rows[0]["predicted_disordered_fraction"], "0.500000000")
            self.assertEqual(
                tuple(protein_rows[0]),
                (
                    "dataset",
                    "protein_id",
                    "sequence_length",
                    "source_parts",
                    "predicted_disordered_residues",
                    "predicted_disordered_fraction",
                    "score_mean",
                    "score_std",
                    "score_min",
                    "score_max",
                ),
            )
            self.assertNotIn("label", "\t".join(protein_rows[0]))
            self.assertNotIn("tp", protein_rows[0])

            manifest = json.loads((output_one / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["datasets"]["SL329"]["archive_member_count"], 330)
            self.assertEqual(
                manifest["datasets"]["SL329"]["multipart_proteins"]["sl_multipart"],
                {
                    "members": ["SL329/sl_multipart_P1.npy", "SL329/sl_multipart_P2.npy"],
                    "part_lengths": [2, 2],
                    "concatenated_length": 4,
                },
            )
            self.assertIn("predictions but no model", manifest["caveats"][0])
            self.assertTrue(manifest["archive"]["sha256"])
            self.assertEqual(manifest["archive"]["filename"], "predictions.zip")
            self.assertNotIn("path", manifest["archive"])
            self.assertFalse(manifest["protocol"]["archive_class_column_metadata_present"])
            self.assertTrue(manifest["sources"]["SL329"]["sha256"])
            self.assertNotIn(
                "DP00072",
                (output_one / "README.md").read_text(encoding="utf-8"),
            )

    def test_manifest_does_not_publish_absolute_input_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self._write_fixture(root)
            project_root = root / "public_project"
            project_root.mkdir()
            output = project_root / "results"

            package_predictions(archive, root / "data", output, project_root=project_root)

            manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            self.assertNotIn(str(root), manifest_text)
            self.assertNotIn("path", manifest["archive"])
            self.assertEqual(manifest["sources"]["SL329"]["path"], "SL329_test.fasta")
            self.assertEqual(
                manifest["generation"]["script"]["path"],
                "package_disorderunetlm_predictions.py",
            )

    def test_rejects_missing_or_extra_exact_protein_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self._write_fixture(root)
            malformed = root / "missing_extra.zip"
            extra = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
            self._rewrite_archive(
                archive,
                malformed,
                remove={"MXD494/mxd494_0000.npy"},
                add={"MXD494/not_in_source.npy": _npy_bytes(extra)},
            )
            with self.assertRaises(PackagingError):
                package_predictions(malformed, root / "data", root / "output", project_root=root)

    def test_rejects_malformed_npy_contract_without_pickle(self) -> None:
        mutations: Iterable[np.ndarray] = (
            np.asarray([[0.5, 0.5, 0.0]], dtype=np.float32),
            np.asarray([[0.2, 0.2]], dtype=np.float32),
            np.asarray([[np.nan, 1.0]], dtype=np.float32),
            np.asarray([["a", "b"]], dtype=object),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self._write_fixture(root)
            for index, mutation in enumerate(mutations):
                malformed = root / "malformed_{}.zip".format(index)
                self._rewrite_archive(
                    archive,
                    malformed,
                    replace={"SL329/sl_multipart_P1.npy": _npy_bytes(mutation)},
                )
                with self.subTest(index=index):
                    with self.assertRaises(PackagingError):
                        package_predictions(malformed, root / "data", root / "output_{}".format(index), project_root=root)


if __name__ == "__main__":
    unittest.main()
