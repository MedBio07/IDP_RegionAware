from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ensemble_disorder_predictions import average_predictions, main, parse_args


class EnsembleDisorderPredictionsWeightsTest(unittest.TestCase):
    def write_prediction_file(self, directory: Path, name: str, rows: list[tuple[str, list[float]]]) -> Path:
        path = directory / name
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("id\tscores\n")
            for protein_id, scores in rows:
                handle.write(f"{protein_id}\t{scores}\n")
        return path

    def test_weighted_cli_uses_normalized_residue_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            first = self.write_prediction_file(directory, "first.tsv", [("p1", [0.0, 1.0]), ("p2", [0.2])])
            second = self.write_prediction_file(directory, "second.tsv", [("p1", [1.0, 0.0]), ("p2", [0.8])])
            output = directory / "weighted.tsv"

            with patch.object(
                sys,
                "argv",
                [
                    "ensemble_disorder_predictions.py",
                    "--inputs",
                    str(first),
                    str(second),
                    "--weights",
                    "2",
                    "1",
                    "--out",
                    str(output),
                ],
            ):
                main()

            self.assertEqual(
                average_predictions([first, second], "\t", [2.0, 1.0]),
                {"p1": [1.0 / 3.0, 2.0 / 3.0], "p2": [0.4]},
            )
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "id\tscores\n"
                "p1\t[0.33333333, 0.66666667]\n"
                "p2\t[0.4]\n",
            )

    def test_missing_weights_preserves_legacy_equal_average(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            first = self.write_prediction_file(directory, "first.tsv", [("p1", [0.1, 0.9])])
            second = self.write_prediction_file(directory, "second.tsv", [("p1", [0.3, 0.7])])
            self.assertEqual(
                average_predictions([first, second], "\t"),
                {"p1": [(0.1 + 0.3) / 2.0, (0.9 + 0.7) / 2.0]},
            )

            output = directory / "equal.tsv"
            with patch.object(
                sys,
                "argv",
                [
                    "ensemble_disorder_predictions.py",
                    "--inputs",
                    str(first),
                    str(second),
                    "--out",
                    str(output),
                ],
            ):
                main()
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "id\tscores\n"
                "p1\t[0.2, 0.8]\n",
            )

    def test_invalid_weights_are_rejected_by_cli_parser(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            first = directory / "first.tsv"
            second = directory / "second.tsv"
            cases = (
                (["1"], "exactly one value per"),
                (["-1", "2"], "non-negative"),
                (["0", "0"], "greater than zero"),
                (["nan", "1"], "finite"),
            )
            for weights, message in cases:
                with self.subTest(weights=weights):
                    with patch.object(
                        sys,
                        "argv",
                        [
                            "ensemble_disorder_predictions.py",
                            "--inputs",
                            str(first),
                            str(second),
                            "--weights",
                            *weights,
                            "--out",
                            str(directory / "out.tsv"),
                        ],
                    ):
                        with self.assertRaisesRegex(ValueError, message):
                            parse_args()

    def test_id_and_length_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            reference = self.write_prediction_file(directory, "reference.tsv", [("p1", [0.1, 0.2])])
            different_id = self.write_prediction_file(directory, "different_id.tsv", [("p2", [0.3, 0.4])])
            different_length = self.write_prediction_file(directory, "different_length.tsv", [("p1", [0.3])])

            with self.assertRaisesRegex(ValueError, "prediction IDs differ"):
                average_predictions([reference, different_id], "\t", [1.0, 1.0])
            with self.assertRaisesRegex(ValueError, "length mismatch for p1"):
                average_predictions([reference, different_length], "\t", [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
