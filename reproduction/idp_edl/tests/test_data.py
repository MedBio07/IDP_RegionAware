import unittest
from pathlib import Path

from reproduction.idp_edl.data import (
    default_sample_paths,
    parse_author_samples,
    truncation_stats,
    truncate_labels,
)


class AuthorSampleParsingTest(unittest.TestCase):
    def test_real_author_samples_are_aligned_and_filtered(self):
        sequence_path, label_path = default_sample_paths(Path(__file__).resolve().parents[3])
        examples = parse_author_samples(sequence_path, label_path, ["mxd494", "sl329", "disorder723"])
        counts = {}
        residues = {}
        label_sets = {}
        for example in examples:
            counts[example.dataset] = counts.get(example.dataset, 0) + 1
            residues[example.dataset] = residues.get(example.dataset, 0) + len(example.sequence)
            label_sets.setdefault(example.dataset, set()).update(example.labels)
            self.assertEqual(len(example.sequence), len(example.labels))
            self.assertTrue(example.identifier)
        self.assertEqual(counts, {"mxd494": 494, "sl329": 322, "disorder723": 723})
        self.assertEqual(residues, {"mxd494": 196501, "sl329": 157376, "disorder723": 215229})
        self.assertEqual(label_sets["mxd494"], set("01"))
        self.assertEqual(label_sets["sl329"], set("012"))
        self.assertEqual(label_sets["disorder723"], set("01"))

    def test_truncation_reserves_eos_and_reports_residues(self):
        stats = truncation_stats(1500, max_length=1024)
        self.assertEqual(stats.retained_length, 1023)
        self.assertEqual(stats.truncated_residues, 477)
        self.assertEqual(stats.eos_tokens, 1)
        labels, same_stats = truncate_labels("0" * 1500, 1500, max_length=1024)
        self.assertEqual(len(labels), 1023)
        self.assertEqual(same_stats, stats)
