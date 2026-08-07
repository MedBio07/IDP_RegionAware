import unittest

import numpy as np

from reproduction.idp_edl.metrics import compute_binary_metrics


class MetricsTest(unittest.TestCase):
    def test_perfect_binary_metrics(self):
        result = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(result["n_evaluated"], 4)
        self.assertAlmostEqual(result["auc"], 1.0)
        self.assertAlmostEqual(result["bacc"], 1.0)
        self.assertAlmostEqual(result["mcc"], 1.0)
        self.assertAlmostEqual(result["fmax"], 1.0)

    def test_label_two_is_excluded_from_metrics(self):
        result = compute_binary_metrics([0, 2, 1], [0.1, 0.99, 0.9])
        self.assertEqual(result["n_total"], 3)
        self.assertEqual(result["n_evaluated"], 2)
        self.assertEqual(result["n_excluded_label_2"], 1)
        self.assertAlmostEqual(result["auc"], 1.0)

    def test_fmax_matches_direct_unique_threshold_scan(self):
        labels = np.asarray([0, 1, 0, 1, 1, 0, 2])
        scores = np.asarray([0.2, 0.2, 0.7, 0.9, 0.7, 0.1, 0.99])
        valid = labels != 2
        y_true = labels[valid]
        y_score = scores[valid]
        expected = 0.0
        for threshold in np.unique(y_score):
            predicted = y_score >= threshold
            tp = float(np.sum(predicted & (y_true == 1)))
            fp = float(np.sum(predicted & (y_true == 0)))
            fn = float(np.sum((~predicted) & (y_true == 1)))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            if precision + recall:
                expected = max(expected, 2.0 * precision * recall / (precision + recall))
        self.assertAlmostEqual(compute_binary_metrics(labels, scores)["fmax"], expected)
