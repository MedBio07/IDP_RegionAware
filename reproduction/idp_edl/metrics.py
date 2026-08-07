"""Metrics used by the IDP-EDL paper tables."""

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def _fmax(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if labels.size == 0 or np.unique(labels).size < 2:
        return None

    # Evaluate the same unique score thresholds as the direct implementation,
    # deriving every confusion count from one stable sort.
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    threshold_starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    suffix_tp = np.cumsum(sorted_labels[::-1], dtype=np.int64)[::-1]
    tp = suffix_tp[threshold_starts].astype(float)
    predicted_positive = (labels.size - threshold_starts).astype(float)
    total_positive = float(np.sum(labels == 1))
    precision = np.divide(tp, predicted_positive, out=np.zeros_like(tp), where=predicted_positive > 0)
    recall = tp / total_positive
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(np.max(f1))


def compute_binary_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Optional[float]]:
    """Compute binary residue metrics while excluding label ``2``.

    The paper's class prediction is ``argmax(softmax(logits))``.  For a
    binary positive-class score, ``score > 0.5`` reproduces argmax's class-0
    tie behavior.
    """

    labels_array = np.asarray(labels, dtype=int).reshape(-1)
    scores_array = np.asarray(scores, dtype=float).reshape(-1)
    if labels_array.size != scores_array.size:
        raise ValueError("labels and scores must have the same length")
    valid = np.isin(labels_array, [0, 1]) & np.isfinite(scores_array)
    y_true = labels_array[valid]
    y_score = scores_array[valid]
    result: Dict[str, Optional[float]] = {
        "n_total": int(labels_array.size),
        "n_evaluated": int(y_true.size),
        "n_excluded_label_2": int(np.sum(labels_array == 2)),
        "threshold": float(threshold),
    }
    if y_true.size == 0:
        result.update(
            {
                "accuracy": None,
                "auc": None,
                "sensitivity": None,
                "specificity": None,
                "bacc": None,
                "mcc": None,
                "fmax": None,
            }
        )
        return result

    predicted = y_score > threshold
    tp = int(np.sum(predicted & (y_true == 1)))
    tn = int(np.sum((~predicted) & (y_true == 0)))
    fp = int(np.sum(predicted & (y_true == 0)))
    fn = int(np.sum((~predicted) & (y_true == 1)))
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    result.update(
        {
            "accuracy": float(np.mean(predicted == y_true)),
            "auc": float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size == 2 else None,
            "sensitivity": _safe_ratio(tp, tp + fn),
            "specificity": _safe_ratio(tn, tn + fp),
            "bacc": (
                float((_safe_ratio(tp, tp + fn) + _safe_ratio(tn, tn + fp)) / 2.0)
                if _safe_ratio(tp, tp + fn) is not None and _safe_ratio(tn, tn + fp) is not None
                else None
            ),
            "mcc": float((tp * tn - fp * fn) / np.sqrt(denominator)) if denominator else None,
            "fmax": _fmax(y_true, y_score),
        }
    )
    result.update({"tp": tp, "tn": tn, "fp": fp, "fn": fn})
    return result
