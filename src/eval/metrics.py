"""Plain-Python classification metrics -- no sklearn dependency needed for
accuracy/precision/recall/F1/confusion-matrix on a handful of labels.
"""


def compute_classification_metrics(y_true, y_pred, labels):
    """Compute accuracy, per-label precision/recall/F1, and a confusion matrix.

    Returns:
        {
            "accuracy": float,
            "per_label": {label: {"precision", "recall", "f1", "support"}},
            "confusion_matrix": {true_label: {predicted_label: count}},
        }
    """
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n else 0.0

    confusion = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}
    for true_label, pred_label in zip(y_true, y_pred):
        confusion.setdefault(true_label, {pred_label: 0})
        confusion[true_label][pred_label] = confusion[true_label].get(pred_label, 0) + 1

    per_label = {}
    for label in labels:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(
            confusion.get(other, {}).get(label, 0) for other in labels if other != label
        )
        fn = sum(
            count
            for pred_label, count in confusion.get(label, {}).items()
            if pred_label != label
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    return {
        "accuracy": accuracy,
        "per_label": per_label,
        "confusion_matrix": confusion,
    }
