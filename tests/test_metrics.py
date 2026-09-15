from src.eval.metrics import compute_classification_metrics


def test_compute_classification_metrics_known_values():
    # Hand-worked confusion matrix (rows=true, cols=predicted):
    #        pred A  pred B  pred C
    # true A    1       1       0
    # true B    0       1       1
    # true C    0       0       1
    y_true = ["A", "A", "B", "B", "C"]
    y_pred = ["A", "B", "B", "C", "C"]
    labels = ["A", "B", "C"]

    result = compute_classification_metrics(y_true, y_pred, labels)

    assert result["accuracy"] == 3 / 5

    assert result["confusion_matrix"] == {
        "A": {"A": 1, "B": 1, "C": 0},
        "B": {"A": 0, "B": 1, "C": 1},
        "C": {"A": 0, "B": 0, "C": 1},
    }

    # A: tp=1, fp=0 (no other true label predicted A), fn=1 (true A predicted B)
    a = result["per_label"]["A"]
    assert a["precision"] == 1.0
    assert a["recall"] == 0.5
    assert round(a["f1"], 4) == round(2 * 1.0 * 0.5 / 1.5, 4)
    assert a["support"] == 2

    # B: tp=1, fp=1 (true A predicted B), fn=1 (true B predicted C)
    b = result["per_label"]["B"]
    assert b["precision"] == 0.5
    assert b["recall"] == 0.5
    assert b["f1"] == 0.5
    assert b["support"] == 2

    # C: tp=1, fp=1 (true B predicted C), fn=0
    c = result["per_label"]["C"]
    assert c["precision"] == 0.5
    assert c["recall"] == 1.0
    assert round(c["f1"], 4) == round(2 * 0.5 * 1.0 / 1.5, 4)
    assert c["support"] == 1


def test_compute_classification_metrics_perfect_predictions():
    y_true = ["A", "B", "C"]
    y_pred = ["A", "B", "C"]
    labels = ["A", "B", "C"]

    result = compute_classification_metrics(y_true, y_pred, labels)

    assert result["accuracy"] == 1.0
    for label in labels:
        stats = result["per_label"][label]
        assert stats["precision"] == 1.0
        assert stats["recall"] == 1.0
        assert stats["f1"] == 1.0


def test_compute_classification_metrics_label_never_predicted_has_zero_precision_and_f1():
    # Label "B" never predicted at all -> its precision/f1 should be 0.0
    # (defined as 0 rather than raising a division error), while recall is
    # also 0 since none of its true examples were ever correctly predicted.
    y_true = ["A", "B", "A"]
    y_pred = ["A", "A", "A"]
    labels = ["A", "B"]

    result = compute_classification_metrics(y_true, y_pred, labels)

    b = result["per_label"]["B"]
    assert b["precision"] == 0.0
    assert b["recall"] == 0.0
    assert b["f1"] == 0.0
    assert b["support"] == 1

    a = result["per_label"]["A"]
    assert a["precision"] == 2 / 3
    assert a["recall"] == 1.0


def test_compute_classification_metrics_empty_input():
    result = compute_classification_metrics([], [], ["A", "B"])
    assert result["accuracy"] == 0.0
    assert result["per_label"]["A"]["precision"] == 0.0
    assert result["confusion_matrix"] == {"A": {"A": 0, "B": 0}, "B": {"A": 0, "B": 0}}
