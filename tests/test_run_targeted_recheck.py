from src.eval.run_targeted_recheck import split_predictions


def _prediction(example_id, correct):
    return {
        "example_id": example_id,
        "true_intent": "FLIGHT_DISRUPTION",
        "predicted_intent": "FLIGHT_DISRUPTION" if correct else "BAGGAGE_PROBLEM",
        "correct": correct,
        "second_choice": None,
        "reasoning": "r",
        "intent_ambiguous": False,
    }


def test_split_predictions_separates_by_correct_field():
    predictions = [
        _prediction("1", correct=False),
        _prediction("2", correct=True),
        _prediction("3", correct=False),
        _prediction("4", correct=True),
        _prediction("5", correct=True),
    ]

    previously_wrong, control_sample = split_predictions(predictions, control_size=10, seed=42)

    assert {p["example_id"] for p in previously_wrong} == {"1", "3"}
    # control_size (10) exceeds the number of previously-right examples (3),
    # so all of them should be used.
    assert {p["example_id"] for p in control_sample} == {"2", "4", "5"}


def test_split_predictions_samples_control_group_deterministically_by_seed():
    predictions = [_prediction(str(i), correct=True) for i in range(50)]

    sample_a = split_predictions(predictions, control_size=10, seed=42)[1]
    sample_b = split_predictions(predictions, control_size=10, seed=42)[1]

    assert len(sample_a) == 10
    assert sample_a == sample_b


def test_split_predictions_control_sample_is_subset_of_previously_right():
    predictions = [_prediction(str(i), correct=True) for i in range(20)] + [
        _prediction(f"wrong-{i}", correct=False) for i in range(5)
    ]

    previously_wrong, control_sample = split_predictions(predictions, control_size=10, seed=42)

    assert len(previously_wrong) == 5
    assert len(control_sample) == 10
    control_ids = {p["example_id"] for p in control_sample}
    right_ids = {p["example_id"] for p in predictions if p["correct"] is True}
    assert control_ids.issubset(right_ids)
