from src.eval.run_baseline_eval import run_trivial_baseline, run_simple_baseline


# --- trivial baseline ---


def test_trivial_baseline_predicts_the_mode_for_every_row():
    # Mode is "FLIGHT_DISRUPTION" (3 occurrences), others appear once each.
    y_true = [
        "FLIGHT_DISRUPTION",
        "FLIGHT_DISRUPTION",
        "FLIGHT_DISRUPTION",
        "BAGGAGE_PROBLEM",
        "PRAISE_FEEDBACK",
    ]

    mode_intent, metrics = run_trivial_baseline(y_true)

    assert mode_intent == "FLIGHT_DISRUPTION"
    # 3 correct out of 5 -> accuracy 0.6
    assert metrics["accuracy"] == 3 / 5


def test_trivial_baseline_picks_mode_from_data_not_hardcoded():
    # A different mode this time ("PRAISE_FEEDBACK") -- confirms the
    # function computes the mode from the data, not a hardcoded category.
    y_true = ["PRAISE_FEEDBACK", "PRAISE_FEEDBACK", "BAGGAGE_PROBLEM"]

    mode_intent, metrics = run_trivial_baseline(y_true)

    assert mode_intent == "PRAISE_FEEDBACK"
    assert metrics["accuracy"] == 2 / 3


# --- simple (keyword) baseline: real classify(), no mocking ---


def test_simple_baseline_calls_real_keyword_classifier():
    # Real text matching known keyword patterns from the golden-set
    # keyword classifier (see tests/test_golden_set.py) -- no mocking,
    # since this baseline makes zero API calls.
    texts = [
        "my flight was delayed for hours",  # -> FLIGHT_DISRUPTION
        "my baggage never arrived at the carousel",  # -> BAGGAGE_PROBLEM
        "thank you so much for your help today",  # -> PRAISE_FEEDBACK
    ]
    y_true = ["FLIGHT_DISRUPTION", "BAGGAGE_PROBLEM", "PRAISE_FEEDBACK"]

    metrics = run_simple_baseline(y_true, texts)

    # All three keyword matches are correct -> perfect accuracy on this fixture.
    assert metrics["accuracy"] == 1.0


def test_simple_baseline_reports_wrong_predictions_as_incorrect():
    # A message with no matching category keyword falls back to
    # SERVICE_COMPLAINT (per classify()'s documented fallback behavior),
    # which is wrong against a true_intent of PRAISE_FEEDBACK here.
    texts = ["this is ridiculous, nobody responded to me"]
    y_true = ["PRAISE_FEEDBACK"]

    metrics = run_simple_baseline(y_true, texts)

    assert metrics["accuracy"] == 0.0
