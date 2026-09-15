import pytest

from src.eval.compute_judge_agreement import _parse_human_score, compute_agreement, is_row_complete


def _judge_result(example_id, grounded, factually_consistent, tone_appropriate, actionable):
    return {
        "example_id": example_id,
        "scores": {
            "grounded": grounded,
            "factually_consistent": factually_consistent,
            "tone_appropriate": tone_appropriate,
            "actionable": actionable,
            "notes": "note",
        },
    }


def _human_row(
    example_id,
    human_grounded="",
    human_factually_consistent="",
    human_tone_appropriate="",
    human_actionable="",
):
    return {
        "example_id": example_id,
        "human_grounded": human_grounded,
        "human_factually_consistent": human_factually_consistent,
        "human_tone_appropriate": human_tone_appropriate,
        "human_actionable": human_actionable,
        "human_notes": "",
    }


# --- is_row_complete ---


def test_is_row_complete_true_when_all_four_filled():
    row = _human_row("1", "3", "2", "3", "1")
    assert is_row_complete(row) is True


def test_is_row_complete_false_when_any_field_blank():
    row = _human_row("1", "3", "", "3", "1")
    assert is_row_complete(row) is False


def test_is_row_complete_false_when_field_non_numeric():
    row = _human_row("1", "3", "n/a", "3", "1")
    assert is_row_complete(row) is False


def test_is_row_complete_false_when_field_out_of_range():
    row = _human_row("1", "3", "5", "3", "1")
    assert is_row_complete(row) is False


# --- _parse_human_score: spreadsheet-reformatted decimals like "3.0" ---


def test_parse_human_score_accepts_clean_integer_string():
    assert _parse_human_score("3") == 3


def test_parse_human_score_accepts_whole_number_float_string():
    # Excel/Numbers/Sheets commonly reformat an integer CSV column as a
    # decimal on save -- "3.0" is a realistic, recurring value, not an
    # edge case, and must still parse as the integer 3.
    assert _parse_human_score("3.0") == 3
    assert _parse_human_score("1.0") == 1
    assert _parse_human_score("2.0") == 2


def test_parse_human_score_rejects_non_whole_float_string():
    # "2.5" must be rejected as genuinely invalid -- never silently rounded.
    assert _parse_human_score("2.5") is None


def test_parse_human_score_rejects_out_of_range_whole_float():
    assert _parse_human_score("4.0") is None
    assert _parse_human_score("0.0") is None


def test_is_row_complete_true_with_whole_number_float_strings():
    row = _human_row("1", "3.0", "2.0", "3.0", "1.0")
    assert is_row_complete(row) is True


# --- compute_agreement ---


def test_compute_agreement_exact_match_and_within_one():
    judge_results_by_id = {
        "1": _judge_result("1", grounded=3, factually_consistent=3, tone_appropriate=2, actionable=1),
        "2": _judge_result("2", grounded=2, factually_consistent=1, tone_appropriate=3, actionable=3),
    }
    human_rows = [
        # example 1: grounded exact match (3==3); factually_consistent off
        # by 2 (3 vs 1, neither exact nor within-1); tone exact (2==2);
        # actionable within-1 but not exact (1 vs 2).
        _human_row("1", human_grounded="3", human_factually_consistent="1", human_tone_appropriate="2", human_actionable="2"),
        # example 2: all exact matches.
        _human_row("2", human_grounded="2", human_factually_consistent="1", human_tone_appropriate="3", human_actionable="3"),
    ]

    per_dimension, n_scored, n_skipped = compute_agreement(judge_results_by_id, human_rows)

    assert n_scored == 2
    assert n_skipped == 0

    # grounded: judge=[3,2], human=[3,2] -> both exact
    assert per_dimension["grounded"]["exact_match_rate"] == 1.0
    assert per_dimension["grounded"]["within_one_rate"] == 1.0

    # factually_consistent: judge=[3,1], human=[1,1] -> row1 mismatch by 2
    # (not exact, not within-1), row2 exact -> exact=0.5, within_one=0.5
    fc = per_dimension["factually_consistent"]
    assert fc["exact_match_rate"] == 0.5
    assert fc["within_one_rate"] == 0.5

    # tone_appropriate: judge=[2,3], human=[2,3] -> both exact
    assert per_dimension["tone_appropriate"]["exact_match_rate"] == 1.0

    # actionable: judge=[1,3], human=[2,3] -> row1 within-1 (|1-2|=1) but
    # not exact, row2 exact -> exact=0.5, within_one=1.0
    actionable = per_dimension["actionable"]
    assert actionable["exact_match_rate"] == 0.5
    assert actionable["within_one_rate"] == 1.0


def test_compute_agreement_skips_incomplete_rows_and_counts_them():
    judge_results_by_id = {
        "1": _judge_result("1", grounded=3, factually_consistent=3, tone_appropriate=2, actionable=1),
        "2": _judge_result("2", grounded=2, factually_consistent=1, tone_appropriate=3, actionable=3),
        "3": _judge_result("3", grounded=1, factually_consistent=1, tone_appropriate=1, actionable=1),
    }
    human_rows = [
        _human_row("1", human_grounded="3", human_factually_consistent="3", human_tone_appropriate="2", human_actionable="1"),
        # incomplete: human_actionable left blank -- must be skipped, not crash.
        _human_row("2", human_grounded="2", human_factually_consistent="1", human_tone_appropriate="3", human_actionable=""),
        # incomplete: non-numeric value.
        _human_row("3", human_grounded="1", human_factually_consistent="1", human_tone_appropriate="oops", human_actionable="1"),
    ]

    per_dimension, n_scored, n_skipped = compute_agreement(judge_results_by_id, human_rows)

    assert n_scored == 1
    assert n_skipped == 2
    # Only row "1" contributed -- perfect agreement on every dimension.
    for field, stats in per_dimension.items():
        assert stats["n"] == 1
        assert stats["exact_match_rate"] == 1.0
        assert stats["within_one_rate"] == 1.0


def test_compute_agreement_raises_on_missing_judge_result():
    judge_results_by_id = {
        "1": _judge_result("1", grounded=3, factually_consistent=3, tone_appropriate=2, actionable=1),
    }
    human_rows = [
        _human_row("2", human_grounded="2", human_factually_consistent="1", human_tone_appropriate="3", human_actionable="3"),
    ]

    with pytest.raises(RuntimeError, match="no judge result found"):
        compute_agreement(judge_results_by_id, human_rows)


def test_compute_agreement_all_rows_incomplete_reports_zero_scored():
    judge_results_by_id = {
        "1": _judge_result("1", grounded=3, factually_consistent=3, tone_appropriate=2, actionable=1),
    }
    human_rows = [_human_row("1")]  # nothing filled in at all

    per_dimension, n_scored, n_skipped = compute_agreement(judge_results_by_id, human_rows)

    assert n_scored == 0
    assert n_skipped == 1
    for stats in per_dimension.values():
        assert stats["n"] == 0
        assert stats["exact_match_rate"] == 0.0
        assert stats["within_one_rate"] == 0.0
