import csv
import json

from src.golden_set.sample_candidates import (
    CATEGORY_TARGETS,
    build_pools,
    classify,
    run as run_sample_candidates,
    sample_up_to,
)
from src.golden_set.validate_labels import (
    normalize_bool,
    print_summary,
    validate_row,
    validate_rows,
)


def make_pair(tweet_id, customer_text_clean, customer_text_raw=None):
    return {
        "customer_tweet_id": tweet_id,
        "customer_text_raw": customer_text_raw or customer_text_clean,
        "customer_text_clean": customer_text_clean,
        "brand_reply_tweet_id": f"r-{tweet_id}",
        "brand_reply_text_raw": "brand reply raw",
        "brand_reply_text_clean": "brand reply clean",
        "created_at": "2020-01-01",
        "is_thread_root": True,
        "is_dm_redirect": False,
    }


# --- classify: one hand-picked example per category ---


def test_classify_flight_disruption():
    assert classify("my flight was delayed for hours") == "FLIGHT_DISRUPTION"


def test_classify_baggage_problem():
    assert classify("my baggage never arrived at the carousel") == "BAGGAGE_PROBLEM"


def test_classify_refund_compensation():
    assert classify("i want a refund for this mess") == "REFUND_COMPENSATION"


def test_classify_booking_change():
    assert classify("i need to reschedule my flight") == "BOOKING_CHANGE"


def test_classify_account_loyalty():
    assert classify("how do i check my skymiles balance") == "ACCOUNT_LOYALTY"


def test_classify_policy_question():
    assert classify("what is the carry-on policy") == "POLICY_QUESTION"


def test_classify_praise_feedback():
    assert classify("thank you so much for your help today") == "PRAISE_FEEDBACK"


def test_classify_service_complaint_is_fallback():
    assert classify("this is ridiculous, nobody responded to me") == "SERVICE_COMPLAINT"


def test_classify_risk_signal_overrides_category_keyword():
    # Contains "bag" (BAGGAGE_PROBLEM keyword) AND "unsafe" (risk keyword) --
    # risk must win.
    text = "my bag was lost and i felt unsafe boarding the plane"
    assert classify(text) == "RISK_SIGNAL"


def test_classify_risk_signal_wheelchair():
    assert classify("my wheelchair was damaged during the flight") == "RISK_SIGNAL"


def test_classify_issue_word_does_not_trigger_sue_risk_keyword():
    # Regression test: "sue" must not match as a substring of "issue"/
    # "issues" (i-s-s-u-e contains s-u-e). A message about an ordinary
    # complaint containing "issue" should NOT land in the risk pool.
    text = "my flight was delayed and this has been a real issue for me, multiple issues today"
    assert classify(text) != "RISK_SIGNAL"
    assert classify(text) == "FLIGHT_DISRUPTION"


def test_classify_actual_sue_keyword_still_matches_risk_signal():
    # A real, standalone use of "sue" must still trigger the risk bucket.
    assert classify("i am going to sue the airline over this") == "RISK_SIGNAL"


# --- build_pools: risk match is never double-counted into a category pool ---


def test_build_pools_risk_match_excluded_from_category_pool():
    risky_pair = make_pair("1", "my bag was lost and i felt unsafe boarding")
    normal_pair = make_pair("2", "my baggage never arrived")

    risk_pool, category_pools = build_pools([risky_pair, normal_pair])

    assert [p["customer_tweet_id"] for p in risk_pool] == ["1"]
    assert [p["customer_tweet_id"] for p in category_pools["BAGGAGE_PROBLEM"]] == ["2"]


# --- sample_up_to: respects target, falls back gracefully when undersized ---


def test_sample_up_to_returns_exactly_target_when_pool_larger():
    pool = [make_pair(str(i), "my flight was delayed") for i in range(50)]
    sampled = sample_up_to(pool, target=10, seed=42)
    assert len(sampled) == 10
    sampled_ids = {p["customer_tweet_id"] for p in sampled}
    pool_ids = {p["customer_tweet_id"] for p in pool}
    assert sampled_ids.issubset(pool_ids)


def test_sample_up_to_uses_all_when_pool_smaller_than_target():
    pool = [make_pair(str(i), "my flight was delayed") for i in range(5)]
    sampled = sample_up_to(pool, target=34, seed=42)
    assert sampled == pool


def test_sample_up_to_is_reproducible():
    pool = [make_pair(str(i), "my flight was delayed") for i in range(50)]
    a = sample_up_to(pool, target=10, seed=42)
    b = sample_up_to(pool, target=10, seed=42)
    assert a == b


# --- run(): end-to-end undersized-bucket warning + CSV output ---


def test_run_warns_on_undersized_bucket_and_writes_csv(tmp_path, capsys):
    pairs = []
    # Only 3 FLIGHT_DISRUPTION candidates, far fewer than the target of 34.
    for i in range(3):
        pairs.append(make_pair(f"flight-{i}", "my flight was delayed again"))
    # A non-root pair that must be excluded entirely.
    non_root = make_pair("non-root", "my flight was delayed")
    non_root["is_thread_root"] = False
    pairs.append(non_root)

    input_path = tmp_path / "pairs.jsonl"
    with input_path.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    output_path = tmp_path / "candidates.csv"
    run_sample_candidates(str(input_path), str(output_path), seed=42)

    out = capsys.readouterr().out
    assert "WARNING: bucket FLIGHT_DISRUPTION has only 3 candidates" in out
    assert f"fewer than the target of {CATEGORY_TARGETS['FLIGHT_DISRUPTION']}" in out

    with output_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert all(row["sampling_stratum"] == "keyword_bucket:FLIGHT_DISRUPTION" for row in rows)
    assert {row["example_id"] for row in rows} == {"flight-0", "flight-1", "flight-2"}


# --- validate_labels ---


def _valid_row(**overrides):
    row = {
        "example_id": "1",
        "true_intent": "FLIGHT_DISRUPTION",
        "intent_ambiguous": "FALSE",
        "true_escalate": "FALSE",
        "true_escalation_rule": "none_auto_handle",
        "reply_acceptability_notes": "reply was accurate and on point",
    }
    row.update(overrides)
    return row


def test_validate_row_bad_category_name():
    row = _valid_row(true_intent="FLIGHT_ISSUE")
    problems = validate_row(row)
    assert any("true_intent is not a valid category" in p and "FLIGHT_ISSUE" in p for p in problems)


def test_validate_row_malformed_boolean():
    row = _valid_row(intent_ambiguous="maybe")
    problems = validate_row(row)
    assert any("intent_ambiguous is not a valid boolean" in p for p in problems)


def test_validate_row_escalate_true_with_none_auto_handle_is_inconsistent():
    row = _valid_row(true_escalate="TRUE", true_escalation_rule="none_auto_handle")
    problems = validate_row(row)
    assert any("true_escalate is True but true_escalation_rule is invalid" in p for p in problems)


def test_validate_row_escalate_false_with_real_rule_is_inconsistent():
    row = _valid_row(true_escalate="FALSE", true_escalation_rule="hard_risk_keyword")
    problems = validate_row(row)
    assert any("true_escalate is False but true_escalation_rule is" in p for p in problems)


def test_validate_row_empty_notes():
    row = _valid_row(reply_acceptability_notes="")
    problems = validate_row(row)
    assert any("reply_acceptability_notes is empty or too short" in p for p in problems)


def test_validate_rows_zero_problems_for_valid_fixture():
    rows = [
        _valid_row(example_id="1"),
        _valid_row(
            example_id="2",
            true_intent="BAGGAGE_PROBLEM",
            true_escalate="TRUE",
            true_escalation_rule="hard_risk_keyword",
            intent_ambiguous="yes",
        ),
    ]
    assert validate_rows(rows) == {}


def test_print_summary_reports_correct_counts(capsys):
    rows = [
        _valid_row(example_id="1"),
        _valid_row(
            example_id="2",
            true_intent="BAGGAGE_PROBLEM",
            true_escalate="TRUE",
            true_escalation_rule="hard_risk_keyword",
            intent_ambiguous="yes",
        ),
    ]
    print_summary(rows)
    out = capsys.readouterr().out
    assert "Total labelled rows: 2" in out
    assert "FLIGHT_DISRUPTION: 1" in out
    assert "BAGGAGE_PROBLEM: 1" in out
    assert "none_auto_handle: 1" in out
    assert "hard_risk_keyword: 1" in out
    assert "intent_ambiguous=True: 1" in out
    assert "true_escalate=True: 1, true_escalate=False: 1" in out


def test_normalize_bool_accepts_common_variants():
    for value in ["TRUE", "True", "true", "1", "yes", "YES"]:
        assert normalize_bool(value) is True
    for value in ["FALSE", "False", "false", "0", "no", "NO"]:
        assert normalize_bool(value) is False
    assert normalize_bool("maybe") is None
    assert normalize_bool("") is None
    assert normalize_bool(None) is None
