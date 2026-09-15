from src.agent.route import route_message


def _classify_result(intent, second_choice=None):
    return {"intent": intent, "second_choice": second_choice, "reasoning": "r"}


# --- rule 1: hard_risk_keyword ---


def test_rule1_hard_risk_keyword_fires_and_names_the_keyword():
    result = route_message(
        "I'm going to sue Delta over this lawsuit-worthy mess",
        _classify_result("SERVICE_COMPLAINT"),
        top_similarity=0.90,
    )
    assert result["escalate"] is True
    assert result["rule"] == "hard_risk_keyword"
    assert "lawsuit" in result["reason"] or "sue" in result["reason"]


def test_rule1_wins_over_rule2_when_both_match():
    # Matches rule 1 (a hard-risk keyword) AND rule 2 (BAGGAGE_PROBLEM
    # is a risk-category-default intent) -- rule 1 must win since it's
    # checked first.
    result = route_message(
        "they lost my wheelchair and I need it back",
        _classify_result("BAGGAGE_PROBLEM"),
        top_similarity=0.90,
    )
    assert result["escalate"] is True
    assert result["rule"] == "hard_risk_keyword"


def test_rule1_word_boundary_issue_does_not_match_sue():
    # Regression test, same spirit as the golden-set keyword-matching fix:
    # "sue" must not match as a substring of "issue"/"issues". This
    # message has no real risk keyword, a non-risk intent, strong
    # similarity, and no ambiguity -- should fall through to rule 5.
    result = route_message(
        "I have an issue with my seat, there were multiple issues today",
        _classify_result("BOOKING_CHANGE"),
        top_similarity=0.90,
    )
    assert result["rule"] != "hard_risk_keyword"
    assert result["rule"] == "none_auto_handle"
    assert result["escalate"] is False


# --- rule 2: risk_category_default ---


def test_rule2_baggage_problem_defaults_to_escalate():
    result = route_message(
        "my bag never arrived",
        _classify_result("BAGGAGE_PROBLEM"),
        top_similarity=0.90,
    )
    assert result["escalate"] is True
    assert result["rule"] == "risk_category_default"
    assert "BAGGAGE_PROBLEM" in result["reason"]


def test_rule2_refund_compensation_defaults_to_escalate():
    result = route_message(
        "please refund my ticket",
        _classify_result("REFUND_COMPENSATION"),
        top_similarity=0.90,
    )
    assert result["escalate"] is True
    assert result["rule"] == "risk_category_default"
    assert "REFUND_COMPENSATION" in result["reason"]


# --- rule 3: weak_grounding ---


def test_rule3_weak_grounding_fires_below_threshold():
    result = route_message(
        "what is the carry-on policy",
        _classify_result("POLICY_QUESTION"),
        top_similarity=0.50,
    )
    assert result["escalate"] is True
    assert result["rule"] == "weak_grounding"
    assert "0.50" in result["reason"]


def test_rule3_threshold_is_configurable():
    # Similarity of 0.85 is below a custom, stricter threshold of 0.90.
    result = route_message(
        "what is the carry-on policy",
        _classify_result("POLICY_QUESTION"),
        top_similarity=0.85,
        weak_grounding_threshold=0.90,
    )
    assert result["rule"] == "weak_grounding"


# --- rule 4: ambiguous_classification ---


def test_rule4_ambiguous_classification_fires_with_second_choice():
    result = route_message(
        "thanks but the delay was still annoying",
        _classify_result("PRAISE_FEEDBACK", second_choice="SERVICE_COMPLAINT"),
        top_similarity=0.90,
    )
    assert result["escalate"] is True
    assert result["rule"] == "ambiguous_classification"
    assert "PRAISE_FEEDBACK" in result["reason"]
    assert "SERVICE_COMPLAINT" in result["reason"]


# --- rule 5: none_auto_handle ---


def test_rule5_auto_handles_routine_confident_case():
    result = route_message(
        "thank you so much for your help",
        _classify_result("PRAISE_FEEDBACK", second_choice=None),
        top_similarity=0.90,
    )
    assert result["escalate"] is False
    assert result["rule"] == "none_auto_handle"
    assert "0.90" in result["reason"]


# --- escalate is True for rules 1-4, False ONLY for rule 5 ---


def test_escalate_is_true_for_every_rule_except_none_auto_handle():
    cases = [
        ("hard_risk_keyword", route_message("I will sue you", _classify_result("SERVICE_COMPLAINT"), 0.90)),
        ("risk_category_default", route_message("my bag is lost", _classify_result("BAGGAGE_PROBLEM"), 0.90)),
        ("weak_grounding", route_message("hello", _classify_result("POLICY_QUESTION"), 0.10)),
        (
            "ambiguous_classification",
            route_message(
                "hello", _classify_result("POLICY_QUESTION", second_choice="SERVICE_COMPLAINT"), 0.90
            ),
        ),
    ]
    for expected_rule, result in cases:
        assert result["rule"] == expected_rule
        assert result["escalate"] is True

    auto_handled = route_message(
        "thanks!", _classify_result("PRAISE_FEEDBACK", second_choice=None), 0.90
    )
    assert auto_handled["rule"] == "none_auto_handle"
    assert auto_handled["escalate"] is False


def test_custom_hard_risk_keywords_override_default_list():
    result = route_message(
        "this involves a very specific custom trigger phrase",
        _classify_result("SERVICE_COMPLAINT"),
        top_similarity=0.90,
        hard_risk_keywords=["custom trigger phrase"],
    )
    assert result["rule"] == "hard_risk_keyword"
    assert "custom trigger phrase" in result["reason"]
