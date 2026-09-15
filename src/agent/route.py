"""Escalation routing -- pure rule logic, no LLM call, no API cost, fully
offline. Implements the 5-rule checklist from the decision log: checked
in order, first match wins, and the rule name IS the stated reason (not
reverse-engineered after the fact from an opaque score).
"""
import re

# Same list used for golden-set risk-signal sampling, for consistency
# between how examples were sampled/labelled and how the live agent routes.
HARD_RISK_KEYWORDS = [
    "discriminat",
    "racist",
    "racism",
    "assault",
    "harass",
    "unsafe",
    "danger",
    "safety concern",
    "legal action",
    "lawsuit",
    "attorney",
    "sue",
    "wheelchair",
    "medical emergency",
    "allergic reaction",
    "unaccompanied minor",
    "my son",
    "my daughter",
    "my child",
]

RISK_CATEGORY_DEFAULTS = {"BAGGAGE_PROBLEM", "REFUND_COMPENSATION"}

# Picked from eyeballing Phase 2/4 real similarity score distributions.
# KNOWN LIMITATION, documented rather than silently patched: Phase 4's
# failure analysis found this embedding model's similarity scores cluster
# narrowly (roughly 0.83-0.90) regardless of true relevance, so this rule
# may rarely fire meaningfully against real data with the current model.
# That's a limitation of the embedding model, not a bug in this rule.
DEFAULT_WEAK_GROUNDING_THRESHOLD = 0.80


def _compile_keyword_pattern(keyword):
    """Word-boundary match for a whole keyword/phrase, not a substring.

    Same approach as src/golden_set/sample_candidates.py's fixed keyword
    matching -- \\b sits at the start/end of the ENTIRE escaped phrase, so
    multi-word phrases like "safety concern" still match as a contiguous
    phrase, and this is what stops "sue" from matching inside "issue"
    (no word boundary between the 's' and 'u' of "sue" within "issue").
    """
    return re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)


def _find_matching_keyword(text, keywords):
    """Return the first keyword (in list order) that matches text as a
    whole word/phrase, or None if none match."""
    for keyword in keywords:
        if _compile_keyword_pattern(keyword).search(text or ""):
            return keyword
    return None


def route_message(
    raw_text,
    classify_result,
    top_similarity,
    hard_risk_keywords=None,
    weak_grounding_threshold=DEFAULT_WEAK_GROUNDING_THRESHOLD,
):
    """Decide whether a message should be escalated to a human, and why.

    Checked in this exact order -- the first rule that matches decides
    the outcome:
      1. hard_risk_keyword       -- raw_text contains genuine risk language
      2. risk_category_default   -- intent is BAGGAGE_PROBLEM/REFUND_COMPENSATION
      3. weak_grounding          -- top retrieved similarity is too low to trust
      4. ambiguous_classification -- classifier had a real second choice
      5. none_auto_handle        -- none of the above; safe to auto-handle

    Returns {"escalate": bool, "rule": <rule name>, "reason": <string>}.
    escalate is True for rules 1-4, False only for rule 5.
    """
    keywords = hard_risk_keywords if hard_risk_keywords is not None else HARD_RISK_KEYWORDS

    matched_keyword = _find_matching_keyword(raw_text, keywords)
    if matched_keyword is not None:
        return {
            "escalate": True,
            "rule": "hard_risk_keyword",
            "reason": f"High-risk keyword match: '{matched_keyword}'.",
        }

    intent = classify_result.get("intent")
    if intent in RISK_CATEGORY_DEFAULTS:
        return {
            "escalate": True,
            "rule": "risk_category_default",
            "reason": f"Category defaults to human review: {intent}.",
        }

    if top_similarity < weak_grounding_threshold:
        return {
            "escalate": True,
            "rule": "weak_grounding",
            "reason": (
                f"No sufficiently similar historical precedent found "
                f"(similarity={top_similarity:.2f})."
            ),
        }

    second_choice = classify_result.get("second_choice")
    if second_choice is not None:
        return {
            "escalate": True,
            "rule": "ambiguous_classification",
            "reason": f"Intent classification ambiguous: {intent} vs {second_choice}.",
        }

    return {
        "escalate": False,
        "rule": "none_auto_handle",
        "reason": (
            f"Routine case with strong historical precedent "
            f"(similarity={top_similarity:.2f}) and confident classification."
        ),
    }
