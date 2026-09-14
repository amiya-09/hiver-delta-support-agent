"""Shared constants for golden-set sampling and validation.

Locked-in design: 8 intent categories, 5 escalation rules (4 real reasons
to escalate + the "no escalation" terminal value).
"""

INTENT_CATEGORIES = [
    "FLIGHT_DISRUPTION",
    "BAGGAGE_PROBLEM",
    "BOOKING_CHANGE",
    "REFUND_COMPENSATION",
    "ACCOUNT_LOYALTY",
    "POLICY_QUESTION",
    "SERVICE_COMPLAINT",
    "PRAISE_FEEDBACK",
]

ESCALATION_RULES = [
    "hard_risk_keyword",
    "risk_category_default",
    "weak_grounding",
    "ambiguous_classification",
]

NONE_AUTO_HANDLE = "none_auto_handle"
