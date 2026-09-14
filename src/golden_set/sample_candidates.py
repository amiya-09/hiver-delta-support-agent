"""Sample candidate examples for hand-labelling the golden set.

The keyword bucketing here is a rough heuristic used only to get
stratified coverage across intents and risk language for a human labeller
to review -- it is NOT the production classifier and is never used at
inference time.
"""
import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

RISK_KEYWORDS = [
    "discriminat", "racist", "racism", "racial", "assault", "harass", "unsafe",
    "danger", "safety concern", "legal action", "lawsuit", "attorney",
    "my lawyer", "sue", "wheelchair", "medical emergency", "allergic reaction",
    "unaccompanied minor", "my son", "my daughter", "my child",
]

# Order matters: first matching category wins. SERVICE_COMPLAINT has no
# keyword list -- it's the fallback for anything matching none of these.
CATEGORY_KEYWORDS = [
    ("FLIGHT_DISRUPTION", [
        "delay", "delayed", "cancel", "cancelled", "canceled", "ground stop",
        "missed connection", "diverted", "still on the ground",
    ]),
    ("BAGGAGE_PROBLEM", ["bag", "baggage", "luggage", "suitcase"]),
    ("REFUND_COMPENSATION", [
        "refund", "reimburse", "compensation", "credit", "voucher",
        "money back", "waive",
    ]),
    ("BOOKING_CHANGE", [
        "seat", "rebook", "reschedule", "change my flight", "change my seat",
        "reservation", "misspelled name", "name change",
    ]),
    ("ACCOUNT_LOYALTY", [
        "skymiles", "medallion", "miles", "gold status", "diamond", "loyalty",
    ]),
    ("POLICY_QUESTION", [
        "can i", "how do i", "policy", "allowed", "carry-on", "personal item",
    ]),
    ("PRAISE_FEEDBACK", [
        "thank", "thanks", "great", "awesome", "wonderful", "appreciate",
        "shoutout", "shout out",
    ]),
]

def _compile_keyword_pattern(keyword):
    """Word-boundary match for a whole keyword/phrase, not a substring.

    \b sits at the start and end of the ENTIRE escaped phrase, so
    multi-word keywords like "safety concern" or "my son" match as a
    contiguous phrase bounded by word breaks, not as separately-bounded
    words. This is what stops "sue" from matching inside "issue" (i-s-s-
    u-e has no word boundary between the 's' and 'u' of "sue").
    """
    return re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)


RISK_KEYWORD_PATTERNS = [_compile_keyword_pattern(k) for k in RISK_KEYWORDS]

CATEGORY_KEYWORD_PATTERNS = [
    (category, [_compile_keyword_pattern(k) for k in keywords])
    for category, keywords in CATEGORY_KEYWORDS
]

FALLBACK_CATEGORY = "SERVICE_COMPLAINT"

CATEGORY_TARGETS = {
    "FLIGHT_DISRUPTION": 34,
    "BAGGAGE_PROBLEM": 33,
    "REFUND_COMPENSATION": 33,
    "BOOKING_CHANGE": 20,
    "ACCOUNT_LOYALTY": 20,
    "POLICY_QUESTION": 20,
    "SERVICE_COMPLAINT": 20,
    "PRAISE_FEEDBACK": 20,
}

RISK_POOL_TARGET = 40

CSV_FIELDS = [
    "example_id",
    "sampling_stratum",
    "customer_text_raw",
    "customer_text_clean",
    "brand_reply_text_raw",
    "brand_reply_text_clean",
    "true_intent",
    "intent_ambiguous",
    "true_escalate",
    "true_escalation_rule",
    "reply_acceptability_notes",
    "labeller_notes",
]


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def classify(text_clean):
    """Bucket a cleaned customer message.

    Risk keywords are checked before any of the 8 category keyword lists
    so risk language is never miscategorized into a normal bucket (e.g. a
    message mentioning both "bag" and "unsafe" must land in RISK_SIGNAL,
    not BAGGAGE_PROBLEM). First match wins within each tier.
    """
    text = text_clean or ""

    for pattern in RISK_KEYWORD_PATTERNS:
        if pattern.search(text):
            return "RISK_SIGNAL"

    for category, patterns in CATEGORY_KEYWORD_PATTERNS:
        for pattern in patterns:
            if pattern.search(text):
                return category

    return FALLBACK_CATEGORY


def build_pools(pairs):
    """Split thread-root pairs into the risk-signal pool and the 8
    per-category pools. A pair lands in exactly one pool -- risk-signal
    matches never also appear in a category pool."""
    risk_pool = []
    category_pools = {category: [] for category, _ in CATEGORY_KEYWORDS}
    category_pools[FALLBACK_CATEGORY] = []

    for pair in pairs:
        bucket = classify(pair["customer_text_clean"])
        if bucket == "RISK_SIGNAL":
            risk_pool.append(pair)
        else:
            category_pools[bucket].append(pair)

    return risk_pool, category_pools


def sample_up_to(pool, target, seed):
    """Sample up to `target` items reproducibly; return all if fewer are available."""
    if len(pool) > target:
        rng = random.Random(seed)
        return rng.sample(pool, target)
    return list(pool)


def run(input_path, output_path, seed):
    pairs = load_pairs(input_path)
    root_pairs = [p for p in pairs if p.get("is_thread_root") is True]

    risk_pool, category_pools = build_pools(root_pairs)

    rows = []

    sampled_risk = sample_up_to(risk_pool, RISK_POOL_TARGET, seed)
    if len(risk_pool) < RISK_POOL_TARGET:
        print(
            f"WARNING: bucket RISK_SIGNAL has only {len(risk_pool)} candidates, "
            f"fewer than the target of {RISK_POOL_TARGET}. Using all of them."
        )
    for pair in sampled_risk:
        rows.append((pair, "risk_signal"))

    for category, target in CATEGORY_TARGETS.items():
        pool = category_pools.get(category, [])
        sampled = sample_up_to(pool, target, seed)
        if len(pool) < target:
            print(
                f"WARNING: bucket {category} has only {len(pool)} candidates, "
                f"fewer than the target of {target}. Using all of them."
            )
        for pair in sampled:
            rows.append((pair, f"keyword_bucket:{category}"))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)
        for pair, stratum in rows:
            writer.writerow(
                [
                    pair["customer_tweet_id"],
                    stratum,
                    pair["customer_text_raw"],
                    pair["customer_text_clean"],
                    pair["brand_reply_text_raw"],
                    pair["brand_reply_text_clean"],
                    "",  # true_intent
                    "",  # intent_ambiguous
                    "",  # true_escalate
                    "",  # true_escalation_rule
                    "",  # reply_acceptability_notes
                    "",  # labeller_notes
                ]
            )

    stratum_counts = Counter(stratum for _, stratum in rows)
    print(f"Total candidates written: {len(rows)}")
    print("Counts per sampling_stratum:")
    for stratum, count in sorted(stratum_counts.items()):
        print(f"  {stratum}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Sample golden-set labelling candidates.")
    parser.add_argument("--input", default="data/processed/delta_pairs.jsonl")
    parser.add_argument("--output", default="data/golden_set/candidates.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
