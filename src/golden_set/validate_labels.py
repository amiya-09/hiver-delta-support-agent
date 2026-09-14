"""Validate hand-labelled golden-set rows before they're trusted as ground truth."""
import argparse
import csv
from collections import Counter

from src.golden_set.constants import ESCALATION_RULES, INTENT_CATEGORIES, NONE_AUTO_HANDLE

MIN_NOTES_LENGTH = 3

TRUE_VALUES = {"true", "1", "yes"}
FALSE_VALUES = {"false", "0", "no"}


def normalize_bool(value):
    """Return True/False for a recognized boolean string, else None."""
    normalized = (value or "").strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def validate_row(row):
    """Return a list of human-readable problem strings for one labelled row."""
    problems = []

    true_intent = (row.get("true_intent") or "").strip()
    if not true_intent:
        problems.append("true_intent is empty")
    elif true_intent.upper() not in INTENT_CATEGORIES:
        problems.append(f"true_intent is not a valid category: {true_intent!r}")

    ambiguous = normalize_bool(row.get("intent_ambiguous"))
    if ambiguous is None:
        problems.append(
            f"intent_ambiguous is not a valid boolean: {row.get('intent_ambiguous')!r}"
        )

    escalate = normalize_bool(row.get("true_escalate"))
    if escalate is None:
        problems.append(
            f"true_escalate is not a valid boolean: {row.get('true_escalate')!r}"
        )

    escalation_rule = (row.get("true_escalation_rule") or "").strip()
    if escalate is True:
        if not escalation_rule or escalation_rule not in ESCALATION_RULES:
            problems.append(
                f"true_escalate is True but true_escalation_rule is invalid: "
                f"{escalation_rule!r} (must be one of {ESCALATION_RULES})"
            )
    elif escalate is False:
        if escalation_rule != NONE_AUTO_HANDLE:
            problems.append(
                f"true_escalate is False but true_escalation_rule is "
                f"{escalation_rule!r}, expected {NONE_AUTO_HANDLE!r}"
            )

    notes = (row.get("reply_acceptability_notes") or "").strip()
    if len(notes) < MIN_NOTES_LENGTH:
        problems.append("reply_acceptability_notes is empty or too short")

    return problems


def validate_rows(rows):
    """Return {example_id: [problems]} for every row with at least one problem."""
    problems_by_id = {}
    for row in rows:
        problems = validate_row(row)
        if problems:
            example_id = row.get("example_id", "<unknown>")
            problems_by_id[example_id] = problems
    return problems_by_id


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def print_summary(rows):
    intent_counts = Counter((row.get("true_intent") or "").strip().upper() for row in rows)
    rule_counts = Counter((row.get("true_escalation_rule") or "").strip() for row in rows)
    ambiguous_true = sum(
        1 for row in rows if normalize_bool(row.get("intent_ambiguous")) is True
    )
    escalate_true = sum(1 for row in rows if normalize_bool(row.get("true_escalate")) is True)
    escalate_false = sum(1 for row in rows if normalize_bool(row.get("true_escalate")) is False)

    print(f"Total labelled rows: {len(rows)}")
    print("Counts per true_intent:")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")
    print("Counts per true_escalation_rule:")
    for rule, count in sorted(rule_counts.items()):
        print(f"  {rule}: {count}")
    print(f"intent_ambiguous=True: {ambiguous_true}")
    print(f"true_escalate=True: {escalate_true}, true_escalate=False: {escalate_false}")


def run(input_path):
    rows = load_rows(input_path)
    problems_by_id = validate_rows(rows)

    if problems_by_id:
        print(f"Found problems in {len(problems_by_id)} row(s):")
        for example_id, problems in problems_by_id.items():
            print(f"  {example_id}:")
            for problem in problems:
                print(f"    - {problem}")
    else:
        print_summary(rows)


def main():
    parser = argparse.ArgumentParser(description="Validate a hand-labelled golden-set CSV.")
    parser.add_argument("--input", default="data/golden_set/candidates.csv")
    args = parser.parse_args()
    run(args.input)


if __name__ == "__main__":
    main()
