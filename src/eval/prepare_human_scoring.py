"""Prepare a blind human-scoring template from Phase 4's generated replies.

CRITICAL: this file must NOT include any judge scores from
run_judge_eval.py's output -- the human must score blind to the judge's
opinion, or the agreement measurement in compute_judge_agreement.py is
meaningless.
"""
import argparse
import csv
import json
from pathlib import Path

CSV_FIELDS = [
    "example_id",
    "customer_text",
    "generated_reply",
    "retrieved_examples_summary",
    "human_grounded",
    "human_factually_consistent",
    "human_tone_appropriate",
    "human_actionable",
    "human_notes",
]


def load_generation_results(path):
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def summarize_retrieved_examples(retrieved_examples):
    """Short, readable rendering of what was retrieved, for human context
    -- deliberately terse (unlike judge.py's full formatted block), since
    this is just orientation for a human, not a model prompt."""
    if not retrieved_examples:
        return "(none retrieved)"

    parts = []
    for i, example in enumerate(retrieved_examples):
        similarity = example.get("similarity_score")
        similarity_str = f"{similarity:.2f}" if isinstance(similarity, (int, float)) else "?"
        customer_text = example.get("customer_text_clean", "")
        parts.append(f"[{i}, sim={similarity_str}] {customer_text}")
    return " || ".join(parts)


def run(input_path, output_path):
    generation_results = load_generation_results(input_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)
        for result in generation_results:
            writer.writerow(
                [
                    result.get("example_id"),
                    result.get("customer_text_clean"),
                    result.get("reply"),
                    summarize_retrieved_examples(result.get("retrieved_examples", [])),
                    "",  # human_grounded
                    "",  # human_factually_consistent
                    "",  # human_tone_appropriate
                    "",  # human_actionable
                    "",  # human_notes
                ]
            )

    print(f"Wrote {len(generation_results)} rows to {output_path} for blind human scoring.")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a blind human-scoring CSV template from Phase 4's generated replies."
    )
    parser.add_argument("--input", default="data/eval/phase4_generation_results.json")
    parser.add_argument("--output", default="data/eval/human_scoring_template.csv")
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
