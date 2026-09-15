"""Targeted recheck after a classify_intent prompt/logic fix: does it
actually fix the earlier misclassifications, and does it break anything
that used to be correct?

Reuses classify_intent's existing caching/pacing/retry logic as-is --
this is just a different slice of examples to run it over.
"""
import argparse
import csv
import json
import random
from pathlib import Path

from src.agent.classify import classify_intent

GOLDEN_SET_PATH = "data/golden_set/candidates.csv"


def load_previous_predictions(path):
    with open(path) as f:
        data = json.load(f)
    return data["predictions"]


def load_text_by_example_id(golden_set_path):
    with open(golden_set_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["example_id"]: row["customer_text_clean"] for row in rows}


def split_predictions(predictions, control_size, seed):
    """Split previous predictions into (previously_wrong, control_sample).

    previously_wrong: every prediction where correct was False -- the
    actual misclassifications the fix is meant to address.
    control_sample: up to control_size predictions sampled reproducibly
    (via seed) from those where correct was True, to check the fix didn't
    break anything that used to work.
    """
    previously_wrong = [p for p in predictions if p.get("correct") is False]
    previously_right = [p for p in predictions if p.get("correct") is True]

    if len(previously_right) > control_size:
        rng = random.Random(seed)
        control_sample = rng.sample(previously_right, control_size)
    else:
        control_sample = list(previously_right)

    return previously_wrong, control_sample


def recheck_group(predictions, text_by_id, group_name="group"):
    """Re-run classify_intent on a group of previous predictions.

    Prints one progress line per example as it completes -- previously the
    only output during a run was rate-limit warnings, which made a slow
    run indistinguishable from a stuck one.
    """
    results = []
    now_correct_count = 0
    total = len(predictions)
    for i, pred in enumerate(predictions, start=1):
        example_id = pred["example_id"]
        text = text_by_id.get(example_id, "")
        true_intent = pred["true_intent"]
        was_correct_before = pred.get("correct")

        result = classify_intent(text)
        predicted_intent = result["intent"]
        now_correct = predicted_intent == true_intent
        if now_correct:
            now_correct_count += 1

        print(
            f"[{group_name}] {i}/{total}: {example_id} -- "
            f"was_correct={was_correct_before}, now_correct={now_correct}"
        )

        results.append(
            {
                "example_id": example_id,
                "true_intent": true_intent,
                "previous_predicted_intent": pred.get("predicted_intent"),
                "previous_correct": pred.get("correct"),
                "new_predicted_intent": predicted_intent,
                "new_correct": now_correct,
            }
        )

    return results, now_correct_count


def run(previous_path, output_path, control_size, seed):
    predictions = load_previous_predictions(previous_path)
    text_by_id = load_text_by_example_id(GOLDEN_SET_PATH)

    previously_wrong, control_sample = split_predictions(predictions, control_size, seed)

    wrong_results, fixed_count = recheck_group(previously_wrong, text_by_id, group_name="previously_wrong")
    control_results, still_correct_count = recheck_group(control_sample, text_by_id, group_name="control")

    results = {
        "n_previously_wrong": len(previously_wrong),
        "n_fixed": fixed_count,
        "n_control_sample": len(control_sample),
        "n_still_correct": still_correct_count,
        "previously_wrong_recheck": wrong_results,
        "control_sample_recheck": control_results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Previously wrong: {len(previously_wrong)} examples")
    print(f"  Now correct (fixed): {fixed_count}/{len(previously_wrong)}")
    print(f"Control sample (previously correct): {len(control_sample)} examples")
    print(f"  Still correct (no regression): {still_correct_count}/{len(control_sample)}")
    print(f"Full results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recheck previous misclassifications and a control sample after a classify_intent fix."
    )
    parser.add_argument("--previous", default="data/eval/phase3_classification_results.json")
    parser.add_argument("--output", default="data/eval/phase3_targeted_recheck.json")
    parser.add_argument("--control-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.previous, args.output, args.control_size, args.seed)


if __name__ == "__main__":
    main()
