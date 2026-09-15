"""Run intent classification over the golden set and score it.

Reports overall metrics AND metrics excluding intent_ambiguous rows side
by side on purpose -- accuracy looks different once genuinely ambiguous
cases are excluded, and that's a "misleading headline number" worth
surfacing from day one rather than discovering later in the report.
"""
import argparse
import csv
import json
import time
from pathlib import Path

from src.agent.classify import classify_intent
from src.eval.metrics import compute_classification_metrics
from src.golden_set.constants import INTENT_CATEGORIES
from src.golden_set.validate_labels import normalize_bool

# Groq's free tier is 30 RPM (one request every 2s exactly) -- 2.5s leaves
# a small safety margin rather than pacing right at the edge of the limit.
PACING_DELAY = 2.5
PROGRESS_INTERVAL = 20


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run(input_path, output_path):
    rows = load_rows(input_path)

    y_true = []
    y_pred = []
    is_ambiguous = []
    predictions = []

    for i, row in enumerate(rows):
        true_intent = (row.get("true_intent") or "").strip().upper()
        text = row.get("customer_text_clean", "")

        result, was_cache_hit = classify_intent(text, return_cache_hit=True)
        predicted_intent = result["intent"]

        y_true.append(true_intent)
        y_pred.append(predicted_intent)
        ambiguous = normalize_bool(row.get("intent_ambiguous")) is True
        is_ambiguous.append(ambiguous)

        predictions.append(
            {
                "example_id": row.get("example_id"),
                "true_intent": true_intent,
                "predicted_intent": predicted_intent,
                "correct": true_intent == predicted_intent,
                "second_choice": result.get("second_choice"),
                "reasoning": result.get("reasoning"),
                "intent_ambiguous": ambiguous,
            }
        )

        processed = i + 1
        if processed % PROGRESS_INTERVAL == 0 or processed == len(rows):
            print(f"Classified {processed}/{len(rows)} examples...")

        # Pacing delay between classify_intent calls -- kept here rather than
        # inside classify_intent itself, since classify_intent is also
        # called individually elsewhere and shouldn't always pay this cost.
        # Skipped entirely on a cache hit: no API call happened, so there's
        # nothing to pace against.
        if processed < len(rows) and not was_cache_hit:
            time.sleep(PACING_DELAY)

    overall_metrics = compute_classification_metrics(y_true, y_pred, INTENT_CATEGORIES)

    clean_indices = [i for i, ambiguous in enumerate(is_ambiguous) if not ambiguous]
    y_true_clean = [y_true[i] for i in clean_indices]
    y_pred_clean = [y_pred[i] for i in clean_indices]
    clean_metrics = compute_classification_metrics(y_true_clean, y_pred_clean, INTENT_CATEGORIES)

    n_ambiguous = len(rows) - len(clean_indices)

    results = {
        "n_total": len(rows),
        "n_ambiguous_excluded": n_ambiguous,
        "overall": overall_metrics,
        "excluding_ambiguous": clean_metrics,
        "predictions": predictions,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Total examples: {len(rows)}")
    print(f"Overall accuracy: {overall_metrics['accuracy']:.4f}")
    print(
        f"Accuracy excluding {n_ambiguous} ambiguous example(s): "
        f"{clean_metrics['accuracy']:.4f}"
    )
    print("Per-category F1 (overall):")
    for label in INTENT_CATEGORIES:
        f1 = overall_metrics["per_label"][label]["f1"]
        print(f"  {label}: {f1:.4f}")
    print("Confusion matrix (overall, rows=true, cols=predicted):")
    for true_label in INTENT_CATEGORIES:
        print(f"  {true_label}: {overall_metrics['confusion_matrix'].get(true_label, {})}")
    print(f"Full results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 3 intent classification eval against the golden set."
    )
    parser.add_argument("--input", default="data/golden_set/candidates.csv")
    parser.add_argument("--output", default="data/eval/phase3_classification_results.json")
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
