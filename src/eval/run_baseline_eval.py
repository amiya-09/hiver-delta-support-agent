"""Two required baselines for the Phase 3 classification eval, so the LLM
classifier's accuracy has context: a trivial most-common-class predictor,
and the existing rule-based keyword classifier (zero API calls, both).
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.eval.metrics import compute_classification_metrics
from src.golden_set.constants import INTENT_CATEGORIES
from src.golden_set.sample_candidates import classify as keyword_classify

LLM_RESULTS_PATH = "data/eval/phase3_classification_results.json"


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_trivial_baseline(y_true):
    """Predict the single most common true_intent value for every example."""
    mode_intent = Counter(y_true).most_common(1)[0][0]
    y_pred = [mode_intent] * len(y_true)
    metrics = compute_classification_metrics(y_true, y_pred, INTENT_CATEGORIES)
    return mode_intent, metrics


def run_simple_baseline(y_true, texts):
    """Predict via the existing rule-based keyword classifier -- zero API calls."""
    y_pred = [keyword_classify(text) for text in texts]
    metrics = compute_classification_metrics(y_true, y_pred, INTENT_CATEGORIES)
    return metrics


def load_llm_metrics(path):
    with open(path) as f:
        results = json.load(f)
    return results["overall"]


def print_comparison_table(trivial_metrics, simple_metrics, llm_metrics, mode_intent):
    systems = [
        ("Trivial (most-common-class)", trivial_metrics),
        ("Simple (keyword rules)", simple_metrics),
        ("LLM classifier", llm_metrics),
    ]

    print(f"Trivial baseline always predicts: {mode_intent}")
    print()

    name_width = max(len(name) for name, _ in systems) + 2
    print(f"{'System':<{name_width}}{'Accuracy':>10}")
    for name, metrics in systems:
        print(f"{name:<{name_width}}{metrics['accuracy']:>10.4f}")
    print()

    print("Per-category F1:")
    header = f"{'Category':<22}"
    for name, _ in systems:
        header += f"{name:>30}"
    print(header)
    for label in INTENT_CATEGORIES:
        row = f"{label:<22}"
        for _, metrics in systems:
            f1 = metrics["per_label"].get(label, {}).get("f1", 0.0)
            row += f"{f1:>30.4f}"
        print(row)


def run(input_path, output_path):
    rows = load_rows(input_path)
    y_true = [(row.get("true_intent") or "").strip().upper() for row in rows]
    texts = [row.get("customer_text_clean", "") for row in rows]

    mode_intent, trivial_metrics = run_trivial_baseline(y_true)
    simple_metrics = run_simple_baseline(y_true, texts)
    llm_metrics = load_llm_metrics(LLM_RESULTS_PATH)

    print_comparison_table(trivial_metrics, simple_metrics, llm_metrics, mode_intent)

    results = {
        "n_total": len(rows),
        "trivial_baseline": {"predicted_class": mode_intent, "metrics": trivial_metrics},
        "simple_baseline": {"metrics": simple_metrics},
        "llm_classifier": {"metrics": llm_metrics},
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Full results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the trivial + keyword baselines and compare against the LLM classifier."
    )
    parser.add_argument("--input", default="data/golden_set/candidates.csv")
    parser.add_argument("--output", default="data/eval/phase3_baseline_results.json")
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
