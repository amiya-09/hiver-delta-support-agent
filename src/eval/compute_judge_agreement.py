"""Judge-vs-human agreement -- Phase 6's second required check.

n is small (currently up to 10, from Phase 4's sample) -- any agreement
figure here is DIRECTIONAL, not statistically robust. This is a
documented scope limitation of a small-sample sanity check, not
something to present as more rigorous than it is.
"""
import argparse
import csv
import json
from pathlib import Path

from src.eval.judge import RUBRIC_FIELDS

HUMAN_COLUMN_BY_FIELD = {field: f"human_{field}" for field in RUBRIC_FIELDS}


def load_judge_results(path):
    with open(path) as f:
        data = json.load(f)
    return {r["example_id"]: r for r in data["results"]}


def load_human_scores(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _parse_human_score(value):
    """Return an int 1-3, or None if blank/non-numeric/out of range.

    Tries a direct int() first; if that fails, falls back to float()
    parsing to handle spreadsheet apps (Excel/Numbers/Sheets) reformatting
    an integer CSV column as a decimal on save -- "3.0" is a realistic,
    recurring value here, not an edge case. A non-whole float (e.g. "2.5")
    is still rejected as genuinely invalid, never silently rounded.
    """
    value = (value or "").strip()
    if not value:
        return None
    try:
        score = int(value)
    except ValueError:
        try:
            float_score = float(value)
        except ValueError:
            return None
        if float_score != int(float_score):
            return None
        score = int(float_score)
    if score not in (1, 2, 3):
        return None
    return score


def is_row_complete(row):
    """True only if ALL FOUR human_* columns parse to a valid 1-3 score."""
    return all(
        _parse_human_score(row.get(column)) is not None
        for column in HUMAN_COLUMN_BY_FIELD.values()
    )


def compute_agreement(judge_results_by_id, human_rows):
    """Compute per-dimension exact-match and within-1 agreement rates.

    Skips rows with incomplete human scoring (any of the 4 human_*
    columns blank or non-numeric) -- reports the skip count explicitly
    rather than crashing or silently including partial rows.

    Returns (per_dimension_stats, n_scored, n_skipped_incomplete).
    """
    complete_rows = []
    skipped_count = 0

    for row in human_rows:
        if is_row_complete(row):
            complete_rows.append(row)
        else:
            skipped_count += 1

    per_dimension = {}
    for field in RUBRIC_FIELDS:
        column = HUMAN_COLUMN_BY_FIELD[field]
        exact_matches = 0
        within_one = 0
        n = 0

        for row in complete_rows:
            example_id = row["example_id"]
            if example_id not in judge_results_by_id:
                raise RuntimeError(
                    f"compute_judge_agreement: no judge result found for "
                    f"example_id {example_id!r} -- the human-scoring CSV "
                    f"and judge-results file don't cover the same examples."
                )
            judge_score = judge_results_by_id[example_id]["scores"][field]
            human_score = _parse_human_score(row[column])

            n += 1
            if judge_score == human_score:
                exact_matches += 1
            if abs(judge_score - human_score) <= 1:
                within_one += 1

        per_dimension[field] = {
            "n": n,
            "exact_match_rate": exact_matches / n if n else 0.0,
            "within_one_rate": within_one / n if n else 0.0,
        }

    return per_dimension, len(complete_rows), skipped_count


def run(judge_results_path, human_scores_path, output_path):
    judge_results_by_id = load_judge_results(judge_results_path)
    human_rows = load_human_scores(human_scores_path)

    per_dimension, n_scored, n_skipped = compute_agreement(judge_results_by_id, human_rows)

    results = {
        "n_scored": n_scored,
        "n_skipped_incomplete": n_skipped,
        "per_dimension": per_dimension,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Rows with complete human scoring: {n_scored}")
    print(f"Rows skipped (incomplete human scoring): {n_skipped}")
    print("Per-dimension judge-vs-human agreement:")
    for field, stats in per_dimension.items():
        print(
            f"  {field}: exact_match={stats['exact_match_rate']:.2f} "
            f"within_1={stats['within_one_rate']:.2f} (n={stats['n']})"
        )
    print()
    print(
        f"NOTE: n={n_scored} is a small sample -- any agreement figure "
        f"above is DIRECTIONAL, not statistically robust. This is a "
        f"documented scope limitation, not something to present as more "
        f"rigorous than it is."
    )
    print(f"Full results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute judge-vs-human agreement on the rubric dimensions."
    )
    parser.add_argument("--judge-results", default="data/eval/phase6_judge_results.json")
    parser.add_argument("--human-scores", default="data/eval/human_scoring_template.csv")
    parser.add_argument("--output", default="data/eval/phase6_judge_agreement.json")
    args = parser.parse_args()
    run(args.judge_results, args.human_scores, args.output)


if __name__ == "__main__":
    main()
