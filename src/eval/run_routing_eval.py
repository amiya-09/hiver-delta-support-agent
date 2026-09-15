"""Phase 5 golden-set routing evaluation.

route_message() is pure rule logic (zero API cost) and has only been
unit-tested against synthetic fixtures so far -- this checks it against
the real 240 hand-labelled escalation decisions, two ways:

  ISOLATED: route using the golden true_intent and second_choice=None,
  to test the routing rules on their own, independent of classifier error.

  END-TO-END: route using Phase 3's actual predicted_intent/second_choice,
  to test what a real end-to-end run would have decided.

Retrieval uses the local embedding model (no API call either), so this
whole evaluation is zero-cost to run and re-run.

DOCUMENTED SCOPING LIMIT: in the ISOLATED evaluation, second_choice is
always None by construction, so the "ambiguous_classification" rule can
never fire there -- its rule-level numbers in that mode are not
meaningful. This is a known limit of testing routing in isolation from
classification, not a bug in the rule or the eval.
"""
import argparse
import csv
import json
from pathlib import Path

from src.agent.route import route_message
from src.eval.metrics import compute_classification_metrics
from src.golden_set.constants import ESCALATION_RULES, NONE_AUTO_HANDLE
from src.golden_set.validate_labels import normalize_bool
from src.retrieval.query import load_index, top_k_similar

RULE_LABELS = ESCALATION_RULES + [NONE_AUTO_HANDLE]
BINARY_LABELS = ["ESCALATE", "AUTO_HANDLE"]


def load_golden_set(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_phase3_predictions(path):
    """Load Phase 3's saved predictions, indexed by example_id for lookup."""
    with open(path) as f:
        data = json.load(f)
    return {p["example_id"]: p for p in data["predictions"]}


def _extra_margin_sequence():
    """Buffer margins to add to k when requesting from top_k_similar:
    1, 3, 6, then doubling (12, 24, 48, ...) -- an unbounded but strictly
    increasing sequence; the caller stops it once enough results are
    found or the retrieval pool itself is exhausted."""
    yield 1
    yield 3
    margin = 6
    while True:
        yield margin
        margin *= 2


def get_top_similarity(query_text, example_id, embeddings, metadata, k):
    """Retrieve top-k precedent for one golden-set row, filtering out ALL self-matches.

    The golden set and the retrieval pool were sampled independently from
    the same source data and may overlap -- and Phase 1's dedup was on
    normalized text, not tweet_id directly, so a given example_id is NOT
    guaranteed to appear at most once in the pool. Matching a row against
    itself would trivially return a ~1.0 similarity that isn't real
    precedent, so every self-match must be filtered, not just the first.

    Requests an increasing buffer from top_k_similar (k+1, k+3, k+6, then
    doubling margins) and re-filters from scratch each time, until either
    at least k non-self results are found, or top_k_similar returns fewer
    results than requested (the only reliable signal that the entire
    retrieval pool has been exhausted -- top_k_similar always returns
    exactly min(k, pool_size) results). This never assumes a fixed number
    of self-matches and never loops forever against a small pool.

    Returns (top_k_results, top_similarity, self_matches_removed) --
    top_k_results may have fewer than k entries if the pool genuinely
    doesn't have enough non-self candidates, in which case a warning is
    printed.
    """
    filtered = []
    self_matches_removed = 0
    request_size = 0

    for extra in _extra_margin_sequence():
        request_size = k + extra
        raw_results = top_k_similar(query_text, embeddings, metadata, k=request_size)
        filtered = [r for r in raw_results if r.get("customer_tweet_id") != example_id]
        self_matches_removed = len(raw_results) - len(filtered)

        pool_exhausted = len(raw_results) < request_size
        if len(filtered) >= k or pool_exhausted:
            break

    top_k_results = filtered[:k]
    top_similarity = top_k_results[0]["similarity_score"] if top_k_results else 0.0

    if len(top_k_results) < k:
        print(
            f"WARNING: only {len(top_k_results)}/{k} non-self precedent(s) "
            f"found for example_id {example_id!r} after removing "
            f"{self_matches_removed} self-match(es) (retrieval pool "
            f"exhausted at {request_size} requested)."
        )

    return top_k_results, top_similarity, self_matches_removed


def _escalate_label(escalate):
    return "ESCALATE" if escalate else "AUTO_HANDLE"


def run(golden_set_path, phase3_results_path, retrieval_index_dir, output_path, k):
    rows = load_golden_set(golden_set_path)
    phase3_by_id = load_phase3_predictions(phase3_results_path)
    embeddings, metadata = load_index(retrieval_index_dir)

    overlap_count = 0
    per_example = []

    isolated_true_escalate, isolated_pred_escalate = [], []
    isolated_true_rule, isolated_pred_rule = [], []

    e2e_true_escalate, e2e_pred_escalate = [], []
    e2e_true_rule, e2e_pred_rule = [], []

    for row in rows:
        example_id = row["example_id"]
        raw_text = row["customer_text_raw"]
        query_text = row["customer_text_clean"]
        true_intent = (row.get("true_intent") or "").strip().upper()
        true_escalate = normalize_bool(row.get("true_escalate")) is True
        true_rule = (row.get("true_escalation_rule") or "").strip()

        top_k_results, top_similarity, self_matches_removed = get_top_similarity(
            query_text, example_id, embeddings, metadata, k
        )
        if self_matches_removed:
            overlap_count += 1

        # --- isolated: golden true_intent, second_choice always None ---
        isolated_classify_result = {"intent": true_intent, "second_choice": None}
        isolated_routing = route_message(raw_text, isolated_classify_result, top_similarity)

        isolated_true_escalate.append(_escalate_label(true_escalate))
        isolated_pred_escalate.append(_escalate_label(isolated_routing["escalate"]))
        isolated_true_rule.append(true_rule)
        isolated_pred_rule.append(isolated_routing["rule"])

        # --- end-to-end: Phase 3's actual saved prediction ---
        if example_id not in phase3_by_id:
            raise RuntimeError(
                f"run_routing_eval: no Phase 3 prediction found for example_id "
                f"{example_id!r} in {phase3_results_path} -- cannot run the "
                f"end-to-end evaluation without it. Re-run Phase 3's eval "
                f"first, or check the two files are for the same golden set."
            )
        phase3_pred = phase3_by_id[example_id]
        e2e_classify_result = {
            "intent": phase3_pred["predicted_intent"],
            "second_choice": phase3_pred.get("second_choice"),
        }
        e2e_routing = route_message(raw_text, e2e_classify_result, top_similarity)

        e2e_true_escalate.append(_escalate_label(true_escalate))
        e2e_pred_escalate.append(_escalate_label(e2e_routing["escalate"]))
        e2e_true_rule.append(true_rule)
        e2e_pred_rule.append(e2e_routing["rule"])

        per_example.append(
            {
                "example_id": example_id,
                "self_matches_removed": self_matches_removed,
                "top_similarity": top_similarity,
                "true_escalate": true_escalate,
                "true_escalation_rule": true_rule,
                "isolated": isolated_routing,
                "end_to_end": e2e_routing,
            }
        )

    isolated_binary_metrics = compute_classification_metrics(
        isolated_true_escalate, isolated_pred_escalate, BINARY_LABELS
    )
    isolated_rule_metrics = compute_classification_metrics(
        isolated_true_rule, isolated_pred_rule, RULE_LABELS
    )
    e2e_binary_metrics = compute_classification_metrics(
        e2e_true_escalate, e2e_pred_escalate, BINARY_LABELS
    )
    e2e_rule_metrics = compute_classification_metrics(e2e_true_rule, e2e_pred_rule, RULE_LABELS)

    results = {
        "n_total": len(rows),
        "overlap_count": overlap_count,
        "isolated": {
            "binary_metrics": isolated_binary_metrics,
            "rule_metrics": isolated_rule_metrics,
        },
        "end_to_end": {
            "binary_metrics": e2e_binary_metrics,
            "rule_metrics": e2e_rule_metrics,
        },
        "per_example": per_example,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Total golden-set rows: {len(rows)}")
    print(f"Overlap count (rows with a self-match filtered out of retrieval): {overlap_count}")
    print()

    for label, binary_metrics, rule_metrics in [
        ("ISOLATED (golden true_intent, second_choice=None)", isolated_binary_metrics, isolated_rule_metrics),
        ("END-TO-END (Phase 3 predicted intent/second_choice)", e2e_binary_metrics, e2e_rule_metrics),
    ]:
        print(f"--- {label} ---")
        escalate_stats = binary_metrics["per_label"]["ESCALATE"]
        print(
            f"  ESCALATE precision={escalate_stats['precision']:.4f} "
            f"recall={escalate_stats['recall']:.4f} f1={escalate_stats['f1']:.4f}"
        )
        print(f"  Rule-level accuracy: {rule_metrics['accuracy']:.4f}")
        print("  Rule-level confusion matrix (rows=true, cols=predicted):")
        for true_rule_label in RULE_LABELS:
            print(f"    {true_rule_label}: {rule_metrics['confusion_matrix'].get(true_rule_label, {})}")
        print()

    print(
        "NOTE: in the ISOLATED evaluation, second_choice is always None by "
        "construction, so 'ambiguous_classification' can never fire there -- "
        "its rule-level numbers in that mode are not meaningful. This is a "
        "documented scoping limit of testing routing independent of "
        "classification, not a bug."
    )
    print(f"Full results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 5 routing against the golden set, isolated and end-to-end."
    )
    parser.add_argument("--golden-set", default="data/golden_set/candidates.csv")
    parser.add_argument("--phase3-results", default="data/eval/phase3_classification_results.json")
    parser.add_argument("--retrieval-index-dir", default="data/processed/retrieval_index")
    parser.add_argument("--output", default="data/eval/phase5_routing_results.json")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    run(args.golden_set, args.phase3_results, args.retrieval_index_dir, args.output, args.k)


if __name__ == "__main__":
    main()
