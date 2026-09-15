import csv
import json

import src.eval.run_classification_eval as run_eval_module


def _write_csv(path, rows):
    fieldnames = ["example_id", "customer_text_clean", "true_intent", "intent_ambiguous"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_run_skips_pacing_delay_on_cache_hits(tmp_path, monkeypatch):
    rows = [
        {
            "example_id": "1",
            "customer_text_clean": "my flight was delayed",
            "true_intent": "FLIGHT_DISRUPTION",
            "intent_ambiguous": "FALSE",
        },
        {
            "example_id": "2",
            "customer_text_clean": "my bag is lost",
            "true_intent": "BAGGAGE_PROBLEM",
            "intent_ambiguous": "FALSE",
        },
        {
            "example_id": "3",
            "customer_text_clean": "thank you so much",
            "true_intent": "PRAISE_FEEDBACK",
            "intent_ambiguous": "FALSE",
        },
    ]
    input_path = tmp_path / "candidates.csv"
    _write_csv(input_path, rows)
    output_path = tmp_path / "results.json"

    # Row 1: fresh call (miss) -> pacing should apply after it.
    # Row 2: cache hit -> pacing should be skipped after it.
    # Row 3: last row -> pacing skipped regardless of hit/miss.
    cache_hit_sequence = [False, True, False]
    call_count = {"n": 0}

    def fake_classify_intent(text, return_cache_hit=False):
        idx = call_count["n"]
        call_count["n"] += 1
        was_hit = cache_hit_sequence[idx]
        result = {"intent": rows[idx]["true_intent"], "second_choice": None, "reasoning": "r"}
        return (result, was_hit) if return_cache_hit else result

    monkeypatch.setattr(run_eval_module, "classify_intent", fake_classify_intent)

    sleep_calls = []
    monkeypatch.setattr(run_eval_module.time, "sleep", lambda s: sleep_calls.append(s))

    run_eval_module.run(str(input_path), str(output_path))

    assert sleep_calls == [run_eval_module.PACING_DELAY]
    assert call_count["n"] == 3

    with output_path.open() as f:
        results = json.load(f)
    assert results["n_total"] == 3
    assert results["overall"]["accuracy"] == 1.0
