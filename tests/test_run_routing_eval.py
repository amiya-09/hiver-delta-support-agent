import csv
import json

import pytest

import src.eval.run_routing_eval as routing_eval_module


# --- self-match filtering logic ---


def test_get_top_similarity_filters_self_match_and_counts_it(monkeypatch):
    def fake_top_k_similar(query_text, embeddings, metadata, k):
        assert k == 4  # k+1 requested for k=3
        return [
            {"customer_tweet_id": "self-id", "similarity_score": 0.99},
            {"customer_tweet_id": "other-1", "similarity_score": 0.85},
            {"customer_tweet_id": "other-2", "similarity_score": 0.80},
            {"customer_tweet_id": "other-3", "similarity_score": 0.75},
        ]

    monkeypatch.setattr(routing_eval_module, "top_k_similar", fake_top_k_similar)

    top_k_results, top_similarity, self_matches_removed = routing_eval_module.get_top_similarity(
        "query text", "self-id", embeddings=None, metadata=None, k=3
    )

    assert self_matches_removed == 1
    assert len(top_k_results) == 3
    assert all(r["customer_tweet_id"] != "self-id" for r in top_k_results)
    # Top of what remains after filtering is "other-1" at 0.85, not the
    # filtered-out self-match's 0.99.
    assert top_similarity == 0.85


def test_get_top_similarity_no_self_match_present(monkeypatch):
    def fake_top_k_similar(query_text, embeddings, metadata, k):
        return [
            {"customer_tweet_id": "other-1", "similarity_score": 0.90},
            {"customer_tweet_id": "other-2", "similarity_score": 0.85},
            {"customer_tweet_id": "other-3", "similarity_score": 0.80},
            {"customer_tweet_id": "other-4", "similarity_score": 0.75},
        ]

    monkeypatch.setattr(routing_eval_module, "top_k_similar", fake_top_k_similar)

    top_k_results, top_similarity, self_matches_removed = routing_eval_module.get_top_similarity(
        "query text", "self-id", embeddings=None, metadata=None, k=3
    )

    assert self_matches_removed == 0
    assert len(top_k_results) == 3
    assert top_similarity == 0.90


def test_get_top_similarity_empty_results_returns_zero_similarity(monkeypatch):
    monkeypatch.setattr(
        routing_eval_module, "top_k_similar", lambda query_text, embeddings, metadata, k: []
    )

    top_k_results, top_similarity, self_matches_removed = routing_eval_module.get_top_similarity(
        "query text", "self-id", embeddings=None, metadata=None, k=3
    )

    assert top_k_results == []
    assert top_similarity == 0.0
    assert self_matches_removed == 0


def test_get_top_similarity_filters_duplicate_self_matches_via_buffer_expansion(monkeypatch):
    # Simulates a pool where the same example_id appears TWICE (a
    # duplicate not caught by Phase 1's text-based dedup) -- the first,
    # small request (k+1=4) doesn't leave enough non-self results, so this
    # must expand the buffer (k+3=6) and retry rather than settling for
    # fewer than k results when the pool actually has enough.
    full_pool = [
        {"customer_tweet_id": "dup-id", "similarity_score": 0.99},
        {"customer_tweet_id": "dup-id", "similarity_score": 0.97},
        {"customer_tweet_id": "other-1", "similarity_score": 0.85},
        {"customer_tweet_id": "other-2", "similarity_score": 0.80},
        {"customer_tweet_id": "other-3", "similarity_score": 0.75},
        {"customer_tweet_id": "other-4", "similarity_score": 0.70},
    ]
    requested_sizes = []

    def fake_top_k_similar(query_text, embeddings, metadata, k):
        requested_sizes.append(k)
        return full_pool[:k]

    monkeypatch.setattr(routing_eval_module, "top_k_similar", fake_top_k_similar)

    top_k_results, top_similarity, self_matches_removed = routing_eval_module.get_top_similarity(
        "query text", "dup-id", embeddings=None, metadata=None, k=3
    )

    # First request (k+1=4) only surfaces 2 non-self results (both
    # duplicates land in the top 4) -- not enough, so it must have
    # retried with a bigger buffer (k+3=6) to find the 3rd.
    assert requested_sizes == [4, 6]

    assert self_matches_removed == 2
    assert len(top_k_results) == 3
    assert all(r["customer_tweet_id"] != "dup-id" for r in top_k_results)
    assert top_similarity == 0.85


def test_get_top_similarity_warns_and_returns_fewer_than_k_when_pool_exhausted(monkeypatch, capsys):
    # The entire pool only has 2 non-self candidates, less than k=3 --
    # must stop retrying (pool exhausted) and return what's available,
    # with a warning, rather than looping forever.
    full_pool = [
        {"customer_tweet_id": "dup-id", "similarity_score": 0.99},
        {"customer_tweet_id": "other-1", "similarity_score": 0.85},
        {"customer_tweet_id": "other-2", "similarity_score": 0.80},
    ]

    def fake_top_k_similar(query_text, embeddings, metadata, k):
        return full_pool[: min(k, len(full_pool))]

    monkeypatch.setattr(routing_eval_module, "top_k_similar", fake_top_k_similar)

    top_k_results, top_similarity, self_matches_removed = routing_eval_module.get_top_similarity(
        "query text", "dup-id", embeddings=None, metadata=None, k=3
    )

    assert self_matches_removed == 1
    assert len(top_k_results) == 2
    assert top_similarity == 0.85

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "2/3" in out


# --- isolated vs end-to-end: correct intent source ---


def _write_golden_csv(path, row):
    fieldnames = [
        "example_id",
        "customer_text_raw",
        "customer_text_clean",
        "true_intent",
        "true_escalate",
        "true_escalation_rule",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def test_run_isolated_uses_golden_label_end_to_end_uses_phase3_prediction(tmp_path, monkeypatch):
    golden_csv = tmp_path / "candidates.csv"
    _write_golden_csv(
        golden_csv,
        {
            "example_id": "1",
            "customer_text_raw": "@Delta my flight was delayed",
            "customer_text_clean": "my flight was delayed",
            "true_intent": "FLIGHT_DISRUPTION",
            "true_escalate": "FALSE",
            "true_escalation_rule": "none_auto_handle",
        },
    )

    phase3_path = tmp_path / "phase3.json"
    phase3_path.write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "example_id": "1",
                        "true_intent": "FLIGHT_DISRUPTION",
                        # Deliberately different from the golden label, so we
                        # can tell which source each evaluation actually used.
                        "predicted_intent": "SERVICE_COMPLAINT",
                        "correct": False,
                        "second_choice": "BAGGAGE_PROBLEM",
                        "intent_ambiguous": False,
                    }
                ]
            }
        )
    )

    monkeypatch.setattr(routing_eval_module, "load_index", lambda path: (None, None))
    monkeypatch.setattr(
        routing_eval_module,
        "top_k_similar",
        lambda query_text, embeddings, metadata, k: [
            {"customer_tweet_id": "other", "similarity_score": 0.90}
        ],
    )

    captured_calls = []

    def fake_route_message(raw_text, classify_result, top_similarity):
        captured_calls.append(dict(classify_result))
        return {"escalate": False, "rule": "none_auto_handle", "reason": "r"}

    monkeypatch.setattr(routing_eval_module, "route_message", fake_route_message)

    output_path = tmp_path / "results.json"
    routing_eval_module.run(str(golden_csv), str(phase3_path), "unused-index-dir", str(output_path), k=3)

    assert len(captured_calls) == 2
    isolated_call, e2e_call = captured_calls

    assert isolated_call["intent"] == "FLIGHT_DISRUPTION"
    assert isolated_call["second_choice"] is None

    assert e2e_call["intent"] == "SERVICE_COMPLAINT"
    assert e2e_call["second_choice"] == "BAGGAGE_PROBLEM"


def test_run_raises_clear_error_when_phase3_prediction_missing(tmp_path, monkeypatch):
    golden_csv = tmp_path / "candidates.csv"
    _write_golden_csv(
        golden_csv,
        {
            "example_id": "missing-id",
            "customer_text_raw": "@Delta hello",
            "customer_text_clean": "hello",
            "true_intent": "SERVICE_COMPLAINT",
            "true_escalate": "FALSE",
            "true_escalation_rule": "none_auto_handle",
        },
    )

    phase3_path = tmp_path / "phase3.json"
    phase3_path.write_text(json.dumps({"predictions": []}))

    monkeypatch.setattr(routing_eval_module, "load_index", lambda path: (None, None))
    monkeypatch.setattr(
        routing_eval_module, "top_k_similar", lambda query_text, embeddings, metadata, k: []
    )

    output_path = tmp_path / "results.json"
    with pytest.raises(RuntimeError, match="no Phase 3 prediction"):
        routing_eval_module.run(
            str(golden_csv), str(phase3_path), "unused-index-dir", str(output_path), k=3
        )
