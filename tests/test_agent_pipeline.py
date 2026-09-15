"""Tests for src/agent/pipeline.py -- the end-to-end agent wiring.

NOT to be confused with tests/test_pipeline.py, which covers the Phase 1
data pipeline (src/pipeline/*). This file covers src/agent/pipeline.py.

Zero real API calls: classify_intent, top_k_similar, and generate_reply
are all monkeypatched at their point of use in src.agent.pipeline, so
this only verifies the wiring/data flow between stages, not any real
model behavior.
"""
import numpy as np

import src.agent.pipeline as pipeline_module


def test_process_message_wires_all_stages_together(monkeypatch):
    # --- fakes for each of the four stages ---

    def fake_classify_intent(clean_text):
        assert clean_text == "my flight was delayed again"
        return {"intent": "FLIGHT_DISRUPTION", "second_choice": None, "reasoning": "mentions delay"}

    monkeypatch.setattr(pipeline_module, "classify_intent", fake_classify_intent)

    fake_embeddings = np.array([[1.0, 0.0]])
    fake_metadata = [
        {
            "customer_tweet_id": "1",
            "customer_text_clean": "my flight was delayed",
            "brand_reply_text_clean": "So sorry, we've rebooked you.",
        }
    ]

    def fake_load_index(index_dir):
        assert index_dir == "data/processed/retrieval_index"
        return fake_embeddings, fake_metadata

    fake_retrieved = [
        {
            "customer_tweet_id": "1",
            "customer_text_clean": "my flight was delayed",
            "brand_reply_text_clean": "So sorry, we've rebooked you.",
            "similarity_score": 0.91,
        }
    ]

    def fake_top_k_similar(clean_text, embeddings, metadata, k=3):
        assert clean_text == "my flight was delayed again"
        assert embeddings is fake_embeddings
        assert metadata is fake_metadata
        assert k == 3
        return fake_retrieved

    monkeypatch.setattr(pipeline_module, "load_index", fake_load_index)
    monkeypatch.setattr(pipeline_module, "top_k_similar", fake_top_k_similar)

    def fake_generate_reply(clean_text, intent, retrieved_examples):
        assert clean_text == "my flight was delayed again"
        assert intent == "FLIGHT_DISRUPTION"
        assert retrieved_examples == fake_retrieved
        return {
            "reply": "So sorry for the delay, we've rebooked you on the next flight.",
            "grounding_source_indices": [0],
            "confidence_note": "High confidence, strong precedent match.",
        }

    monkeypatch.setattr(pipeline_module, "generate_reply", fake_generate_reply)

    captured_routing_args = {}

    def fake_route_message(raw_text, classify_result, top_similarity):
        captured_routing_args["raw_text"] = raw_text
        captured_routing_args["classify_result"] = classify_result
        captured_routing_args["top_similarity"] = top_similarity
        return {
            "escalate": False,
            "rule": "none_auto_handle",
            "reason": "Routine case with strong historical precedent (similarity=0.91) and confident classification.",
        }

    monkeypatch.setattr(pipeline_module, "route_message", fake_route_message)

    # --- run it ---

    raw_text = "@Delta my flight was delayed again"
    result = pipeline_module.process_message(raw_text, k=3)

    # Cleaning happened: @mention stripped before being passed downstream.
    assert result["clean_text"] == "my flight was delayed again"
    assert result["raw_text"] == raw_text

    # Each stage's real output flowed through into the combined result.
    assert result["classification"]["intent"] == "FLIGHT_DISRUPTION"
    assert result["retrieved_examples"] == fake_retrieved
    assert result["generation"]["reply"].startswith("So sorry for the delay")
    assert result["routing"]["escalate"] is False
    assert result["routing"]["rule"] == "none_auto_handle"

    # Routing was called with raw_text (not clean_text), the real
    # classification result, and the top retrieved similarity score.
    assert captured_routing_args["raw_text"] == raw_text
    assert captured_routing_args["classify_result"]["intent"] == "FLIGHT_DISRUPTION"
    assert captured_routing_args["top_similarity"] == 0.91


def test_process_message_uses_zero_similarity_when_no_examples_retrieved(monkeypatch):
    monkeypatch.setattr(
        pipeline_module,
        "classify_intent",
        lambda clean_text: {"intent": "SERVICE_COMPLAINT", "second_choice": None, "reasoning": "r"},
    )
    monkeypatch.setattr(pipeline_module, "load_index", lambda index_dir: (np.array([]), []))
    monkeypatch.setattr(
        pipeline_module, "top_k_similar", lambda clean_text, embeddings, metadata, k=3: []
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_reply",
        lambda clean_text, intent, retrieved_examples: {
            "reply": "Thanks for reaching out, could you share more detail?",
            "grounding_source_indices": [],
            "confidence_note": "Low confidence, no precedent found.",
        },
    )

    captured = {}

    def fake_route_message(raw_text, classify_result, top_similarity):
        captured["top_similarity"] = top_similarity
        return {"escalate": True, "rule": "weak_grounding", "reason": "no precedent"}

    monkeypatch.setattr(pipeline_module, "route_message", fake_route_message)

    result = pipeline_module.process_message("something totally unrelated")

    assert result["retrieved_examples"] == []
    assert captured["top_similarity"] == 0.0
    assert result["routing"]["rule"] == "weak_grounding"
