import json

import pytest

import src.agent.generate as generate_module


def _fake_response(payload):
    return json.dumps(payload)


def _examples():
    return [
        {
            "customer_tweet_id": "1",
            "customer_text_clean": "my flight was delayed for hours",
            "brand_reply_text_clean": "So sorry, we've rebooked you on the next flight.",
            "similarity_score": 0.81,
        },
        {
            "customer_tweet_id": "2",
            "customer_text_clean": "my bag never arrived",
            "brand_reply_text_clean": "Please DM us your bag claim number.",
            "similarity_score": 0.42,
        },
    ]


# --- valid response parsing ---


def test_generate_reply_parses_valid_json(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(generate_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(
            {
                "reply": "So sorry for the delay -- we've rebooked you on the next available flight.",
                "grounding_source_indices": [0],
                "confidence_note": "High confidence, strong precedent match.",
            }
        )

    monkeypatch.setattr(generate_module, "_call_generate_api", fake_call)

    result = generate_module.generate_reply(
        "my flight was delayed", "FLIGHT_DISRUPTION", _examples(), model="test-model"
    )

    assert result["reply"].startswith("So sorry")
    assert result["grounding_source_indices"] == [0]


def test_generate_reply_strips_markdown_code_fence(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(generate_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        payload = {
            "reply": "Thanks for reaching out, we'll look into this.",
            "grounding_source_indices": [],
            "confidence_note": "Low confidence, no strong precedent.",
        }
        return "```json\n" + json.dumps(payload) + "\n```"

    monkeypatch.setattr(generate_module, "_call_generate_api", fake_call)

    result = generate_module.generate_reply(
        "something unrelated", "SERVICE_COMPLAINT", _examples(), model="test-model"
    )
    assert result["grounding_source_indices"] == []


# --- reply field validation ---


def test_generate_reply_accepts_valid_nonempty_reply(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(generate_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(
            {
                "reply": "So sorry for the delay, we've rebooked you.",
                "grounding_source_indices": [0],
                "confidence_note": "note",
            }
        )

    monkeypatch.setattr(generate_module, "_call_generate_api", fake_call)

    result = generate_module.generate_reply(
        "text", "FLIGHT_DISRUPTION", _examples(), model="test-model"
    )
    assert result["reply"] == "So sorry for the delay, we've rebooked you."


def test_generate_reply_raises_on_empty_string_reply(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(generate_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(
            {"reply": "   ", "grounding_source_indices": [], "confidence_note": "note"}
        )

    monkeypatch.setattr(generate_module, "_call_generate_api", fake_call)

    with pytest.raises(RuntimeError, match="empty or missing reply"):
        generate_module.generate_reply("text", "FLIGHT_DISRUPTION", _examples(), model="test-model")


def test_generate_reply_raises_on_missing_reply_key(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(generate_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response({"grounding_source_indices": [], "confidence_note": "note"})

    monkeypatch.setattr(generate_module, "_call_generate_api", fake_call)

    with pytest.raises(RuntimeError, match="empty or missing reply"):
        generate_module.generate_reply("text", "FLIGHT_DISRUPTION", _examples(), model="test-model")


# --- grounding_source_indices validation ---


def test_generate_reply_accepts_valid_grounding_indices(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(generate_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(
            {
                "reply": "reply text",
                "grounding_source_indices": [0, 1],
                "confidence_note": "note",
            }
        )

    monkeypatch.setattr(generate_module, "_call_generate_api", fake_call)

    result = generate_module.generate_reply(
        "text", "FLIGHT_DISRUPTION", _examples(), model="test-model"
    )
    assert result["grounding_source_indices"] == [0, 1]


def test_generate_reply_accepts_empty_grounding_indices(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(generate_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(
            {
                "reply": "Thanks for reaching out, could you share more detail?",
                "grounding_source_indices": [],
                "confidence_note": "Low confidence, no relevant precedent found.",
            }
        )

    monkeypatch.setattr(generate_module, "_call_generate_api", fake_call)

    result = generate_module.generate_reply(
        "something totally unrelated", "SERVICE_COMPLAINT", _examples(), model="test-model"
    )
    assert result["grounding_source_indices"] == []


def test_generate_reply_raises_on_out_of_range_grounding_index(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(generate_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(
            {
                "reply": "reply text",
                "grounding_source_indices": [5],  # only indices 0-1 exist
                "confidence_note": "note",
            }
        )

    monkeypatch.setattr(generate_module, "_call_generate_api", fake_call)

    with pytest.raises(RuntimeError, match="out-of-range"):
        generate_module.generate_reply("text", "FLIGHT_DISRUPTION", _examples(), model="test-model")


def test_generate_reply_raises_on_out_of_range_index_with_no_examples(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(generate_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(
            {
                "reply": "reply text",
                "grounding_source_indices": [0],  # no examples were given at all
                "confidence_note": "note",
            }
        )

    monkeypatch.setattr(generate_module, "_call_generate_api", fake_call)

    with pytest.raises(RuntimeError, match="out-of-range"):
        generate_module.generate_reply("text", "SERVICE_COMPLAINT", [], model="test-model")


# --- malformed JSON: retry once then raise ---


def test_generate_reply_retries_once_on_malformed_json_then_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(generate_module, "_get_client", lambda: object())

    call_count = {"n": 0}

    def fake_call(client, model, prompt_text):
        call_count["n"] += 1
        return "this is definitely not json"

    monkeypatch.setattr(generate_module, "_call_generate_api", fake_call)

    with pytest.raises(RuntimeError, match="not valid JSON"):
        generate_module.generate_reply("text", "FLIGHT_DISRUPTION", _examples(), model="test-model")

    assert call_count["n"] == 2


def test_generate_reply_recovers_after_one_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(generate_module, "_get_client", lambda: object())

    call_count = {"n": 0}

    def fake_call(client, model, prompt_text):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "not json at all"
        assert prompt_text.endswith("Respond with ONLY the JSON object, no other text.")
        return _fake_response(
            {"reply": "reply text", "grounding_source_indices": [0], "confidence_note": "note"}
        )

    monkeypatch.setattr(generate_module, "_call_generate_api", fake_call)

    result = generate_module.generate_reply(
        "text", "FLIGHT_DISRUPTION", _examples(), model="test-model"
    )

    assert result["reply"] == "reply text"
    assert call_count["n"] == 2


# --- caching ---


def test_generate_reply_caches_and_avoids_second_api_call(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(generate_module, "_get_client", lambda: object())

    call_count = {"n": 0}

    def fake_call(client, model, prompt_text):
        call_count["n"] += 1
        return _fake_response(
            {"reply": "reply text", "grounding_source_indices": [0], "confidence_note": "note"}
        )

    monkeypatch.setattr(generate_module, "_call_generate_api", fake_call)

    first = generate_module.generate_reply(
        "text", "FLIGHT_DISRUPTION", _examples(), model="test-model"
    )
    second = generate_module.generate_reply(
        "text", "FLIGHT_DISRUPTION", _examples(), model="test-model"
    )

    assert first == second
    assert call_count["n"] == 1
