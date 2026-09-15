import json

import pytest

import src.eval.judge as judge_module


def _fake_response(payload):
    return json.dumps(payload)


def _examples():
    return [
        {
            "customer_text_clean": "my flight was delayed for hours",
            "brand_reply_text_clean": "So sorry, we've rebooked you on the next flight.",
            "similarity_score": 0.81,
        }
    ]


def _valid_scores(**overrides):
    scores = {
        "grounded": 3,
        "factually_consistent": 3,
        "tone_appropriate": 2,
        "actionable": 2,
        "notes": "Reasonable reply, grounded in precedent.",
    }
    scores.update(overrides)
    return scores


# --- valid response parsing ---


def test_judge_reply_parses_valid_json(monkeypatch, tmp_path):
    monkeypatch.setattr(judge_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(judge_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(_valid_scores())

    monkeypatch.setattr(judge_module, "_call_generate_api", fake_call)

    result = judge_module.judge_reply(
        "my flight was delayed",
        "So sorry, we've rebooked you.",
        _examples(),
        model="test-model",
    )

    assert result["grounded"] == 3
    assert result["factually_consistent"] == 3
    assert result["tone_appropriate"] == 2
    assert result["actionable"] == 2


def test_judge_reply_strips_markdown_code_fence(monkeypatch, tmp_path):
    monkeypatch.setattr(judge_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(judge_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return "```json\n" + json.dumps(_valid_scores()) + "\n```"

    monkeypatch.setattr(judge_module, "_call_generate_api", fake_call)

    result = judge_module.judge_reply(
        "text", "reply", _examples(), model="test-model"
    )
    assert result["grounded"] == 3


# --- rubric field range validation ---


@pytest.mark.parametrize("field", judge_module.RUBRIC_FIELDS)
def test_judge_reply_raises_on_score_zero(monkeypatch, tmp_path, field):
    monkeypatch.setattr(judge_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(judge_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(_valid_scores(**{field: 0}))

    monkeypatch.setattr(judge_module, "_call_generate_api", fake_call)

    with pytest.raises(RuntimeError, match=field):
        judge_module.judge_reply("text", "reply", _examples(), model="test-model")


@pytest.mark.parametrize("field", judge_module.RUBRIC_FIELDS)
def test_judge_reply_raises_on_score_four(monkeypatch, tmp_path, field):
    monkeypatch.setattr(judge_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(judge_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(_valid_scores(**{field: 4}))

    monkeypatch.setattr(judge_module, "_call_generate_api", fake_call)

    with pytest.raises(RuntimeError, match=field):
        judge_module.judge_reply("text", "reply", _examples(), model="test-model")


@pytest.mark.parametrize("field", judge_module.RUBRIC_FIELDS)
def test_judge_reply_raises_on_non_integer_score(monkeypatch, tmp_path, field):
    monkeypatch.setattr(judge_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(judge_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(_valid_scores(**{field: "3"}))  # string, not int

    monkeypatch.setattr(judge_module, "_call_generate_api", fake_call)

    with pytest.raises(RuntimeError, match=field):
        judge_module.judge_reply("text", "reply", _examples(), model="test-model")


def test_judge_reply_raises_on_missing_field(monkeypatch, tmp_path):
    monkeypatch.setattr(judge_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(judge_module, "_get_client", lambda: object())

    scores = _valid_scores()
    del scores["actionable"]

    def fake_call(client, model, prompt_text):
        return _fake_response(scores)

    monkeypatch.setattr(judge_module, "_call_generate_api", fake_call)

    with pytest.raises(RuntimeError, match="actionable"):
        judge_module.judge_reply("text", "reply", _examples(), model="test-model")


def test_judge_reply_rejects_boolean_masquerading_as_score(monkeypatch, tmp_path):
    # Python quirk: isinstance(True, int) is True and True == 1, so a
    # literal JSON `true` for a rubric field must still be rejected.
    monkeypatch.setattr(judge_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(judge_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(_valid_scores(grounded=True))

    monkeypatch.setattr(judge_module, "_call_generate_api", fake_call)

    with pytest.raises(RuntimeError, match="grounded"):
        judge_module.judge_reply("text", "reply", _examples(), model="test-model")


# --- malformed JSON: retry once then raise ---


def test_judge_reply_retries_once_on_malformed_json_then_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(judge_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(judge_module, "_get_client", lambda: object())

    call_count = {"n": 0}

    def fake_call(client, model, prompt_text):
        call_count["n"] += 1
        return "this is definitely not json"

    monkeypatch.setattr(judge_module, "_call_generate_api", fake_call)

    with pytest.raises(RuntimeError, match="not valid JSON"):
        judge_module.judge_reply("text", "reply", _examples(), model="test-model")

    assert call_count["n"] == 2


def test_judge_reply_recovers_after_one_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(judge_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(judge_module, "_get_client", lambda: object())

    call_count = {"n": 0}

    def fake_call(client, model, prompt_text):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "not json at all"
        assert prompt_text.endswith("Respond with ONLY the JSON object, no other text.")
        return _fake_response(_valid_scores())

    monkeypatch.setattr(judge_module, "_call_generate_api", fake_call)

    result = judge_module.judge_reply("text", "reply", _examples(), model="test-model")

    assert result["grounded"] == 3
    assert call_count["n"] == 2


# --- caching ---


def test_judge_reply_caches_and_avoids_second_api_call(monkeypatch, tmp_path):
    monkeypatch.setattr(judge_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(judge_module, "_get_client", lambda: object())

    call_count = {"n": 0}

    def fake_call(client, model, prompt_text):
        call_count["n"] += 1
        return _fake_response(_valid_scores())

    monkeypatch.setattr(judge_module, "_call_generate_api", fake_call)

    first = judge_module.judge_reply("text", "reply", _examples(), model="test-model")
    second = judge_module.judge_reply("text", "reply", _examples(), model="test-model")

    assert first == second
    assert call_count["n"] == 1
