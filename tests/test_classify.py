import json

import pytest
from groq import RateLimitError as GroqRateLimitError

import src.agent.classify as classify_module


class FakeGroqResponse:
    """Stands in for httpx.Response -- just enough for `.headers.get(...)`."""

    def __init__(self, headers):
        self.headers = headers


class FakeRateLimitError(GroqRateLimitError):
    """A REAL groq.RateLimitError subclass -- classify.py's retry logic now
    does `except RateLimitError`, so the fake must satisfy isinstance(),
    not just look similar. We skip the real APIStatusError.__init__ (which
    requires a full response.request/status_code/body) and just set the
    `.response.headers`/`.body` shape our parsers actually read."""

    def __init__(self, headers, body=None):
        self.response = FakeGroqResponse(headers)
        self.body = body
        Exception.__init__(self, "rate limited")


def _fake_response(payload):
    return json.dumps(payload)


def test_classify_intent_parses_valid_json(monkeypatch, tmp_path):
    monkeypatch.setattr(classify_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(classify_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(
            {"intent": "FLIGHT_DISRUPTION", "second_choice": None, "reasoning": "mentions a delay"}
        )

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call)

    result = classify_module.classify_intent("my flight was delayed", model="test-model")

    assert result["intent"] == "FLIGHT_DISRUPTION"
    assert result["reasoning"] == "mentions a delay"


def test_classify_intent_strips_markdown_code_fence(monkeypatch, tmp_path):
    monkeypatch.setattr(classify_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(classify_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        payload = {"intent": "PRAISE_FEEDBACK", "second_choice": None, "reasoning": "says thanks"}
        return "```json\n" + json.dumps(payload) + "\n```"

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call)

    result = classify_module.classify_intent("thank you so much", model="test-model")
    assert result["intent"] == "PRAISE_FEEDBACK"


def test_classify_intent_retries_once_on_malformed_json_then_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(classify_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(classify_module, "_get_client", lambda: object())

    call_count = {"n": 0}

    def fake_call(client, model, prompt_text):
        call_count["n"] += 1
        return "this is definitely not json"

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call)

    with pytest.raises(RuntimeError, match="not valid JSON"):
        classify_module.classify_intent("some text", model="test-model")

    # Exactly one retry: the initial attempt + one retry, no more.
    assert call_count["n"] == 2


def test_classify_intent_recovers_after_one_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(classify_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(classify_module, "_get_client", lambda: object())

    call_count = {"n": 0}

    def fake_call(client, model, prompt_text):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "not json at all"
        assert prompt_text.endswith("Respond with ONLY the JSON object, no other text.")
        return _fake_response(
            {"intent": "BAGGAGE_PROBLEM", "second_choice": None, "reasoning": "bag lost"}
        )

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call)

    result = classify_module.classify_intent("my bag is lost", model="test-model")

    assert result["intent"] == "BAGGAGE_PROBLEM"
    assert call_count["n"] == 2


def test_classify_intent_raises_on_invalid_intent_name(monkeypatch, tmp_path):
    monkeypatch.setattr(classify_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(classify_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(
            {"intent": "NOT_A_REAL_CATEGORY", "second_choice": None, "reasoning": "..."}
        )

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call)

    with pytest.raises(RuntimeError, match="invalid intent"):
        classify_module.classify_intent("some text", model="test-model")


def test_classify_intent_raises_on_invalid_second_choice(monkeypatch, tmp_path):
    monkeypatch.setattr(classify_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(classify_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(
            {
                "intent": "FLIGHT_DISRUPTION",
                "second_choice": "NOT_A_REAL_CATEGORY",
                "reasoning": "...",
            }
        )

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call)

    with pytest.raises(RuntimeError, match="invalid second_choice"):
        classify_module.classify_intent("some text", model="test-model")


def test_classify_intent_normalizes_second_choice_equal_to_intent(monkeypatch, tmp_path):
    monkeypatch.setattr(classify_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(classify_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(
            {
                "intent": "FLIGHT_DISRUPTION",
                "second_choice": "FLIGHT_DISRUPTION",
                "reasoning": "flight was delayed",
            }
        )

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call)

    result = classify_module.classify_intent("my flight was delayed", model="test-model")

    assert result["second_choice"] is None

    # Confirm the normalization is what got cached too, not just what's
    # returned this call.
    cached = classify_module._read_cache(
        "test-model", classify_module._load_prompt_template().replace(
            "{{MESSAGE_TEXT}}", "my flight was delayed"
        )
    )
    assert cached["second_choice"] is None


@pytest.mark.parametrize("stringified_null", ["null", "none", "Null", "NONE", "  null  "])
def test_classify_intent_normalizes_stringified_null_second_choice(
    monkeypatch, tmp_path, stringified_null
):
    # Confirmed quirk of openai/gpt-oss-20b: it sometimes stringifies an
    # empty second_choice as the word "null"/"none" instead of a real JSON
    # null. This must be normalized to None before any other validation.
    monkeypatch.setattr(classify_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(classify_module, "_get_client", lambda: object())

    text = f"my flight was delayed ({stringified_null})"

    def fake_call(client, model, prompt_text):
        return _fake_response(
            {
                "intent": "FLIGHT_DISRUPTION",
                "second_choice": stringified_null,
                "reasoning": "flight was delayed",
            }
        )

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call)

    result = classify_module.classify_intent(text, model="test-model")

    assert result["second_choice"] is None

    cached = classify_module._read_cache(
        "test-model",
        classify_module._load_prompt_template().replace("{{MESSAGE_TEXT}}", text),
    )
    assert cached["second_choice"] is None


def test_classify_intent_caches_and_avoids_second_api_call(monkeypatch, tmp_path):
    monkeypatch.setattr(classify_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(classify_module, "_get_client", lambda: object())

    call_count = {"n": 0}

    def fake_call(client, model, prompt_text):
        call_count["n"] += 1
        return _fake_response(
            {"intent": "PRAISE_FEEDBACK", "second_choice": None, "reasoning": "thanks"}
        )

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call)

    first = classify_module.classify_intent("thank you so much", model="test-model")
    second = classify_module.classify_intent("thank you so much", model="test-model")

    assert first == second
    assert call_count["n"] == 1


# --- rate-limit retry logic (Groq's actual exception/header shape; no real sleep()) ---


def test_generate_with_retry_parses_retry_delay_from_retry_after_header(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(classify_module.time, "sleep", lambda s: sleep_calls.append(s))

    attempts = {"n": 0}

    def fake_call_generate_api(client, model, prompt_text):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise FakeRateLimitError({"retry-after": "55"})
        return "ok"

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call_generate_api)

    result = classify_module._generate_with_retry(object(), "model", "prompt")

    assert result == "ok"
    assert sleep_calls == [55.0]


def test_generate_with_retry_prefers_retry_after_ms_header(monkeypatch):
    # groq's own SDK prefers the non-standard "retry-after-ms" header
    # (more precise) over "retry-after" in seconds when both are present.
    sleep_calls = []
    monkeypatch.setattr(classify_module.time, "sleep", lambda s: sleep_calls.append(s))

    attempts = {"n": 0}

    def fake_call_generate_api(client, model, prompt_text):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise FakeRateLimitError({"retry-after-ms": "2500", "retry-after": "55"})
        return "ok"

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call_generate_api)

    result = classify_module._generate_with_retry(object(), "model", "prompt")

    assert result == "ok"
    assert sleep_calls == [2.5]


def test_generate_with_retry_falls_back_to_60s_when_unparseable(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(classify_module.time, "sleep", lambda s: sleep_calls.append(s))

    attempts = {"n": 0}

    def fake_call_generate_api(client, model, prompt_text):
        attempts["n"] += 1
        if attempts["n"] == 1:
            # A real RateLimitError, but with no usable retry-after header --
            # this must still fall back to 60s, not skip retrying entirely.
            raise FakeRateLimitError({})
        return "ok"

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call_generate_api)

    result = classify_module._generate_with_retry(object(), "model", "prompt")

    assert result == "ok"
    assert sleep_calls == [60]


def test_generate_with_retry_gives_up_after_five_attempts(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(classify_module.time, "sleep", lambda s: sleep_calls.append(s))

    def always_fails(client, model, prompt_text):
        raise FakeRateLimitError({"retry-after": "1"})

    monkeypatch.setattr(classify_module, "_call_generate_api", always_fails)

    with pytest.raises(RuntimeError):
        classify_module._generate_with_retry(object(), "model", "prompt")

    # max_retries=5 -> 5 attempts total, 4 waits between them, no sleep after the last.
    assert len(sleep_calls) == 4


def test_generate_with_retry_logs_quota_metric_from_error_body(monkeypatch, capsys):
    monkeypatch.setattr(classify_module.time, "sleep", lambda s: None)

    attempts = {"n": 0}

    def fake_call_generate_api(client, model, prompt_text):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise FakeRateLimitError(
                {"retry-after": "5"},
                body={
                    "error": {
                        "message": (
                            "Rate limit reached for model `openai/gpt-oss-20b` in "
                            "organization `org_x` on requests per day (RPD): "
                            "Limit 1000, Used 1000, Please try again in 5s."
                        )
                    }
                },
            )
        return "ok"

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call_generate_api)

    classify_module._generate_with_retry(object(), "model", "prompt")

    out = capsys.readouterr().out
    assert "metric: RPD, limit: 1000" in out
    assert "waiting 5.0s" in out


def test_generate_with_retry_logs_plain_message_when_no_details_found(monkeypatch, capsys):
    monkeypatch.setattr(classify_module.time, "sleep", lambda s: None)

    attempts = {"n": 0}

    def fake_call_generate_api(client, model, prompt_text):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise FakeRateLimitError({"retry-after": "5"})  # no body, no extra headers
        return "ok"

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call_generate_api)

    classify_module._generate_with_retry(object(), "model", "prompt")

    out = capsys.readouterr().out
    assert "metric:" not in out
    assert "Rate limited, waiting 5.0s before retry 2/5..." in out


def test_generate_with_retry_reraises_non_rate_limit_error_immediately(monkeypatch):
    # Regression test: a real run wasted 4 minutes retrying a 404 "model
    # not found" error as if it were a rate limit. Any exception that
    # isn't RateLimitError must be re-raised on the FIRST occurrence, with
    # no retry attempts and no sleep at all.
    sleep_calls = []
    monkeypatch.setattr(classify_module.time, "sleep", lambda s: sleep_calls.append(s))

    call_count = {"n": 0}

    def fake_call_generate_api(client, model, prompt_text):
        call_count["n"] += 1
        raise RuntimeError("404 model not found")

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call_generate_api)

    with pytest.raises(RuntimeError, match="non-retryable API error"):
        classify_module._generate_with_retry(object(), "model", "prompt")

    assert call_count["n"] == 1
    assert sleep_calls == []


def test_classify_intent_reraises_non_rate_limit_error_immediately(monkeypatch, tmp_path):
    monkeypatch.setattr(classify_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(classify_module, "_get_client", lambda: object())

    sleep_calls = []
    monkeypatch.setattr(classify_module.time, "sleep", lambda s: sleep_calls.append(s))

    call_count = {"n": 0}

    def fake_call_generate_api(client, model, prompt_text):
        call_count["n"] += 1
        raise RuntimeError("404 model not found")

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call_generate_api)

    with pytest.raises(RuntimeError, match="non-retryable API error"):
        classify_module.classify_intent("some text", model="test-model")

    assert call_count["n"] == 1
    assert sleep_calls == []


def test_classify_intent_return_cache_hit_reports_false_then_true(monkeypatch, tmp_path):
    monkeypatch.setattr(classify_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(classify_module, "_get_client", lambda: object())

    call_count = {"n": 0}

    def fake_call(client, model, prompt_text):
        call_count["n"] += 1
        return _fake_response(
            {"intent": "PRAISE_FEEDBACK", "second_choice": None, "reasoning": "thanks"}
        )

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call)

    result1, was_cache_hit1 = classify_module.classify_intent(
        "thank you so much", model="test-model", return_cache_hit=True
    )
    result2, was_cache_hit2 = classify_module.classify_intent(
        "thank you so much", model="test-model", return_cache_hit=True
    )

    assert was_cache_hit1 is False
    assert was_cache_hit2 is True
    assert result1 == result2
    assert call_count["n"] == 1


def test_classify_intent_default_return_shape_unchanged(monkeypatch, tmp_path):
    # return_cache_hit defaults to False -- classify_intent must still
    # return the bare result dict, not a tuple, for existing callers.
    monkeypatch.setattr(classify_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(classify_module, "_get_client", lambda: object())

    def fake_call(client, model, prompt_text):
        return _fake_response(
            {"intent": "PRAISE_FEEDBACK", "second_choice": None, "reasoning": "thanks"}
        )

    monkeypatch.setattr(classify_module, "_call_generate_api", fake_call)

    result = classify_module.classify_intent("thank you so much", model="test-model")
    assert isinstance(result, dict)
    assert result["intent"] == "PRAISE_FEEDBACK"
