"""Intent classification -- the first real agent behavior, evaluated
directly against the golden set (see src/eval/run_classification_eval.py).

Runs on Groq (openai/gpt-oss-20b) rather than Gemini: the project hit
an unusually low daily quota (20 requests/day) on gemini-2.5-flash
specifically, project-level and separate from the embedding model. Groq's
free tier (30 RPM / 1,000 RPD) is far more workable for a ~240-example
eval loop. Embeddings stay on local sentence-transformers, untouched.
"""
import hashlib
import json
import time
from pathlib import Path

from groq import Groq

from src.golden_set.constants import INTENT_CATEGORIES
from src.llm.groq_retry import call_with_retry

CACHE_DIR = Path(".cache/classify")
PROMPT_TEMPLATE_PATH = Path("prompts/classify_intent.txt")

_STRICT_JSON_SUFFIX = "\n\nRespond with ONLY the JSON object, no other text."


def _get_client():
    """Build a Groq client from GROQ_API_KEY, with a clear error if unset.

    Same reasoning as gemini_client.py: a RuntimeError naming the exact env
    var is far more actionable than a raw SDK auth exception.
    """
    import os

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Set it to your Groq API key before calling classify_intent(), "
            "e.g. `export GROQ_API_KEY=...`."
        )
    return Groq(api_key=api_key)


def _load_prompt_template():
    return PROMPT_TEMPLATE_PATH.read_text()


def _cache_path(model, prompt_text):
    key = hashlib.sha256(f"{model}:{prompt_text}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _read_cache(model, prompt_text):
    path = _cache_path(model, prompt_text)
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    return data["result"]


def _write_cache(model, prompt_text, result):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(model, prompt_text)
    with path.open("w") as f:
        json.dump({"model": model, "prompt_text": prompt_text, "result": result}, f)


def _call_generate_api(client, model, prompt_text):
    """The actual network call, isolated so tests can monkeypatch just this
    without needing to fake JSON-parsing/retry behavior too."""
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt_text}]
    )
    return response.choices[0].message.content


def _generate_with_retry(client, model, prompt_text, max_retries=5):
    """Call _call_generate_api with Groq rate-limit retry logic.

    Thin wrapper around the shared src.llm.groq_retry.call_with_retry --
    see that module for the actual retry/delay-parsing/logging behavior.
    Kept as a same-named, same-signature local function so existing tests
    (and any other caller) don't need to change.
    """
    return call_with_retry(
        lambda: _call_generate_api(client, model, prompt_text), max_retries=max_retries
    )


def _parse_response(raw_text):
    """Parse the model's response text as the expected JSON object.

    Models sometimes wrap JSON in a markdown code fence despite being told
    not to -- strip that before parsing so an otherwise-correct response
    isn't treated as malformed.
    """
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def classify_intent(text, model="openai/gpt-oss-20b", return_cache_hit=False):
    """Classify a customer message into one of the 8 intent categories.

    Cached on disk by hash of (model, prompt_text), same pattern as
    gemini_client.py, so re-running an eval over the same examples is free
    on repeat. On a malformed JSON response, retries exactly once with an
    added "respond with ONLY JSON" instruction; a second failure raises
    loudly rather than silently returning a default. An intent outside the
    8 valid categories also raises loudly -- that should never happen and
    indicates a prompt problem worth surfacing, not papering over.

    If `return_cache_hit` is True, returns (result, was_cache_hit) instead
    of just result -- was_cache_hit is True only when the disk cache
    answered the call with no API call made at all (a fresh generation
    counts as a miss regardless of whether it needed the malformed-JSON
    retry). Callers that don't pass this flag see unchanged behavior.
    """
    template = _load_prompt_template()
    prompt_text = template.replace("{{MESSAGE_TEXT}}", text)

    cached = _read_cache(model, prompt_text)
    if cached is not None:
        return (cached, True) if return_cache_hit else cached

    client = _get_client()

    raw_text = _generate_with_retry(client, model, prompt_text)
    try:
        parsed = _parse_response(raw_text)
    except (json.JSONDecodeError, ValueError):
        retry_prompt_text = prompt_text + _STRICT_JSON_SUFFIX
        raw_text_retry = _generate_with_retry(client, model, retry_prompt_text)
        try:
            parsed = _parse_response(raw_text_retry)
        except (json.JSONDecodeError, ValueError):
            raise RuntimeError(
                "classify_intent: model response was not valid JSON, even "
                f"after retrying with an explicit JSON-only instruction. "
                f"Last raw response: {raw_text_retry!r}"
            )

    # openai/gpt-oss-20b has a confirmed quirk: it sometimes stringifies an
    # empty second_choice as the four-character string "null" (or "none")
    # instead of emitting a real JSON null. Normalize that BEFORE any
    # validation/comparison below, so it's treated exactly like a genuine
    # None from this point on -- including what gets cached.
    second_choice_raw = parsed.get("second_choice")
    if isinstance(second_choice_raw, str) and second_choice_raw.strip().lower() in ("null", "none"):
        parsed["second_choice"] = None

    intent = parsed.get("intent")
    if intent not in INTENT_CATEGORIES:
        raise RuntimeError(
            f"classify_intent: model returned an invalid intent {intent!r}, "
            f"which is not one of the 8 valid categories: {INTENT_CATEGORIES}."
        )

    second_choice = parsed.get("second_choice")
    if second_choice is not None and second_choice not in INTENT_CATEGORIES:
        raise RuntimeError(
            f"classify_intent: model returned an invalid second_choice "
            f"{second_choice!r}, which is not one of the 8 valid categories "
            f"(or null): {INTENT_CATEGORIES}. The escalation phase's "
            f"ambiguous_classification rule reads this field directly, so "
            f"it can't be allowed through with bad data."
        )
    if second_choice == intent:
        # The model redundantly repeated the primary intent instead of
        # returning null for "not genuinely close" -- this isn't a real
        # second choice, and left as-is would look like a false "close
        # call" signal to the ambiguous_classification escalation rule.
        parsed["second_choice"] = None

    _write_cache(model, prompt_text, parsed)
    return (parsed, False) if return_cache_hit else parsed
