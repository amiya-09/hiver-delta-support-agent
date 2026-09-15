"""LLM-as-judge -- scores a generated reply on 4 rubric dimensions.

Reuses the exact same infrastructure as classify.py/generate.py: disk
caching by hash of (model, prompt_text), the shared Groq rate-limit retry
logic (src/llm/groq_retry.py), and the malformed-JSON-retry-once pattern.
"""
import hashlib
import json
from pathlib import Path

from groq import Groq

from src.llm.groq_retry import call_with_retry

CACHE_DIR = Path(".cache/judge")
PROMPT_TEMPLATE_PATH = Path("prompts/judge_reply.txt")

_STRICT_JSON_SUFFIX = "\n\nRespond with ONLY the JSON object, no other text."

RUBRIC_FIELDS = ["grounded", "factually_consistent", "tone_appropriate", "actionable"]


def _get_client():
    """Build a Groq client from GROQ_API_KEY, with a clear error if unset.

    Same reasoning as classify.py/generate.py: a RuntimeError naming the
    exact env var is far more actionable than a raw SDK auth exception.
    """
    import os

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Set it to your Groq API key before calling judge_reply(), "
            "e.g. `export GROQ_API_KEY=...`."
        )
    return Groq(api_key=api_key)


def _load_prompt_template():
    return PROMPT_TEMPLATE_PATH.read_text()


def _format_retrieved_examples(retrieved_examples):
    """Render retrieved_examples into the numbered block the prompt expects.

    Same formatting style as generate.py's _format_retrieved_examples, so
    the judge sees precedent in the same shape the generator saw it in.
    """
    if not retrieved_examples:
        return "(No retrieved examples -- none were found or provided.)"

    lines = []
    for i, example in enumerate(retrieved_examples):
        similarity = example.get("similarity_score")
        similarity_str = f"{similarity:.2f}" if isinstance(similarity, (int, float)) else "unknown"
        customer_text = example.get("customer_text_clean", "")
        brand_text = example.get("brand_reply_text_clean", "")
        lines.append(
            f"[Example {i}, similarity={similarity_str}] "
            f"Customer said: '{customer_text}' | Delta replied: '{brand_text}'"
        )
    return "\n".join(lines)


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
    same pattern as classify.py/generate.py's _generate_with_retry.
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


def judge_reply(customer_text, generated_reply, retrieved_examples, model="openai/gpt-oss-20b"):
    """Score a generated reply on 4 rubric dimensions (1-3 each).

    Cached on disk by hash of (model, prompt_text), same pattern as
    classify.py/generate.py, so re-running an eval over the same examples
    is free on repeat. On a malformed JSON response, retries exactly once
    with an added "respond with ONLY JSON" instruction; a second failure
    raises loudly. Every rubric field (grounded, factually_consistent,
    tone_appropriate, actionable) must be present and an integer in
    {1, 2, 3} -- a missing field or out-of-range value raises immediately,
    same fail-loud standard applied to every other validated field in
    this project (intent, reply, grounding_source_indices).
    """
    template = _load_prompt_template()
    prompt_text = (
        template.replace("{{CUSTOMER_MESSAGE}}", customer_text)
        .replace("{{GENERATED_REPLY}}", generated_reply)
        .replace("{{RETRIEVED_EXAMPLES}}", _format_retrieved_examples(retrieved_examples))
    )

    cached = _read_cache(model, prompt_text)
    if cached is not None:
        return cached

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
                "judge_reply: model response was not valid JSON, even "
                f"after retrying with an explicit JSON-only instruction. "
                f"Last raw response: {raw_text_retry!r}"
            )

    for field in RUBRIC_FIELDS:
        value = parsed.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value not in (1, 2, 3):
            raise RuntimeError(
                f"judge_reply: rubric field {field!r} is missing or invalid "
                f"({value!r}) -- must be an integer in {{1, 2, 3}}."
            )

    _write_cache(model, prompt_text, parsed)
    return parsed
