"""Reply generation -- the second real agent behavior: draft a reply
grounded in retrieved historical precedent, with an auditable grounding
trail (see grounding_source_indices).

Reuses the exact infrastructure proven in src/agent/classify.py: disk
caching by hash of (model, prompt_text), the shared Groq rate-limit retry
logic (src/llm/groq_retry.py), and the malformed-JSON-retry-once pattern.
"""
import hashlib
import json
from pathlib import Path

from groq import Groq

from src.llm.groq_retry import call_with_retry

CACHE_DIR = Path(".cache/generate")
PROMPT_TEMPLATE_PATH = Path("prompts/generate_reply.txt")

_STRICT_JSON_SUFFIX = "\n\nRespond with ONLY the JSON object, no other text."


def _get_client():
    """Build a Groq client from GROQ_API_KEY, with a clear error if unset.

    Same reasoning as classify.py/gemini_client.py: a RuntimeError naming
    the exact env var is far more actionable than a raw SDK auth exception.
    """
    import os

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Set it to your Groq API key before calling generate_reply(), "
            "e.g. `export GROQ_API_KEY=...`."
        )
    return Groq(api_key=api_key)


def _load_prompt_template():
    return PROMPT_TEMPLATE_PATH.read_text()


def _format_retrieved_examples(retrieved_examples):
    """Render retrieved_examples into the numbered block the prompt expects.

    Indices here are exactly what grounding_source_indices must reference
    back into, so the numbering must match retrieved_examples' order.
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
    same pattern as classify.py's _generate_with_retry.
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


def generate_reply(text, intent, retrieved_examples, model="openai/gpt-oss-20b"):
    """Draft a reply grounded in retrieved_examples, with an auditable
    grounding trail.

    retrieved_examples is a list of dicts, each with at least
    customer_text_clean, brand_reply_text_clean, and similarity_score --
    exactly what src/retrieval/query.py's top_k_similar returns, so its
    output can be passed straight through.

    Cached on disk by hash of (model, prompt_text), same pattern as
    classify.py, so re-running an eval over the same examples is free on
    repeat. On a malformed JSON response, retries exactly once with an
    added "respond with ONLY JSON" instruction; a second failure raises
    loudly. Every index in grounding_source_indices must be a valid
    position in retrieved_examples (0 to len-1) -- an out-of-range index
    raises immediately, since this field is the actual auditability
    mechanism for "grounded in historical resolution" and can't be
    allowed through untrustworthy.
    """
    template = _load_prompt_template()
    prompt_text = (
        template.replace("{{MESSAGE_TEXT}}", text)
        .replace("{{INTENT}}", intent)
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
                "generate_reply: model response was not valid JSON, even "
                f"after retrying with an explicit JSON-only instruction. "
                f"Last raw response: {raw_text_retry!r}"
            )

    reply = parsed.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError("generate_reply: model returned an empty or missing reply field")

    grounding_source_indices = parsed.get("grounding_source_indices") or []
    valid_range = range(len(retrieved_examples))
    for idx in grounding_source_indices:
        if not isinstance(idx, int) or idx not in valid_range:
            raise RuntimeError(
                f"generate_reply: model returned an out-of-range "
                f"grounding_source_indices value {idx!r}; valid indices for "
                f"the {len(retrieved_examples)} retrieved example(s) passed "
                f"in are 0 to {len(retrieved_examples) - 1}. This field is "
                f"the auditability mechanism for grounding claims and can't "
                f"be trusted if it references examples that don't exist."
            )

    _write_cache(model, prompt_text, parsed)
    return parsed
