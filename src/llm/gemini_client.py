"""Thin wrapper around the Gemini embeddings API with disk caching and retries.

Uses the new unified `google.genai` SDK (not the deprecated
`google.generativeai` package). Embeddings are cached to disk keyed by a
hash of (model, text) so re-running the pipeline over the same texts never
re-pays for API calls we've already made.
"""
import hashlib
import json
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

CACHE_DIR = Path(".cache/llm")


def _get_client():
    """Build a genai Client from GOOGLE_API_KEY, with a clear error if unset.

    We check the env var ourselves rather than letting the SDK raise its
    own auth exception -- a raw SDK error here is confusing to a first-time
    user; a RuntimeError telling them exactly which env var to set is not.
    """
    import os

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Set it to your Gemini API key before calling embed_texts(), "
            "e.g. `export GOOGLE_API_KEY=...`."
        )
    return genai.Client(api_key=api_key)


def _cache_path(model, text, task_type=None):
    # CRITICAL: task_type must be part of the key -- the same text embedded
    # as a RETRIEVAL_DOCUMENT vs. a RETRIEVAL_QUERY produces different
    # vectors, so reusing one for the other would silently corrupt results.
    key = hashlib.sha256(f"{model}:{task_type}:{text}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _read_cache(model, text, task_type=None):
    path = _cache_path(model, text, task_type)
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    return data["embedding"]


def _write_cache(model, text, embedding, task_type=None):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(model, text, task_type)
    with path.open("w") as f:
        json.dump(
            {"model": model, "text": text, "task_type": task_type, "embedding": embedding},
            f,
        )


def _call_embed_api(client, model, batch, task_type=None):
    """The actual network call, isolated so tests can monkeypatch just this.

    Kept as a thin, single-purpose function (not inlined into
    _embed_batch_with_retry) so unit tests can fake the network boundary
    without needing to fake retry/backoff behavior too.
    """
    kwargs = {"model": model, "contents": batch}
    if task_type is not None:
        kwargs["config"] = types.EmbedContentConfig(task_type=task_type)
    response = client.models.embed_content(**kwargs)
    return [item.values for item in response.embeddings]


_RETRY_DELAY_RE = re.compile(r"([\d.]+)\s*s")
_FALLBACK_RETRY_DELAY = 60


def _parse_retry_delay_seconds(exc):
    """Best-effort extraction of Google's suggested retry delay from a 429.

    google.genai's ClientError carries the parsed error body on `.details`,
    shaped like {"error": {..., "details": [{"@type": ".../RetryInfo",
    "retryDelay": "55s"}, ...]}} (or without the "error" wrapper, depending
    on the response). We dig for a RetryInfo entry and parse the numeric
    seconds out of its "retryDelay" string. Any mismatch in shape (a
    different exception type, missing fields, a non-numeric delay) just
    means we can't find a delay -- returning None so the caller falls back
    to a fixed wait, never crashing the retry loop itself.
    """
    try:
        details = getattr(exc, "details", None)
        if not isinstance(details, dict):
            return None
        error_body = details.get("error", details)
        if not isinstance(error_body, dict):
            return None
        for entry in error_body.get("details", []) or []:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("@type", "")).endswith("RetryInfo"):
                match = _RETRY_DELAY_RE.match(str(entry.get("retryDelay", "")))
                if match:
                    return float(match.group(1))
    except Exception:
        return None
    return None


def _embed_batch_with_retry(client, model, batch, task_type=None, max_retries=5):
    """Call _call_embed_api, retrying on failure with Google's suggested delay.

    Free-tier Gemini rate limits (429s) come back with a RetryInfo telling
    us exactly how long to wait -- we honor that instead of guessing with a
    fixed backoff schedule. This is an offline batch job with no latency
    requirement, so a patient retry budget (5 attempts, waits up to a
    minute+ each) is the right trade-off over failing fast.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            embeddings = _call_embed_api(client, model, batch, task_type=task_type)
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Embedding API returned {len(embeddings)} embeddings "
                    f"for a batch of {len(batch)} texts."
                )
            return embeddings
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = _parse_retry_delay_seconds(exc)
                if delay is None:
                    delay = _FALLBACK_RETRY_DELAY
                print(
                    f"Rate limited, waiting {delay}s before retry "
                    f"{attempt + 2}/{max_retries}..."
                )
                time.sleep(delay)
    raise RuntimeError(
        f"Failed to embed batch after {max_retries} attempts: {last_exc}"
    ) from last_exc


def embed_texts(texts, model="gemini-embedding-001", batch_size=50, task_type=None, pacing_delay=3):
    """Embed a list of texts, using the disk cache and batching API calls.

    Cache is checked first for every text; only cache misses are sent to
    the API, in batches of `batch_size`, with retries. Each freshly-fetched
    embedding is written to the cache immediately after its batch succeeds
    -- so a crash partway through a large run doesn't lose earlier batches.
    Results are returned in the same order as `texts`, regardless of
    whether each one came from cache or a fresh call.

    `task_type` (e.g. "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY") changes the
    embedding the API returns for the same text, so it's part of the cache
    key -- see _cache_path.

    After each successful batch we sleep `pacing_delay` seconds before the
    next one -- a small fixed cost, but it reduces the chance of tripping
    the free-tier per-minute rate limit in the first place, rather than
    only reacting to it after the fact via retries.
    """
    results = [None] * len(texts)
    miss_indices = []
    cache_hits = 0

    for i, text in enumerate(texts):
        cached = _read_cache(model, text, task_type)
        if cached is not None:
            results[i] = cached
            cache_hits += 1
        else:
            miss_indices.append(i)

    if miss_indices:
        client = _get_client()
        total_to_embed = len(miss_indices)
        total_batches = (total_to_embed + batch_size - 1) // batch_size

        for batch_num, start in enumerate(range(0, total_to_embed, batch_size), start=1):
            batch_indices = miss_indices[start : start + batch_size]
            batch_texts = [texts[i] for i in batch_indices]
            embeddings = _embed_batch_with_retry(client, model, batch_texts, task_type=task_type)
            for idx, text, embedding in zip(batch_indices, batch_texts, embeddings):
                results[idx] = embedding
                _write_cache(model, text, embedding, task_type)

            texts_done = start + len(batch_indices)
            print(
                f"Embedded batch {batch_num}/{total_batches} "
                f"({texts_done}/{total_to_embed} texts)..."
            )
            if batch_num < total_batches:
                time.sleep(pacing_delay)

    api_calls = len(miss_indices)
    print(f"embed_texts: {cache_hits} cache hits, {api_calls} fresh API calls")
    return results
