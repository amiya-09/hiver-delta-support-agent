"""Shared Groq rate-limit retry logic, used by every agent that talks to
Groq's chat API (classification, generation, ...). Factored out of
src/agent/classify.py so generate.py doesn't have to reimplement
proven, tested behavior.

Only groq.RateLimitError is treated as retryable -- a 404 "model not
found" or a bad-request error will never succeed just by waiting, so any
other exception is re-raised immediately instead of burning through the
retry budget on misleading "rate limited" messages.
"""
import re
import time

from groq import RateLimitError

FALLBACK_RETRY_DELAY = 60

_RATE_LIMIT_MESSAGE_RE = re.compile(
    r"on\s+(?P<metric_name>[\w /]+?)\s*\((?P<metric_short>[A-Za-z]+)\):\s*Limit\s*(?P<limit>[\d,]+)",
    re.IGNORECASE,
)


def parse_retry_delay_seconds(exc):
    """Best-effort extraction of Groq's suggested retry delay from a 429.

    groq.RateLimitError (a subclass of groq.APIStatusError) carries the
    raw httpx.Response on `.response`. Inspecting groq's own internal
    retry logic (_base_client.py's _parse_retry_after_header) confirmed
    the real header names to check: the non-standard "retry-after-ms"
    (milliseconds, more precise) first, falling back to the standard
    "retry-after" (seconds). Any mismatch in shape (a different exception
    type, no response, missing/non-numeric headers) just means we can't
    find a delay -- returning None so the caller falls back to a fixed
    wait, never crashing the retry loop itself.
    """
    try:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return None

        retry_after_ms = headers.get("retry-after-ms")
        if retry_after_ms is not None:
            return float(retry_after_ms) / 1000

        retry_after = headers.get("retry-after")
        if retry_after is not None:
            return float(retry_after)
    except Exception:
        return None
    return None


def extract_rate_limit_details(exc):
    """Best-effort extraction of the quota metric name/limit, for LOGGING
    ONLY -- never affects retry/delay behavior, and any failure here just
    means we log the plain message instead of a more detailed one.

    Groq's rate-limit error messages read like "...on requests per day
    (RPD): Limit 1000, Used 1000..." in the parsed error body, so that's
    tried first; falls back to common `x-ratelimit-limit-*` response
    headers if the body doesn't have a usable message. Returns a short
    string like "metric: RPD, limit: 1000", or None if nothing is found.
    """
    try:
        body = getattr(exc, "body", None)
        message = None
        if isinstance(body, dict):
            error = body.get("error")
            message = error.get("message") if isinstance(error, dict) else body.get("message")
        if isinstance(message, str):
            match = _RATE_LIMIT_MESSAGE_RE.search(message)
            if match:
                return f"metric: {match.group('metric_short')}, limit: {match.group('limit')}"

        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            limit_requests = headers.get("x-ratelimit-limit-requests")
            if limit_requests is not None:
                return f"metric: requests, limit: {limit_requests}"
            limit_tokens = headers.get("x-ratelimit-limit-tokens")
            if limit_tokens is not None:
                return f"metric: tokens, limit: {limit_tokens}"
    except Exception:
        return None
    return None


def call_with_retry(call_fn, max_retries=5):
    """Call `call_fn()` (a zero-arg callable), retrying ONLY on
    groq.RateLimitError with Groq's suggested delay (falling back to a
    fixed 60s if it can't be parsed), up to max_retries attempts. Any
    other exception is re-raised immediately as a non-retryable
    RuntimeError -- see module docstring.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return call_fn()
        except RateLimitError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = parse_retry_delay_seconds(exc)
                if delay is None:
                    delay = FALLBACK_RETRY_DELAY
                details = extract_rate_limit_details(exc)
                if details:
                    print(
                        f"Rate limited ({details}), waiting {delay}s before "
                        f"retry {attempt + 2}/{max_retries}..."
                    )
                else:
                    print(
                        f"Rate limited, waiting {delay}s before retry "
                        f"{attempt + 2}/{max_retries}..."
                    )
                time.sleep(delay)
        except Exception as exc:
            raise RuntimeError(f"non-retryable API error: {exc}") from exc
    raise RuntimeError(f"Failed after {max_retries} attempts: {last_exc}") from last_exc
