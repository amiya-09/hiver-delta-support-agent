"""Pure text-cleaning helpers for the Delta support-tweet pipeline.

No I/O, no pandas -- everything here operates on plain strings so it's
trivial to unit test and reason about in isolation from the messier
row-reconstruction logic in threads.py.
"""
import re

_MENTION_RE = re.compile(r"@\w+")
_URL_RE = re.compile(r"https?://\S+")
_WHITESPACE_RE = re.compile(r"\s+")
_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_DM_WORD_RE = re.compile(r"\bdm\b", re.IGNORECASE)
_DIRECT_MESSAGE_RE = re.compile(r"direct message", re.IGNORECASE)


def strip_mentions_and_links(text):
    """Remove @mentions and URLs, collapse whitespace.

    This produces the "clean" text used for matching, embedding, and
    classification. We deliberately keep the raw text around separately
    (for display/generation) because tokens like "@Delta" are meaningless
    noise for similarity comparisons -- two totally different complaints
    that both mention "@Delta" would otherwise look artificially similar.
    Returns "" for None input so callers never have to null-check.
    """
    if text is None:
        return ""
    no_mentions = _MENTION_RE.sub("", text)
    no_urls = _URL_RE.sub("", no_mentions)
    return _WHITESPACE_RE.sub(" ", no_urls).strip()


def normalize_for_dedup(text):
    """Lowercase + strip to alphanumeric-only, after removing mentions/links.

    This is deliberately a dumb, deterministic exact-match normalization
    (not fuzzy/near-duplicate matching): it only catches cases where two
    strings are the same content modulo case and punctuation. That's a
    conscious scope decision -- real fuzzy near-dup detection is a bigger
    problem we're not solving in Phase 1.
    """
    stripped = strip_mentions_and_links(text)
    return _ALNUM_RE.sub("", stripped.lower())


def is_dm_redirect(raw_text):
    """True if the reply is a DM/direct-message handoff, not a resolution.

    Checked on RAW text (not the mention-stripped version) because the
    word "DM" or phrase "direct message" isn't affected by @mention noise
    -- there's no benefit to cleaning first, and checking raw keeps this
    signal independent of the cleaning step. We flag these rather than
    drop them: the customer's complaint is still useful for intent
    classification and the eval set, but the reply itself isn't a real
    "here's how we solved it" example for retrieval/generation.
    """
    if not raw_text:
        return False
    return bool(_DM_WORD_RE.search(raw_text) or _DIRECT_MESSAGE_RE.search(raw_text))


def is_too_short(clean_text, min_words=3):
    """True if fewer than min_words remain after cleaning.

    A tweet like "@Delta 😡" becomes empty or near-empty once mentions are
    stripped -- there's no real signal left to embed, classify, or show as
    a meaningful complaint, so we flag it for dropping.
    """
    if not clean_text:
        return True
    return len(clean_text.split()) < min_words
