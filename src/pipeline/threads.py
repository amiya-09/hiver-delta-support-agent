"""Reconstruct (customer message, brand reply) pairs from raw tweet rows.

Operates on plain dicts, not pandas objects/DataFrames, so the core
pairing logic is trivial to unit test with small hand-built fixtures --
no need to spin up a DataFrame to exercise a single branch.
"""
from dataclasses import dataclass, asdict

from src.pipeline.clean import (
    strip_mentions_and_links,
    normalize_for_dedup,
    is_dm_redirect,
    is_too_short,
)


@dataclass
class PipelineStats:
    """Counters for every drop reason and outcome during pair reconstruction.

    Each drop reason gets its own counter rather than a single generic
    "dropped" total -- this is the evidence trail for the project's
    decision log (e.g. "45% of replies are DM redirects"), not just debug
    output we throw away.
    """

    total_brand_replies_seen: int = 0
    dropped_no_parent_id: int = 0
    dropped_broken_chain: int = 0
    dropped_parent_not_customer: int = 0
    dropped_too_short: int = 0
    dropped_duplicate: int = 0
    dm_redirect_count: int = 0
    thread_root_count: int = 0
    thread_nonroot_count: int = 0
    kept_pairs: int = 0

    def as_dict(self):
        return asdict(self)


def build_row_index(rows):
    """Build a tweet_id -> row lookup for O(1) parent lookups."""
    return {row["tweet_id"]: row for row in rows}


def reconstruct_pairs(rows, brand, row_index=None):
    """Reconstruct customer<->brand reply pairs for a single brand.

    Walks every row authored by `brand` that is an outbound reply
    (inbound is False), resolves its parent via in_response_to_tweet_id,
    and keeps the pair only if the parent is a genuine customer message
    and both sides survive cleaning/dedup. Returns (pairs, stats).
    """
    if row_index is None:
        row_index = build_row_index(rows)

    stats = PipelineStats()
    seen_by_customer = {}
    pairs = []

    for row in rows:
        if row["author_id"] != brand or row["inbound"] is not False:
            continue

        stats.total_brand_replies_seen += 1

        parent_id = row.get("in_response_to_tweet_id", "")
        if not parent_id:
            stats.dropped_no_parent_id += 1
            continue

        parent = row_index.get(parent_id)
        if parent is None:
            stats.dropped_broken_chain += 1
            continue

        if parent["inbound"] is not True:
            stats.dropped_parent_not_customer += 1
            continue

        customer_text_clean = strip_mentions_and_links(parent["text"])
        brand_text_clean = strip_mentions_and_links(row["text"])
        if is_too_short(customer_text_clean) or is_too_short(brand_text_clean):
            stats.dropped_too_short += 1
            continue

        customer_author_id = parent["author_id"]
        normalized = normalize_for_dedup(parent["text"])
        seen_for_customer = seen_by_customer.setdefault(customer_author_id, set())
        if normalized in seen_for_customer:
            stats.dropped_duplicate += 1
            continue
        seen_for_customer.add(normalized)

        dm_redirect = is_dm_redirect(row["text"])
        if dm_redirect:
            stats.dm_redirect_count += 1

        is_thread_root = not parent.get("in_response_to_tweet_id", "")
        if is_thread_root:
            stats.thread_root_count += 1
        else:
            stats.thread_nonroot_count += 1

        pairs.append(
            {
                "customer_tweet_id": parent["tweet_id"],
                "customer_text_raw": parent["text"],
                "customer_text_clean": customer_text_clean,
                "brand_reply_tweet_id": row["tweet_id"],
                "brand_reply_text_raw": row["text"],
                "brand_reply_text_clean": brand_text_clean,
                "created_at": row["created_at"],
                "is_thread_root": is_thread_root,
                "is_dm_redirect": dm_redirect,
            }
        )
        stats.kept_pairs += 1

    return pairs, stats
