from src.pipeline.clean import (
    strip_mentions_and_links,
    normalize_for_dedup,
    is_dm_redirect,
    is_too_short,
)
from src.pipeline.threads import reconstruct_pairs


def make_row(
    tweet_id,
    author_id,
    inbound,
    text,
    in_response_to_tweet_id="",
    created_at="2020-01-01 00:00:00",
):
    return {
        "tweet_id": tweet_id,
        "author_id": author_id,
        "inbound": inbound,
        "created_at": created_at,
        "text": text,
        "in_response_to_tweet_id": in_response_to_tweet_id,
    }


# --- strip_mentions_and_links ---


def test_strip_mentions_and_links_removes_mentions_and_urls():
    text = "@Delta my flight https://t.co/xyz123 got cancelled again"
    assert strip_mentions_and_links(text) == "my flight got cancelled again"


def test_strip_mentions_and_links_keeps_real_content():
    text = "my bag never arrived at the carousel"
    assert strip_mentions_and_links(text) == text


def test_strip_mentions_and_links_handles_none():
    assert strip_mentions_and_links(None) == ""


# --- is_dm_redirect ---


def test_is_dm_redirect_true_for_dm_phrase():
    assert is_dm_redirect("Please DM us your confirmation number.") is True


def test_is_dm_redirect_true_for_direct_message_phrase():
    assert is_dm_redirect("Send us a direct message so we can help.") is True


def test_is_dm_redirect_false_for_normal_sentence():
    assert is_dm_redirect("We've refunded your ticket, sorry for the delay.") is False


def test_is_dm_redirect_false_for_empty_and_none():
    assert is_dm_redirect("") is False
    assert is_dm_redirect(None) is False


# --- is_too_short ---


def test_is_too_short_empty():
    assert is_too_short("") is True


def test_is_too_short_one_word():
    assert is_too_short("Thanks") is True


def test_is_too_short_four_words():
    assert is_too_short("my flight was delayed") is False


# --- normalize_for_dedup ---


def test_normalize_for_dedup_matches_near_identical_strings():
    a = "My flight DL123 was Cancelled!!"
    b = "my flight dl123 was cancelled"
    assert normalize_for_dedup(a) == normalize_for_dedup(b)


# --- reconstruct_pairs ---


def test_single_turn_pair_is_kept_and_is_thread_root():
    rows = [
        make_row("1", "cust1", True, "@Delta my flight was cancelled with no notice"),
        make_row("2", "Delta", False, "So sorry, we've rebooked you on the next flight.", in_response_to_tweet_id="1"),
    ]
    pairs, stats = reconstruct_pairs(rows, "Delta")
    assert stats.kept_pairs == 1
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["customer_tweet_id"] == "1"
    assert pair["brand_reply_tweet_id"] == "2"
    assert pair["is_thread_root"] is True


def test_multi_turn_thread_produces_two_pairs_second_not_root():
    rows = [
        make_row("1", "cust1", True, "@Delta my flight was cancelled with no notice"),
        make_row("2", "Delta", False, "So sorry, we've rebooked you on the next flight.", in_response_to_tweet_id="1"),
        make_row("3", "cust1", True, "@Delta thanks but the new flight is also delayed", in_response_to_tweet_id="2"),
        make_row("4", "Delta", False, "We understand, here is a travel voucher for the trouble.", in_response_to_tweet_id="3"),
    ]
    pairs, stats = reconstruct_pairs(rows, "Delta")
    assert stats.kept_pairs == 2
    assert len(pairs) == 2
    assert pairs[0]["is_thread_root"] is True
    assert pairs[1]["is_thread_root"] is False


def test_dm_redirect_reply_is_kept_but_flagged():
    rows = [
        make_row("1", "cust1", True, "@Delta my bag never arrived at the carousel"),
        make_row("2", "Delta", False, "Please DM us your bag claim number.", in_response_to_tweet_id="1"),
    ]
    pairs, stats = reconstruct_pairs(rows, "Delta")
    assert stats.kept_pairs == 1
    assert stats.dm_redirect_count == 1
    assert pairs[0]["is_dm_redirect"] is True


def test_broken_chain_is_dropped_and_counted():
    rows = [
        make_row("2", "Delta", False, "So sorry, we've rebooked you on the next flight.", in_response_to_tweet_id="does-not-exist"),
    ]
    pairs, stats = reconstruct_pairs(rows, "Delta")
    assert len(pairs) == 0
    assert stats.dropped_broken_chain == 1
    assert stats.total_brand_replies_seen == 1


def test_parent_not_customer_is_dropped_and_counted():
    rows = [
        make_row("1", "Delta", False, "We're aware of the storm delays across the network."),
        make_row("2", "Delta", False, "Following up on our earlier network-wide update.", in_response_to_tweet_id="1"),
    ]
    pairs, stats = reconstruct_pairs(rows, "Delta")
    assert len(pairs) == 0
    assert stats.dropped_parent_not_customer == 1


def test_too_short_after_cleaning_is_dropped_and_counted():
    rows = [
        make_row("1", "cust1", True, "@Delta 😡"),
        make_row("2", "Delta", False, "So sorry, we've rebooked you on the next flight.", in_response_to_tweet_id="1"),
    ]
    pairs, stats = reconstruct_pairs(rows, "Delta")
    assert len(pairs) == 0
    assert stats.dropped_too_short == 1


def test_near_duplicate_from_same_customer_is_deduped():
    rows = [
        make_row("1", "cust1", True, "@Delta my flight was cancelled with no notice"),
        make_row("2", "Delta", False, "So sorry, we've rebooked you on the next flight.", in_response_to_tweet_id="1"),
        make_row("3", "cust1", True, "@Delta My Flight Was Cancelled With No Notice!!"),
        make_row("4", "Delta", False, "We understand, here is a travel voucher for the trouble.", in_response_to_tweet_id="3"),
    ]
    pairs, stats = reconstruct_pairs(rows, "Delta")
    assert stats.kept_pairs == 1
    assert stats.dropped_duplicate == 1


def test_reply_from_different_brand_is_ignored_entirely():
    rows = [
        make_row("1", "cust1", True, "@United my flight was cancelled with no notice"),
        make_row("2", "United", False, "So sorry, we've rebooked you on the next flight.", in_response_to_tweet_id="1"),
    ]
    pairs, stats = reconstruct_pairs(rows, "Delta")
    assert len(pairs) == 0
    assert stats.total_brand_replies_seen == 0


def test_empty_in_response_to_tweet_id_is_dropped_and_counted():
    rows = [
        make_row("2", "Delta", False, "So sorry for the trouble, let us know how we can help."),
    ]
    pairs, stats = reconstruct_pairs(rows, "Delta")
    assert len(pairs) == 0
    assert stats.dropped_no_parent_id == 1
    assert stats.total_brand_replies_seen == 1
