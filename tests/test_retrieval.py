import numpy as np
import pytest

import src.llm.gemini_client as gemini_client
from src.retrieval.query import top_k_similar
from src.retrieval.build_index import filter_pool, sample_pool


class FakeClientError(Exception):
    """Stands in for google.genai.errors.ClientError, which carries the
    parsed error body (including RetryInfo) on a `.details` attribute."""

    def __init__(self, details):
        self.details = details
        super().__init__(str(details))


# --- top_k_similar ranking logic (no real API calls, fixed vectors) ---


def test_top_k_similar_ranks_by_cosine_similarity():
    # Hand-picked so we know the exact expected order:
    #   idx 0: identical to the query -> highest similarity (1.0)
    #   idx 1: also identical to the query -> tied for highest (1.0)
    #   idx 2: orthogonal to the query -> similarity 0.0
    #   idx 3: opposite of the query -> lowest similarity (-1.0)
    query_vec = [1.0, 0.0]
    embeddings = np.array(
        [
            [1.0, 0.0],  # idx 0: identical
            [1.0, 0.0],  # idx 1: identical (tie)
            [0.0, 1.0],  # idx 2: orthogonal
            [-1.0, 0.0],  # idx 3: opposite
        ]
    )
    metadata = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]

    results = top_k_similar(
        "irrelevant since query_embedding is precomputed",
        embeddings,
        metadata,
        k=4,
        query_embedding=query_vec,
    )

    ids_in_order = [r["id"] for r in results]
    assert ids_in_order == ["a", "b", "c", "d"]
    assert results[0]["similarity_score"] == 1.0
    assert results[1]["similarity_score"] == 1.0
    assert results[2]["similarity_score"] == 0.0
    assert results[3]["similarity_score"] == -1.0


def test_top_k_similar_raises_on_nan_scores():
    # Inject a NaN directly into the embeddings matrix so the dot product
    # against the query vector produces a non-finite score -- no need to
    # reproduce a real BLAS bug, just prove the guard catches it.
    query_vec = [1.0, 0.0]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [np.nan, 0.0],
            [0.0, 1.0],
        ]
    )
    metadata = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    with pytest.raises(RuntimeError, match="non-finite"):
        top_k_similar(
            "irrelevant since query_embedding is precomputed",
            embeddings,
            metadata,
            k=3,
            query_embedding=query_vec,
        )


def test_top_k_similar_respects_k_and_uses_injectable_embed_fn():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )
    metadata = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    def fake_embed_fn(texts):
        assert texts == ["my flight got cancelled"]
        return [[1.0, 0.0]]

    results = top_k_similar(
        "my flight got cancelled", embeddings, metadata, k=2, embed_fn=fake_embed_fn
    )

    assert len(results) == 2
    assert [r["id"] for r in results] == ["a", "b"]


# --- gemini_client disk cache ---


def test_embed_texts_only_calls_api_for_cache_misses(tmp_path, monkeypatch):
    monkeypatch.setattr(gemini_client, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(gemini_client, "_get_client", lambda: object())

    call_log = []

    def fake_call_embed_api(client, model, batch, task_type=None):
        call_log.append(list(batch))
        return [[float(len(text))] for text in batch]

    monkeypatch.setattr(gemini_client, "_call_embed_api", fake_call_embed_api)

    first = gemini_client.embed_texts(["a", "bb", "ccc"], model="test-model")
    assert first == [[1.0], [2.0], [3.0]]
    assert call_log == [["a", "bb", "ccc"]]

    call_log.clear()
    second = gemini_client.embed_texts(["bb", "ccc", "dddd"], model="test-model")
    assert second == [[2.0], [3.0], [4.0]]
    # Only the uncached text ("dddd") should have triggered a fresh API call.
    assert call_log == [["dddd"]]


def test_embed_texts_preserves_input_order_across_cache_and_misses(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gemini_client, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(gemini_client, "_get_client", lambda: object())

    def fake_call_embed_api(client, model, batch, task_type=None):
        return [[float(len(text))] for text in batch]

    monkeypatch.setattr(gemini_client, "_call_embed_api", fake_call_embed_api)

    gemini_client.embed_texts(["a", "bb"], model="test-model")  # warm the cache

    result = gemini_client.embed_texts(["bb", "a", "ccc"], model="test-model")
    assert result == [[2.0], [1.0], [3.0]]


def test_embed_texts_cache_key_includes_task_type(tmp_path, monkeypatch):
    # CRITICAL behavior: the same text embedded under a different task_type
    # must NOT reuse another task_type's cached vector -- it's a different
    # embedding at the API level (e.g. RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY).
    monkeypatch.setattr(gemini_client, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(gemini_client, "_get_client", lambda: object())

    call_log = []

    def fake_call_embed_api(client, model, batch, task_type=None):
        call_log.append((task_type, list(batch)))
        return [[float(len(text))] for text in batch]

    monkeypatch.setattr(gemini_client, "_call_embed_api", fake_call_embed_api)

    gemini_client.embed_texts(["hello"], model="test-model", task_type="RETRIEVAL_DOCUMENT")
    gemini_client.embed_texts(["hello"], model="test-model", task_type="RETRIEVAL_QUERY")
    gemini_client.embed_texts(["hello"], model="test-model")

    # Three distinct task_type values (including None) -> three fresh calls,
    # zero cache hits between them.
    assert call_log == [
        ("RETRIEVAL_DOCUMENT", ["hello"]),
        ("RETRIEVAL_QUERY", ["hello"]),
        (None, ["hello"]),
    ]

    call_log.clear()
    gemini_client.embed_texts(["hello"], model="test-model", task_type="RETRIEVAL_DOCUMENT")
    # Re-requesting the same (model, task_type, text) triple is a cache hit.
    assert call_log == []


# --- build_index filtering + sampling ---


def _pair(customer_id, is_thread_root, is_dm_redirect):
    return {
        "customer_tweet_id": customer_id,
        "customer_text_raw": "raw",
        "customer_text_clean": "clean",
        "brand_reply_tweet_id": f"r-{customer_id}",
        "brand_reply_text_raw": "raw reply",
        "brand_reply_text_clean": "clean reply",
        "created_at": "2020-01-01",
        "is_thread_root": is_thread_root,
        "is_dm_redirect": is_dm_redirect,
    }


def test_filter_pool_keeps_only_root_non_redirect_pairs():
    pairs = [
        _pair("1", True, False),  # keep
        _pair("2", True, True),  # drop: dm redirect
        _pair("3", False, False),  # drop: not thread root
        _pair("4", False, True),  # drop: neither
        _pair("5", True, False),  # keep
    ]
    result = filter_pool(pairs)
    assert {p["customer_tweet_id"] for p in result} == {"1", "5"}


def test_sample_pool_keeps_all_when_pool_smaller_than_target():
    pool = [_pair(str(i), True, False) for i in range(5)]
    sampled = sample_pool(pool, pool_size=100, seed=42)
    assert len(sampled) == 5
    assert sampled == pool


def test_sample_pool_samples_exactly_pool_size_when_larger():
    pool = [_pair(str(i), True, False) for i in range(50)]
    sampled = sample_pool(pool, pool_size=10, seed=42)
    assert len(sampled) == 10
    sampled_ids = {p["customer_tweet_id"] for p in sampled}
    pool_ids = {p["customer_tweet_id"] for p in pool}
    assert sampled_ids.issubset(pool_ids)


def test_sample_pool_is_reproducible_with_same_seed():
    pool = [_pair(str(i), True, False) for i in range(50)]
    sampled_a = sample_pool(pool, pool_size=10, seed=42)
    sampled_b = sample_pool(pool, pool_size=10, seed=42)
    assert sampled_a == sampled_b


# --- gemini_client rate-limit retry logic (no real sleep() during tests) ---


def test_embed_batch_with_retry_parses_retry_delay_from_client_error(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(gemini_client.time, "sleep", lambda s: sleep_calls.append(s))

    attempts = {"n": 0}

    def fake_call_embed_api(client, model, batch, task_type=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise FakeClientError(
                {
                    "error": {
                        "code": 429,
                        "status": "RESOURCE_EXHAUSTED",
                        "details": [
                            {
                                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                                "retryDelay": "55s",
                            }
                        ],
                    }
                }
            )
        return [[1.0]]

    monkeypatch.setattr(gemini_client, "_call_embed_api", fake_call_embed_api)

    result = gemini_client._embed_batch_with_retry(object(), "model", ["a"])

    assert result == [[1.0]]
    assert sleep_calls == [55.0]


def test_embed_batch_with_retry_falls_back_to_60s_when_unparseable(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(gemini_client.time, "sleep", lambda s: sleep_calls.append(s))

    attempts = {"n": 0}

    def fake_call_embed_api(client, model, batch, task_type=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("some other kind of failure, no structured details")
        return [[1.0]]

    monkeypatch.setattr(gemini_client, "_call_embed_api", fake_call_embed_api)

    result = gemini_client._embed_batch_with_retry(object(), "model", ["a"])

    assert result == [[1.0]]
    assert sleep_calls == [60]


def test_embed_batch_with_retry_gives_up_after_five_attempts(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(gemini_client.time, "sleep", lambda s: sleep_calls.append(s))

    def always_fails(client, model, batch, task_type=None):
        raise RuntimeError("always fails")

    monkeypatch.setattr(gemini_client, "_call_embed_api", always_fails)

    with pytest.raises(RuntimeError):
        gemini_client._embed_batch_with_retry(object(), "model", ["a"])

    # max_retries=5 -> 5 attempts total, 4 waits between them, no sleep after the last.
    assert len(sleep_calls) == 4


def test_embed_texts_paces_between_batches_and_reports_progress(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(gemini_client, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(gemini_client, "_get_client", lambda: object())

    sleep_calls = []
    monkeypatch.setattr(gemini_client.time, "sleep", lambda s: sleep_calls.append(s))

    def fake_call_embed_api(client, model, batch, task_type=None):
        return [[float(len(t))] for t in batch]

    monkeypatch.setattr(gemini_client, "_call_embed_api", fake_call_embed_api)

    texts = ["a", "bb", "ccc", "dddd", "e"]
    gemini_client.embed_texts(texts, model="test-model", batch_size=2)

    # 5 texts, batch_size=2 -> 3 batches -> pacing sleep after batches 1 and 2,
    # not after the final batch.
    assert sleep_calls == [3, 3]

    out = capsys.readouterr().out
    assert "Embedded batch 1/3 (2/5 texts)..." in out
    assert "Embedded batch 2/3 (4/5 texts)..." in out
    assert "Embedded batch 3/3 (5/5 texts)..." in out
