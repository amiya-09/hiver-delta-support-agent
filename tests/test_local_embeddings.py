import pytest

import src.llm.local_embeddings as local_embeddings


# --- task_type -> prefix mapping ---


def test_apply_task_prefix_document():
    assert local_embeddings._apply_task_prefix("my flight was cancelled", "RETRIEVAL_DOCUMENT") == (
        "passage: my flight was cancelled"
    )


def test_apply_task_prefix_query():
    assert local_embeddings._apply_task_prefix("my flight was cancelled", "RETRIEVAL_QUERY") == (
        "query: my flight was cancelled"
    )


def test_apply_task_prefix_none_and_unknown_have_no_prefix():
    assert local_embeddings._apply_task_prefix("my flight was cancelled", None) == "my flight was cancelled"
    assert local_embeddings._apply_task_prefix("my flight was cancelled", "SOMETHING_ELSE") == (
        "my flight was cancelled"
    )


# --- cache hits avoid re-encoding ---


def test_embed_texts_only_encodes_cache_misses(tmp_path, monkeypatch):
    monkeypatch.setattr(local_embeddings, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(local_embeddings, "_get_model", lambda model_name: object())

    call_log = []

    def fake_encode_batch(model, prefixed_texts, batch_size):
        call_log.append(list(prefixed_texts))
        return [[float(len(t))] for t in prefixed_texts]

    monkeypatch.setattr(local_embeddings, "_encode_batch", fake_encode_batch)

    first = local_embeddings.embed_texts(
        ["a", "bb", "ccc"], model="test-model", task_type="RETRIEVAL_DOCUMENT"
    )
    assert first == [[float(len("passage: a"))], [float(len("passage: bb"))], [float(len("passage: ccc"))]]
    assert call_log == [["passage: a", "passage: bb", "passage: ccc"]]

    call_log.clear()
    second = local_embeddings.embed_texts(
        ["bb", "ccc", "dddd"], model="test-model", task_type="RETRIEVAL_DOCUMENT"
    )
    assert second == [
        [float(len("passage: bb"))],
        [float(len("passage: ccc"))],
        [float(len("passage: dddd"))],
    ]
    # Only the uncached text ("dddd") should have triggered a fresh encode call.
    assert call_log == [["passage: dddd"]]


def test_embed_texts_cache_key_includes_task_type(tmp_path, monkeypatch):
    monkeypatch.setattr(local_embeddings, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(local_embeddings, "_get_model", lambda model_name: object())

    call_log = []

    def fake_encode_batch(model, prefixed_texts, batch_size):
        call_log.append(list(prefixed_texts))
        return [[float(len(t))] for t in prefixed_texts]

    monkeypatch.setattr(local_embeddings, "_encode_batch", fake_encode_batch)

    local_embeddings.embed_texts(["hello"], model="test-model", task_type="RETRIEVAL_DOCUMENT")
    local_embeddings.embed_texts(["hello"], model="test-model", task_type="RETRIEVAL_QUERY")
    local_embeddings.embed_texts(["hello"], model="test-model")

    # Three distinct task_type values (including None) -> three fresh encode
    # calls, no cache hits between them despite identical raw text.
    assert call_log == [["passage: hello"], ["query: hello"], ["hello"]]

    call_log.clear()
    local_embeddings.embed_texts(["hello"], model="test-model", task_type="RETRIEVAL_DOCUMENT")
    # Re-requesting the same (model, task_type, text) triple is a cache hit.
    assert call_log == []


# --- input order preserved across cache hits + fresh results ---


def test_embed_texts_preserves_input_order_across_cache_and_misses(tmp_path, monkeypatch):
    monkeypatch.setattr(local_embeddings, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(local_embeddings, "_get_model", lambda model_name: object())

    def fake_encode_batch(model, prefixed_texts, batch_size):
        return [[float(len(t))] for t in prefixed_texts]

    monkeypatch.setattr(local_embeddings, "_encode_batch", fake_encode_batch)

    local_embeddings.embed_texts(["a", "bb"], model="test-model")  # warm the cache

    result = local_embeddings.embed_texts(["bb", "a", "ccc"], model="test-model")
    assert result == [[float(len("bb"))], [float(len("a"))], [float(len("ccc"))]]


# --- missing sentence-transformers dependency ---


def test_require_sentence_transformers_raises_clear_runtime_error(monkeypatch):
    monkeypatch.setattr(local_embeddings, "SentenceTransformer", None)
    monkeypatch.setattr(local_embeddings, "_IMPORT_ERROR", ImportError("no module named sentence_transformers"))

    with pytest.raises(RuntimeError, match="pip install sentence-transformers"):
        local_embeddings._require_sentence_transformers()


def test_get_model_raises_when_sentence_transformers_missing(monkeypatch):
    monkeypatch.setattr(local_embeddings, "SentenceTransformer", None)
    monkeypatch.setattr(local_embeddings, "_IMPORT_ERROR", ImportError("no module named sentence_transformers"))

    with pytest.raises(RuntimeError, match="pip install sentence-transformers"):
        local_embeddings._get_model("intfloat/e5-small-v2")
