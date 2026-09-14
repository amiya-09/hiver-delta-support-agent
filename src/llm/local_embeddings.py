"""Local, open-source embeddings backend -- no API, no rate/quota limits.

Same logical interface as gemini_client.embed_texts, so retrieval code can
switch backends by changing one import. Kept as a separate module (rather
than folded into gemini_client.py) so gemini_client stays untouched as a
documented, working fallback.
"""
import hashlib
import json
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    SentenceTransformer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

CACHE_DIR = Path(".cache/local_embeddings")

_MODEL_CACHE = {}

_TASK_TYPE_PREFIXES = {
    "RETRIEVAL_DOCUMENT": "passage: ",
    "RETRIEVAL_QUERY": "query: ",
}


def _require_sentence_transformers():
    """Raise a clear, actionable error instead of a raw ImportError.

    Factored out (rather than raising at module import time) so the error
    path is directly testable without needing to actually uninstall the
    package -- tests can monkeypatch SentenceTransformer to None and call
    this function to simulate the missing-dependency case.
    """
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed. Run "
            "`pip install sentence-transformers` to use the local embeddings backend."
        ) from _IMPORT_ERROR


def _get_model(model_name):
    """Lazily load and cache the SentenceTransformer model by name.

    Loading a transformer model is slow (disk + memory), so we only want
    to pay that cost once per process per model name, not once per
    embed_texts() call.
    """
    _require_sentence_transformers()
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _apply_task_prefix(text, task_type):
    """e5 models expect a "query: "/"passage: " prefix on the raw text
    instead of a separate API parameter like Gemini's task_type."""
    return _TASK_TYPE_PREFIXES.get(task_type, "") + text


def _cache_path(model, text, task_type=None):
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
            {"model": model, "task_type": task_type, "text": text, "embedding": embedding},
            f,
        )


def _encode_batch(model, prefixed_texts, batch_size):
    """The actual model.encode() call, isolated so tests can monkeypatch
    just this without loading a real SentenceTransformer model."""
    array = model.encode(
        prefixed_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return [[float(x) for x in row] for row in array]


def embed_texts(texts, model="intfloat/e5-small-v2", task_type=None, batch_size=32):
    """Embed a list of texts locally, using the disk cache and batched encoding.

    Cache is checked first for every text; only cache misses are encoded,
    in batches of `batch_size`. Each freshly-computed embedding is written
    to the cache immediately after its batch finishes. Results are
    returned in the same order as `texts`, regardless of whether each one
    came from cache or was freshly computed.
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
        st_model = _get_model(model)
        for start in range(0, len(miss_indices), batch_size):
            batch_indices = miss_indices[start : start + batch_size]
            batch_texts = [texts[i] for i in batch_indices]
            prefixed_texts = [_apply_task_prefix(t, task_type) for t in batch_texts]
            embeddings = _encode_batch(st_model, prefixed_texts, batch_size)
            for idx, text, embedding in zip(batch_indices, batch_texts, embeddings):
                results[idx] = embedding
                _write_cache(model, text, embedding, task_type)

    computed = len(miss_indices)
    print(f"{cache_hits} cache hits, {computed} freshly computed")
    return results
