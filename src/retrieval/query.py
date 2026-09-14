"""Load the retrieval index and find the top-k most similar past complaints."""
import json
from pathlib import Path

import numpy as np

from src.llm.local_embeddings import embed_texts


def load_index(index_dir):
    index_dir = Path(index_dir)
    embeddings = np.load(index_dir / "embeddings.npy")

    metadata = []
    with (index_dir / "metadata.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Index is corrupt: {len(embeddings)} embeddings but "
            f"{len(metadata)} metadata rows."
        )
    return embeddings, metadata


def top_k_similar(
    query_text,
    embeddings,
    metadata,
    k=5,
    query_embedding=None,
    embed_fn=None,
):
    """Return the top-k metadata rows most similar to query_text.

    `query_embedding` and `embed_fn` are injection points purely for
    testing without a real API call: pass a precomputed embedding to skip
    embedding entirely, or a fake embed function to stand in for the real
    embed_texts. The default (real) behavior is unchanged if neither is
    given.
    """
    if query_embedding is None:
        if embed_fn is not None:
            query_embedding = embed_fn([query_text])[0]
        else:
            query_embedding = embed_texts([query_text], task_type="RETRIEVAL_QUERY")[0]

    query_vector = np.array(query_embedding, dtype=np.float64)
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm

    scores = embeddings @ query_vector

    if not np.all(np.isfinite(scores)):
        bad_count = np.sum(~np.isfinite(scores))
        raise RuntimeError(
            f"Similarity computation produced {bad_count} non-finite "
            f"score(s) (NaN/Inf) out of {len(scores)} -- this usually "
            f"indicates a numerical/BLAS backend issue, not a data problem "
            f"(see decision log). Do not trust these results."
        )

    top_indices = np.argsort(-scores)[:k]

    results = []
    for idx in top_indices:
        row = dict(metadata[idx])
        row["similarity_score"] = float(scores[idx])
        results.append(row)
    return results
