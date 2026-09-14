"""Build the retrieval index: embed a sampled pool of resolved customer
complaints so later phases can find similar past issues.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.llm.local_embeddings import embed_texts


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def filter_pool(pairs):
    """Keep only genuine, resolved opening complaints.

    is_thread_root=True excludes mid-thread fragments (we want the
    customer's actual opening complaint, not "@Delta 276"). is_dm_redirect
    is_false excludes replies that just punted to DMs instead of showing
    an actual resolution.
    """
    return [
        p
        for p in pairs
        if p.get("is_thread_root") is True and p.get("is_dm_redirect") is False
    ]


def sample_pool(pool, pool_size, seed):
    """Sample exactly pool_size pairs, or all of them if fewer are available."""
    if len(pool) > pool_size:
        rng = random.Random(seed)
        return rng.sample(pool, pool_size)
    if len(pool) < pool_size:
        print(
            f"WARNING: filtered pool has only {len(pool)} pairs, fewer than "
            f"the target pool-size of {pool_size}. Using all of them."
        )
    return pool


def run(input_path, output_dir, pool_size, seed):
    pairs = load_pairs(input_path)
    before = len(pairs)

    pool = filter_pool(pairs)
    after = len(pool)
    print(f"Pairs before filter: {before}")
    print(f"Pairs after filter (is_thread_root and not is_dm_redirect): {after}")

    sampled = sample_pool(pool, pool_size, seed)
    final_size = len(sampled)

    texts = [p["customer_text_clean"] for p in sampled]
    embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    arr = np.array(embeddings, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = arr / norms

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", normalized)

    with (output_dir / "metadata.jsonl").open("w") as f:
        for pair in sampled:
            f.write(json.dumps(pair) + "\n")

    dim = normalized.shape[1] if final_size else 0
    print("Summary:")
    print(f"  pool size before filter: {before}")
    print(f"  pool size after filter: {after}")
    print(f"  final sampled size: {final_size}")
    print(f"  embedding dimensionality: {dim}")


def main():
    parser = argparse.ArgumentParser(description="Build the Delta retrieval index.")
    parser.add_argument("--input", default="data/processed/delta_pairs.jsonl")
    parser.add_argument("--output-dir", default="data/processed/retrieval_index")
    parser.add_argument("--pool-size", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.input, args.output_dir, args.pool_size, args.seed)


if __name__ == "__main__":
    main()
