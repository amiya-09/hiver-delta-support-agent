"""Run reply generation over a small random sample of the golden set.

Deliberately small default sample size: Phase 3 already taught us Groq's
daily token/request budget is a real constraint, so a full-golden-set
generation run should be an explicit later decision, not this script's
default. This script produces raw generated replies + grounding indices
for manual/later review only -- it does NOT score quality (that's Phase 6).
"""
import argparse
import csv
import json
import random
from pathlib import Path

from src.agent.generate import generate_reply
from src.retrieval.query import load_index, top_k_similar


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run(input_path, retrieval_index_dir, output_path, sample_size, k, seed):
    rows = load_rows(input_path)

    rng = random.Random(seed)
    if len(rows) > sample_size:
        sampled_rows = rng.sample(rows, sample_size)
    else:
        print(
            f"WARNING: golden set has only {len(rows)} rows, fewer than "
            f"the requested sample size of {sample_size}. Using all of them."
        )
        sampled_rows = rows

    embeddings, metadata = load_index(retrieval_index_dir)

    results = []
    total = len(sampled_rows)
    for i, row in enumerate(sampled_rows, start=1):
        example_id = row.get("example_id")
        text = row.get("customer_text_clean", "")
        true_intent = (row.get("true_intent") or "").strip().upper()

        retrieved_examples = top_k_similar(text, embeddings, metadata, k=k)
        result = generate_reply(text, true_intent, retrieved_examples)

        results.append(
            {
                "example_id": example_id,
                "customer_text_clean": text,
                "true_intent": true_intent,
                "retrieved_examples": retrieved_examples,
                "reply": result.get("reply"),
                "grounding_source_indices": result.get("grounding_source_indices"),
                "confidence_note": result.get("confidence_note"),
            }
        )

        print(f"{i}/{total}: {example_id} -- generated reply, grounded on {result.get('grounding_source_indices')}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"n_generated": len(results), "results": results}, f, indent=2)

    print(f"Generated {len(results)} replies. Full results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate replies for a small random sample of the golden set (Phase 4)."
    )
    parser.add_argument("--input", default="data/golden_set/candidates.csv")
    parser.add_argument("--retrieval-index-dir", default="data/processed/retrieval_index")
    parser.add_argument("--output", default="data/eval/phase4_generation_results.json")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.input, args.retrieval_index_dir, args.output, args.sample_size, args.k, args.seed)


if __name__ == "__main__":
    main()
