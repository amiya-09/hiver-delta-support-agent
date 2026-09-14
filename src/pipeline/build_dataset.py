"""CLI entry point: load raw Delta-support-tweet CSV, reconstruct pairs,
write the cleaned dataset and pipeline stats to disk.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.pipeline.threads import build_row_index, reconstruct_pairs

USECOLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "in_response_to_tweet_id",
]


def load_raw_rows(csv_path):
    """Load the raw CSV and parse `inbound` into a real bool.

    Read everything as str with keep_default_na=False so that empty
    in_response_to_tweet_id fields come back as "" (not NaN) -- reconstruct_pairs
    relies on that to detect "no parent" via a simple falsy check.
    """
    df = pd.read_csv(csv_path, usecols=USECOLS, dtype=str, keep_default_na=False)
    df["inbound"] = df["inbound"].str.strip().str.lower() == "true"
    return df.to_dict("records")


def run(input_path, brand, output_dir):
    rows = load_raw_rows(input_path)
    row_index = build_row_index(rows)
    pairs, stats = reconstruct_pairs(rows, brand, row_index=row_index)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = output_dir / f"{brand.lower()}_pairs.jsonl"
    with pairs_path.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    stats_dict = stats.as_dict()
    stats_path = output_dir / "pipeline_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats_dict, f, indent=2)

    print(f"Total rows loaded: {len(rows)}")
    print(f"Pairs written: {len(pairs)} -> {pairs_path}")
    print("Stats:")
    for key, value in stats_dict.items():
        print(f"  {key}: {value}")
    print(f"Stats written -> {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Build Delta support-tweet pairs dataset.")
    parser.add_argument("--input", required=True, help="Path to raw twcs.csv")
    parser.add_argument("--brand", required=True, help="Brand author_id to filter to, e.g. Delta")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()
    run(args.input, args.brand, args.output_dir)


if __name__ == "__main__":
    main()
