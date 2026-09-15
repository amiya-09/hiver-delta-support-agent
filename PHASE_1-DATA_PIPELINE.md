# Phase 1: Data Pipeline

## What I built

A script that takes the raw 492MB Kaggle Twitter support dataset, filters
it down to just Delta, reconstructs customer-to-brand reply pairs, and
applies some basic cleaning. The output is one clean file that every
later phase (retrieval, classification, generation, escalation) reads
from directly.

This is also where I handle a problem I noticed early on: a lot of what
looks like a "customer message" in the raw data is actually just a
mid-conversation reply, like someone tweeting back a confirmation
number, not a real opening complaint. I tag every customer message with
`is_thread_root` so later phases can tell the difference and only use
real opening complaints where it matters.

## Files

- `src/pipeline/__init__.py`
- `src/pipeline/clean.py`, pure text cleaning functions, no I/O, easy to
  unit test on their own
- `src/pipeline/threads.py`, the actual pair reconstruction logic
- `src/pipeline/build_dataset.py`, the CLI script that ties it all
  together: load, pair, clean, save
- `tests/test_pipeline.py`, unit tests against small hand-built fixture
  data, not the real dataset

## Edge cases

| Case | How I handle it |
|---|---|
| Brand reply's parent tweet is missing from the file | Skip it, count as `dropped_broken_chain` |
| Brand reply points to another brand's tweet, not a customer | Skip it, count as `dropped_parent_not_customer` |
| Customer message is actually a mid-thread reply, not an opening one | Keep it, but tag `is_thread_root: false` so it gets filtered out downstream when it matters, still usable for retrieval |
| Same customer posts a near-duplicate complaint again | Drop the repeat (exact match on normalized text), keep the first one |
| Message is empty or meaningless once mentions and links are stripped | Drop it, count as `dropped_too_short` |
| Reply is just a "please DM us" redirect, not a real resolution | Keep the pair but flag `is_dm_redirect: true`, so retrieval can exclude it later without losing the row entirely |
| Multiple response tweet IDs on one row | Not actually relevant here since I only ever walk `in_response_to_tweet_id` backward, and that field is always a single ID |

## Running it against the real data
python3 -m src.pipeline.build_dataset --input data/raw/twcs.csv --brand Delta --output-dir data/processed


Output:

Total rows loaded: 2811774
Pairs written: 34435 -> data/processed/delta_pairs.jsonl
Stats:
total_brand_replies_seen: 42253
dropped_no_parent_id: 56
dropped_broken_chain: 48
dropped_parent_not_customer: 35
dropped_too_short: 2165
dropped_duplicate: 5514
dm_redirect_count: 6331
thread_root_count: 24161
thread_nonroot_count: 10274
kept_pairs: 34435

## Checking the numbers against what I expected

`total_brand_replies_seen: 42253` matches the Delta total I got earlier
from a completely separate exploration script when I was picking a
brand. Two independently written pieces of code agreeing on the same
number is a decent sanity check that neither one is quietly wrong.

The DM-redirect rate needed correcting. Early on, working from a tiny
93-row sample across several brands, I estimated something like 45% of
replies were DM-redirects. The real Delta-only number, from the full
34,435 pairs, is 6,331 / 34,435, which is 18.4%. Meaningfully lower than
my first guess, and it lines up better with what I'd seen reading actual
Delta threads by hand later (1 of 4 samples were redirects, not close to
half). Worth remembering not to trust a tiny early sample too much.

`thread_nonroot_count: 10,274` confirmed something I'd only noticed by
eye before this: those mid-thread fragments aren't a rare edge case,
they're 30% of all the pairs I pulled out. Glad I built the
`is_thread_root` tagging into this phase instead of finding this out
later while labelling the golden set.

The retention math also checks out cleanly: 42,253 minus everything
dropped (56 + 48 + 35 + 2,165 + 5,514) equals 34,435, which matches
`kept_pairs` exactly. Nothing is silently disappearing or getting
double-counted somewhere. Overall I kept about 81.5% of the original
replies, losing most of the rest to duplicates and short/empty messages,
both of which are expected and not concerning.
