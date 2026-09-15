# Phase 2: Retrieval Index

## What I built

A searchable pool of real, historical Delta customer complaints paired
with how they actually got resolved, so that at runtime I can look up
"has something like this happened before, and what worked" for any new
incoming message. This is the piece that makes "grounded in historical
resolution" an actual mechanism rather than something the model just
claims to be doing.

The pool only includes pairs where the customer message is a genuine
opening complaint (`is_thread_root`) and the reply isn't just a
DM-redirect, since those don't count as real resolutions. Target size
was 2,500, randomly sampled with a fixed seed from whatever's left after
filtering. Search itself is just brute-force cosine similarity in numpy,
no vector database, since 2,500 vectors is small enough that it doesn't
need one.

## Files

- `src/llm/gemini_client.py`, my first attempt at an embedding backend,
  kept in the repo as a working, tested fallback even though it's not
  what actually runs anymore
- `src/llm/local_embeddings.py`, the backend I actually use, a local
  model instead of an API
- `src/retrieval/__init__.py`
- `src/retrieval/build_index.py`, filters the pool, samples it, embeds
  it, saves the index
- `src/retrieval/query.py`, loads the index and returns the top-k most
  similar past cases for a new message
- `tests/test_retrieval.py`, `tests/test_local_embeddings.py`, all
  offline, no real API or model calls

## Edge cases

| Case | How I handle it |
|---|---|
| Filtered pool smaller than the target pool size | Use everything available, log a warning, don't crash |
| Script interrupted partway through embedding | Every embedding gets cached to disk as it's computed, so a rerun picks up where it left off instead of starting over |
| Same text embedded more than once | Cache hit, no wasted work |
| Embedding call returns a different count than what was requested | Explicit length check, raise an error rather than silently misaligning text and vectors |
| A similarity score comes back as NaN or infinite | Raise immediately rather than silently return a corrupted ranking (see below for why this actually happened) |

## What actually happened building this

I started with Gemini's embedding API. Caught one real issue before
spending any real calls on it: Gemini's embedding model distinguishes
between embedding something to be searched versus embedding the search
query itself, and if I didn't set that correctly, retrieval quality
would have quietly suffered without any obvious error telling me so.
Fixed that before running anything for real.

Then I actually ran the full 2,500-pair build and hit Gemini's daily
quota partway through. Waiting for it to reset would have cost a real
chunk of my week for what's meant to be a one-time setup step, so I
switched to a small local embedding model instead (e5-small-v2, via
sentence-transformers). No API, no quota, and it rebuilt the whole pool
in under a minute once the model weights were downloaded.
0 cache hits, 2500 freshly computed
Summary:
pool size before filter: 34435
pool size after filter: 19320
final sampled size: 2500
embedding dimensionality: 384

I spot-checked retrieval quality by hand with 4 queries I picked myself
(a flight delay, a lost bag, a seat change, a thank-you message) plus
100 random real ones. In every case the results were actually on-topic,
no baggage complaints turning up for a delay question or anything like
that.

While testing queries I ran into something odd: NumPy occasionally
returned NaN or infinite values from what should have been a completely
normal similarity calculation. Turned out to be a known issue with
Apple's Accelerate framework, which is what NumPy uses as its math
backend on Mac by default, not anything wrong with my data or code. I
could reproduce it once with a real query, but running the same kind of
check 100 more times afterward, it never happened again. Given how
inconsistent it was and how little time I had left to chase down a full
NumPy rebuild against a different backend, I added a check that raises
an error immediately if it ever happens again instead of silently
returning bad results. Not a perfect fix, but it means I'd know
immediately if it recurred instead of quietly trusting a broken ranking.
