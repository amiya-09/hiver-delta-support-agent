# Phase 4: Reply Generation

## What I built

Given a customer message, its intent, and the top few retrieved past
cases, draft a reply that's actually grounded in that precedent instead
of just sounding plausible. The model also has to say which of the
retrieved examples it actually used, so grounding is something I can
check rather than just something it claims. I reused basically all the
caching and retry infrastructure I'd already built for classification,
rather than writing it a second time.

## Files

- `prompts/generate_reply.txt`
- `src/agent/generate.py`
- `src/eval/run_generation_eval.py`
- `tests/test_generate.py`

## Edge cases

| Case | How I handle it |
|---|---|
| None of the retrieved examples are actually relevant | The prompt tells the model to write a conservative, generic acknowledgment instead of forcing a connection, and return an empty grounding list |
| Model names a retrieved example that doesn't exist in what I gave it | Raise immediately, since this field is the whole point of being able to check grounding at all |
| Reply field comes back empty or missing | Raise immediately, same standard I already held the intent field to in Phase 3 |
| Malformed JSON | Retry once with a stricter instruction, then fail loudly |

## What actually happened

I only tested this on 10 examples rather than the full golden set,
partly because of the Groq budget situation and partly because I wanted
to actually read the output by hand first before trusting it at any real
scale. Good thing I did, because reading through it by hand surfaced
something the automated numbers alone wouldn't have caught yet.

Every one of the 10 retrieved-example sets came back with a similarity
score somewhere between 0.83 and 0.90, no matter how relevant the
message actually turned out to be. My prompt tells the model to skip
grounding entirely if nothing retrieved is genuinely relevant, but that
instruction never got a real chance to fire in this sample, since
retrieval never handed it a score low enough to look obviously weak.

That gap showed up as an actual mistake, not just a theoretical one. One
generated reply lifted a phone number straight out of a retrieved
example about a different customer's wifi problem and handed it to
someone with a completely different issue, all while the similarity
score looked exactly as normal as every other case in the sample. On the
other hand, two of the ten examples worked exactly the way I wanted,
reusing real, specific procedural details (a claim tag, a weight and
balance explanation) from genuinely on-topic precedent, not just
matching tone.

I drafted a fix, adding a similarity-floor instruction so the model
hedges more when the best retrieved score is still on the low end of
that narrow band, but haven't tested it yet given the Groq budget I had
left at the time. This same underlying issue turned out to matter again
later, in a more serious way, once I built the judge in Phase 6.
