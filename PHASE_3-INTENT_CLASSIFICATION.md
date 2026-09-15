# Phase 3: Intent Classification

## What I built

Given a customer message, classify it into one of 8 categories I'd
already defined from reading real data. This is the first of the three
required agent behaviors, and the first one I could actually check
against real ground truth, since the golden set already existed by this
point. The model returns structured JSON, not free text, so nothing
downstream has to parse messy output, and it includes a second-choice
field for cases where it was genuinely torn between two categories.

## Files

- `prompts/classify_intent.txt`, the actual prompt, versioned as its own
  file so I can track changes to it separately from the code
- `src/agent/classify.py`
- `src/llm/groq_retry.py`, retry and rate-limit handling shared between
  classification and everything that came after it
- `src/eval/metrics.py`, accuracy/F1/confusion matrix, plain Python, no
  extra dependency needed for something this small
- `src/eval/run_classification_eval.py`, runs the classifier against the
  golden set and scores it
- `src/eval/run_baseline_eval.py`, the two required comparison baselines
- `src/eval/run_targeted_recheck.py`, a cheaper way to test a prompt
  change without re-running all 240 examples every time
- Matching tests for all of the above, no real API calls in any of them

## Edge cases

| Case | How I handle it |
|---|---|
| Model returns malformed JSON | Retry once with a stricter instruction, then fail loudly if it still doesn't parse |
| Model returns a category name that isn't one of my 8 | Raise immediately, this should never happen and means something's actually wrong |
| Model returns the string "null" instead of real JSON null for the optional second-choice field | Normalize it to a real empty value before validating, since the model clearly meant "no second choice" |
| Rate limited | Retry using the wait time the server actually tells me, up to 5 attempts |
| A non-rate-limit error (like a bad model name) | Fail immediately, don't waste time retrying something retrying can't fix |
| A golden-set example I'd marked ambiguous when labelling | Still scored normally, but I also report accuracy with those excluded, since I don't want one number hiding that distinction |

## Real results
Overall accuracy: 0.7583
Accuracy excluding 6 ambiguous examples: 0.7564

Against a trivial baseline (always guess the most common category) of
20.0%, and a simple keyword-rule baseline of 59.6%.

## What actually happened

This phase had more infrastructure problems than actual modeling
problems. I started on Gemini and hit a daily cap of just 20 requests
for this specific project, well under the normal free tier, so I moved
classification over to Groq instead. First model name I tried there
didn't exist for my account either, so I had to actually query what
models I did have access to rather than trust general documentation.
While debugging that I found a real bug in my own retry code: it was
treating every kind of error as a rate limit worth waiting out, which
meant a genuinely unrelated error wasted several minutes retrying
something that was never going to succeed. Fixed it to only retry on
actual rate limit errors.

Once it was actually running against the full golden set, I read
through the wrong answers using the model's own stated reasoning for
each one, not just the raw confusion matrix. Two clear patterns showed
up: it kept correctly identifying the right topic in its own reasoning
(like naming SkyMiles specifically) and then still picking a more
generic category because of how the message was phrased, a question
going to "policy question" instead of "account loyalty," a frustrated
message going to "service complaint" instead of "refund." A third,
smaller pattern turned out to not be a real error at all, just a
genuine gray area in my own taxonomy around messages that thank someone
for fixing a problem with no remaining complaint.

I wrote a rule directly targeting the first pattern and wanted to test
it, but by then I'd also discovered Groq's free tier caps out at
200,000 tokens per day, and my prompt is long enough that a full
240-example run eats a big chunk of that by itself. So instead of
re-running everything, I built a smaller tool that only re-checks the
examples that were previously wrong, plus a random sample of previously
correct ones to catch anything the fix might have broken. That found
22 of 58 known errors fixed, but also 2 new regressions in the sample I
hadn't touched, which nets out to a real improvement (about 79.9% up
from 75.83%) but not a completely clean one. I went back and looked at
those 2 regressions specifically instead of just accepting the net
number. One was the model over-applying my new rule to a case where the
loyalty term was only mentioned in passing, not what the message was
actually about. The other was stranger, a security-incident complaint
getting reclassified as a flight delay for no reason connected to my
actual fix, which suggests the rule's general idea generalized further
than I intended, not just the two specific cases I wrote it for.
