# Phase 4 Failure Analysis: Reply Generation

I only tested this on 10 examples by hand, not the full golden set,
since there was no automated quality score yet at this point (that came
later with the judge in Phase 6). Treat this as a real, specific
finding from reading actual output closely, not a formal statistic.

## Retrieval hands back a "looks fine" score even when nothing is actually relevant

Across all 10 queries, the top retrieved examples' similarity scores
all fell somewhere between 0.83 and 0.90, no matter how relevant the
message actually was, even for an unusual complaint that wasn't well
matched by anything in the pool. My prompt tells the model to skip
grounding and give a generic response if nothing retrieved is genuinely
relevant, but that instruction never got tested in this sample, since
retrieval never handed back a score low enough to obviously look weak.
The instruction itself might be fine, I just never saw the branch where
it should have fired.

My best guess is that with a small, general-purpose embedding model on
short tweet text, a lot of genuinely different problems share enough
surface vocabulary that similarity scores end up clustering in a narrow
band regardless of true relevance, even though the relative ranking,
best match versus worst match, still worked correctly in every case I
checked back in Phase 2.

## That gap produced a real hallucination

One customer complained about a specific paid wifi service failure and
asked for a refund. The generated reply copied a phone number, word for
word, from a retrieved example about a completely different customer's
wifi problem, with a 0.88 similarity score sitting right in the same
"looks fine" range as everything else in the sample, and the model's
own confidence note claimed high confidence.

A support bot confidently handing out a specific phone number that
might not actually be right for a given customer's situation isn't just
unhelpful, it's actively misleading, and it happened specifically
because the prompt's safety valve, returning empty grounding when
nothing's relevant, never had a chance to fire.

## What actually worked

Two of the ten examples did exactly what I wanted. One reused a real
claim-tag and baggage-service-desk process from a genuinely on-topic
retrieved case for a broken-luggage complaint. Another correctly
explained a weight-and-balance reason for a forced bag check, also
pulled from real, relevant precedent. Both reused specific procedural
details, not just matching tone.

## A fix I drafted but haven't tested

I wrote a similarity-floor instruction telling the model to hedge more
when the best retrieved score is still on the low end of that narrow
band, even if it still cites the example. I haven't tested it given the
Groq budget I had left at the time. This same underlying gap showed up
again later, in a more serious way, once I built the judge in Phase 6
and it turned out to have the identical blind spot.
