# Phase 5: Routing and Escalation

## What I built

The last of the three required agent behaviors: given a message's
intent and how well it matched retrieved precedent, decide whether to
auto-handle it or send it to a human, with an actual stated reason. This
one is pure rule logic, no model call, checked in a fixed order where
the first matching rule wins. I also wired everything built so far,
cleaning, classification, retrieval, generation, and this, into one
single function that represents the whole agent end to end.

## Files

- `src/agent/route.py`
- `src/agent/pipeline.py`, the end-to-end wiring
- `tests/test_route.py`, `tests/test_agent_pipeline.py`

## Edge cases

| Case | How I handle it |
|---|---|
| A golden-set example is also sitting inside the retrieval pool itself | Detected and filtered out before computing similarity, otherwise it would just find itself and score artificially close to perfect |
| The same tweet ID somehow appears more than once in the pool | The filtering loops and expands its search instead of assuming there's only ever one match to remove |
| The weak-grounding threshold barely ever fires | Documented directly in the code as a known limitation, since Phase 4 already showed retrieval scores cluster tightly regardless of true relevance |

## Real results

I evaluated this two ways: once giving it the correct golden intent
directly, to test the routing logic in isolation, and once giving it
Phase 3's actual predicted intent, to see how real classification
errors affect it once they're in the loop.
ESCALATE recall (isolated): 1.0000
ESCALATE recall (end-to-end): 1.0000
Rule-level accuracy (isolated): 0.808
Rule-level accuracy (end-to-end): 0.738

Recall being perfect in both cases means every single message that
should have escalated did escalate, even once real classification
mistakes were stacked on top. That's exactly the outcome I designed for,
biased toward the safer of the two possible mistakes, and now it's
actually verified instead of just argued for on paper.

## What actually happened

About 1 in 10 golden examples turned out to also be sitting in the
retrieval pool, since both were sampled from the same underlying data.
Had to filter those out before trusting any similarity score, otherwise
a chunk of the evaluation would have quietly been testing against
retrieval finding itself.

The category rule (auto-escalate anything tagged baggage or refund) is
blunter than I'd realized. Of the 60 golden examples in those two
categories, only 35 actually needed escalation by my own labelling. The
rule escalates all 60 regardless, so it's wrong on 42% of that category
by itself.

The most interesting thing I found came from comparing the isolated and
end-to-end results side by side. The category rule's false positive
count actually looked better in the end-to-end run, dropping from 25
down to 14, which looked at first like the pipeline was somehow doing
better as a whole than its individual pieces. I didn't trust that and
went and traced the actual examples instead of taking the aggregate
number at face value. It turned out almost all of that apparent
improvement was really just Phase 3 misclassifying those messages into
some other category entirely, which accidentally moved them out of the
one rule that would have wrongly escalated them. Three of those
examples were literally the same ones I'd already flagged as a
classification failure back in Phase 3. So the pipeline wasn't actually
getting smarter together, one bug was just quietly hiding another one.
If I fix the Phase 3 classification issue later, I'd actually expect
this specific number to get worse, not better, which is a real,
checkable prediction rather than just a guess.
