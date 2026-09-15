# Phase 6: LLM-as-Judge and Human Agreement

## What I built

An automated way to score generated replies on four separate dimensions
instead of one vague overall impression: whether the reply is actually
grounded in what was retrieved, whether it states anything as fact that
it shouldn't, whether the tone sounds right, and whether it actually
gives the customer something useful. Alongside that, a way to check
whether this judge can actually be trusted, by scoring the same replies
myself, blind to what the judge said, and measuring how often we agree.

## The real finding

This is the most important thing that came out of the whole project, so
I'm putting it first instead of burying it at the end.

I ran the judge and scored the same 30 replies myself, and on three of
the four dimensions the judge and I mostly agreed. But on
`factually_consistent`, the one dimension I specifically wrote to catch
the model inventing or misapplying details, every single disagreement
between us was a full swing from a 1 to a 3, never a soft 2-vs-3
difference like the other dimensions had. That's not normal
disagreement, that's the judge missing something completely.

I pulled the actual six cases where we disagreed that badly and read
through them, and they all share the same real problem. The model
states something specific and concrete that isn't actually true for
that particular customer, and the judge rates it as completely fine
anyway. Two different customers with two unrelated problems, a
companion-certificate question and a seat-refund dispute, both got
handed almost the exact same phone number, clearly reused rather than
actually relevant. One reply flatly claimed "we're aware of the issue
and are working to fix it," which nobody confirmed anywhere. Another
invented an oddly specific rule, exactly 45 linear inches for carry-on
snowshoes, that isn't in anything the system actually retrieved. Two
more stated a flat "no refund" as if it were settled policy, borrowed
from a different customer's situation that happened to end that way.

Every one of these got a perfect score of 3 from the judge. It looks
like the judge is checking whether a reply topically relates to
something retrieved, not whether the specific claim in it is actually
true for this specific customer, which is exactly the blind spot I
wrote the rubric instruction to catch, and it's failing at it anyway.
That means the one dimension built specifically to catch fabrication is
the weakest one the judge has, and it's not a borderline miss either,
it's a clean 0.80 exact-match rate with zero soft disagreements. This
has to be stated plainly in the report rather than smoothed over.

## Files

- `prompts/judge_reply.txt`
- `src/eval/judge.py`
- `src/eval/run_judge_eval.py`
- `src/eval/prepare_human_scoring.py`, builds the blank sheet I score by
  hand
- `src/eval/compute_judge_agreement.py`
- `tests/test_judge.py`, `tests/test_compute_judge_agreement.py`

## Edge cases

| Case | How I handle it |
|---|---|
| A rubric score comes back outside 1 to 3 | Fail loudly, same standard as every other validated field |
| A rubric score comes back as a boolean instead of a real integer | Explicitly rejected, since Python treats `True` as equal to `1` and that could otherwise sneak past the check |
| A human score gets saved as "3.0" instead of "3" | Accepted as long as it's a whole number, since this is just spreadsheet software reformatting a column, not a genuinely different value |
| A row is only partly scored by hand | Skipped in the agreement calculation and counted, not silently included or crashed on |
| Small sample size | Explicitly printed as a limitation every time the agreement script runs, not just mentioned once and forgotten |

## Real results
Rows with complete human scoring: 30
grounded: exact_match=0.73 within_1=0.90
factually_consistent: exact_match=0.80 within_1=0.80
tone_appropriate: exact_match=0.70 within_1=1.00
actionable: exact_match=0.70 within_1=0.97

## What actually happened

I made a real mistake setting this up the first time. My plan was to run
the judge, then score the same replies myself without seeing its
output, so the agreement number would actually mean something. But the
judge script prints every score to the terminal as it runs, and I read
that output before scoring anything myself. There was no honest way to
undo that once I'd seen it, so I generated a fresh batch of 30 examples
I hadn't looked at, built the blank scoring sheet before running the
judge on any of them this time, and did the real scoring on that clean
batch instead. The numbers above are from that clean batch, not the
contaminated one.

Two small bugs came up while building the validation for this. One was
the boolean issue mentioned above. The other was that when I opened my
scored spreadsheet back up, it had quietly turned my scores from "3"
into "3.0", which broke the strict integer check I'd originally written
and would have made every row look incomplete. Had to fix the parser to
accept a whole number written as a float string without accepting an
actual fraction like "2.5".
