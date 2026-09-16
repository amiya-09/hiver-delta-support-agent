# Phase 6 Failure Analysis: LLM-as-Judge

## Finding 1: The judge misses fabrication on the exact dimension built to catch it

I scored the same 30 replies myself, blind to what the judge said, and
compared. On three of the four rubric dimensions, we mostly agreed. On
`factually_consistent`, the one dimension I specifically wrote to catch
the model stating something unsupported, every disagreement between us
was a full swing from 1 to 3, never a soft 2-vs-3 difference like the
other three dimensions had. That pattern by itself told me something
was systematically wrong, not just noisy.

Here are the six actual cases, judge score and my score side by side.

Example 2157846: customer had a website/app/phone issue trying to
change a reservation. Judge scored 3, reasoning "acknowledges the
website/app issue... contains no fabricated details." My score was 1.
The reply claims "we're aware of the issue and are working to fix it,"
which nothing retrieved actually supports, that's an invented promise,
not a neutral acknowledgment.

Example 459167: customer asked about redeeming a companion
certificate. Judge scored 3: "directly references the phone number used
in the relevant precedent." My score was 1. The retrieved examples don't
actually support that specific phone number or the eligibility guidance
given, the model reused a number from a different context.

Example 396186: customer's husband was moved out of a paid
extra-legroom seat, asking about a refund. Judge scored 3: "directly
echoes a relevant precedent, contains no invented details." My score was
1. The reply states a flat "a refund or partial credit would not be an
option" as if it were settled policy, borrowed from an unrelated
seat-recline complaint, not this situation.

Example 2382720: customer denied boarding, charged $75, wants a
refund. Judge scored 3: "accurately reflects the same-day change policy
and fee details from the retrieved precedent." My score was 1. The
retrieved examples are about the standard $75 change fee policy, not
about whether this specific charge, tied to a denied-boarding dispute,
was applied correctly. The reply applies an unrelated policy to a
different kind of complaint.

Example 2523726: customer asked if they can carry on snowshoes.
Judge scored 3: "accurately draws on a relevant precedent about
carry-on size limits." My score was 1. The retrieved examples support a
general carry-on size conversation, not the specific "45 linear inches"
figure and under-seat requirement the reply states as fact, that
specific number isn't actually supported by anything retrieved.

Example 841378: customer charged for an unavailable seat, wants a
refund. Judge scored 3: "uses accurate phone number, maintains
professional tone." My score was 1. Same pattern as 459167 and 396186
combined, a flat no-refund claim plus a reused phone number, neither
actually supported by what was retrieved for this specific complaint.

Every one of these shares the same shape: the model states something
specific and concrete, a phone number, a policy outcome, an exact
measurement, that isn't actually true for this particular customer, and
the judge scores it as fully consistent anyway. It looks like the judge
checks whether a reply is topically in the neighborhood of something
retrieved, not whether its specific claims are actually supported. That
is exactly the blind spot I wrote the rubric instruction to catch, and
it missed all six.

## Finding 2: The sample size here is small

30 examples is enough to notice a real pattern, which is what happened,
but it's not enough to call any of these agreement numbers statistically
solid. I'm treating the pattern above as a real, well-evidenced finding
because I can point at six specific, readable, consistent examples, not
because the underlying sample is large.

## Finding 3: I contaminated my own first attempt at this

My original plan was to run the judge, then score the same replies
myself without seeing its output first, so the agreement number would
mean something. The judge script prints every score to the terminal as
it runs, and I read that output before scoring anything myself the
first time through. There was no honest way to undo that once I'd seen
it, so I built a completely fresh, non-overlapping batch of 30 examples,
created the blank scoring sheet before running the judge on any of them,
and did the real scoring on that clean batch instead. Everything above
is from the clean batch. This wasn't a system failure, it was a process
mistake on my part, but it's worth documenting since it's exactly the
kind of thing that could quietly invalidate a result if I hadn't caught
it.
