# Phase 4 Failure Analysis: Reply Generation

**Caveat up front:** this is based on a manual review of 10 generated
replies (`--sample-size 10`, seeded), not a formal metric -- there is no
automated quality score yet (that's Phase 6's LLM-judge rubric). Treat
these as qualitative, hypothesis-generating findings, not statistics.

## Finding 1: Retrieval returns a "similar enough"-looking result even
when nothing genuinely relevant exists

Across all 10 sampled queries, the top-3 retrieved examples' similarity
scores fell in a narrow, uniformly "fairly similar" band: 0.83 to 0.90.
Not one query returned a visibly low score, even for a message covering
an unusual issue (unaccompanied-minor fees, unauthenticated multi-topic
complaint). This means the generation prompt's instruction --
*"if none of the retrieved examples are genuinely relevant, return an
empty grounding list"* -- never had a genuinely low-scoring case to
correctly decline grounding on, in this sample. The instruction may be
sound; we simply haven't tested the branch where it should fire, because
retrieval doesn't appear to *produce* scores low enough to trigger it,
at least not reliably.

**Hypothesis:** with a small, general-purpose embedding model
(`e5-small-v2`) and a domain (customer support tweets) with a lot of
surface-level lexical/topical overlap between genuinely different issues,
cosine similarity scores may cluster in a narrow range regardless of true
relevance -- meaning the *absolute* similarity number is a weak signal for
"is this actually useful precedent," even though the *relative* ranking
(top result vs. bottom result) still worked well in every case we
reviewed in Phase 2.

## Finding 2: A concrete hallucination traced directly to Finding 1

Example `2052668`: customer complains about a specific paid Wi-Fi service
failure and asks for a refund. The generated reply copies, verbatim, a
phone number from a retrieved example about a *different* wifi-refund
complaint: `"You can gladly reach them at 1(877)350-0038 regarding your
request."` The model did not synthesize a general response -- it
transplanted a specific piece of contact information from one real case
onto an unrelated one, with a 0.88 similarity score (within the same
"looks fine" band as every other query in the sample) and a
`confidence_note` stating high confidence.

**Why this matters more than an ordinary wrong answer:** a support bot
confidently handing out a specific phone number that may not actually be
correct for this customer's situation is actively harmful, not just
unhelpful -- this is a real, concrete example of exactly the failure mode
"grounded, not improvised" was supposed to prevent, occurring specifically
*because* the prompt's escape hatch (empty grounding when not relevant)
never triggers if retrieval doesn't surface a low enough score to warrant
it.

## What worked well, for balance

Two of the ten examples (`2928719`, broken luggage; `2707115`, forced bag
check) show the mechanism working as intended -- the generated reply
reused specific, substantive procedural details (claim tag / baggage
service desk; weight-and-balance reasoning) from genuinely on-topic
retrieved precedent, not just matching tone.

## Proposed fix (not yet tested -- pending Groq daily token budget)

Add an explicit similarity-floor instruction to the generation prompt:
if the highest similarity score among retrieved examples is below a
stated threshold (e.g. 0.85), the model should treat this as weak
precedent and hedge accordingly (acknowledge without citing specifics),
even if it still lists the example in `grounding_source_indices`. This
targets the generation side of the problem directly, without requiring
changes to retrieval -- though the retrieval-side hypothesis (narrow
similarity-score range regardless of true relevance) is itself worth
investigating further with more sample queries once budget allows, since
Finding 1 may need its own fix independent of this one.
