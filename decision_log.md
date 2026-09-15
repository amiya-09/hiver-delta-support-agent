# Decision Log

Non-obvious decisions made during this project, and why. Chronological by
phase. This satisfies the assignment's required decision log; see the
report for the full write-up of headline results and failure analysis.

## Brand & scope selection

1. **Chose Delta over AppleSupport, AmazonHelp, and telecom brands.**
   AmazonHelp had the highest volume but spans unrelated verticals and is
   multilingual. Telecom/cable brands (comcastcares, TMobileHelp) were
   dominated by DM-redirects with almost no real resolution content.
   AppleSupport had the cleanest single-turn replies but almost no
   variation in stakes, making the escalation-policy design trivial.
   Delta had strong volume (42,253 replies), genuine multi-turn resolution
   content, and a natural spread of risk tiers (routine question vs.
   safety/financial/time-critical) -- the last point specifically matters
   because escalation logic is one of the three required agent behaviors.

2. **Thread pairs are adjacent (customer message, immediate brand reply)
   pairs, not whole-thread walks to a single "final" resolution.** This
   captures resolution *steps*, including follow-up exchanges, and is
   simpler to reason about than trying to identify one canonical
   "resolution" per thread.

3. **Tagged `is_thread_root` on every customer message** (true if the
   message itself isn't a reply to something else). Discovered that ~30%
   of raw "customer messages" in reconstructed pairs are actually
   mid-thread fragments (e.g. "@Delta 276", a confirmation number sent
   mid-conversation), not genuine opening complaints. Root-only messages
   are used for intent classification and golden-set sampling; non-root
   pairs are still usable for retrieval grounding.

## Data cleaning

4. **Cleaning is deliberately light**: strip mentions/links for
   matching, drop near-duplicate complaints, drop DM-redirect replies
   from the retrieval *grounding* pool -- but no spelling/grammar
   correction, no sarcasm detection, no non-English translation. The
   assignment explicitly frames the messiness of the data as the point;
   over-cleaning would mean not actually solving the stated problem.

5. **DM-redirect rate corrected from an early ~45% estimate to a
   measured 18.4%.** The 45% figure came from a tiny (93-row), multi-brand
   sample that happened to include DM-heavy telecom accounts. The real,
   full-dataset, Delta-specific rate (6,331 / 34,435 kept pairs) is 18.4%
   -- lower, and part of why Delta was a reasonable choice relative to the
   telecom brands.

## Intent taxonomy & escalation policy

6. **8-category intent taxonomy derived from open-coding 100 real Delta
   messages**, not borrowed from an existing labelled dataset (e.g.
   Banking77, which is a different domain entirely and was only ever
   suggested as optional help for intent *methodology*).

7. **Escalation logic is a fixed, ordered 5-rule checklist, not a
   trained classifier.** Two reasons: (a) there is no escalation-labelled
   data to train on until the golden set exists, and using the golden set
   for both training and evaluation would be circular; (b) a rule-based
   checklist makes the required "stated reason" literally true (the
   reason IS which rule fired), rather than reverse-engineered after the
   fact from an opaque score.

8. **Safety/risk keyword checks run before grounding-confidence or
   classification-confidence checks in the escalation order.** Being
   confident about *how* to answer is a different question from whether a
   bot *should* be the one answering -- a well-grounded, confidently
   classified message can still need a human if it's safety- or
   discrimination-related.

## Embeddings & retrieval

9. **Initially chose Google's Gemini API for both chat and embeddings**
   (one provider, one API key, simpler than mixing providers) -- later
   partially reversed (see #10).

10. **Switched the embedding backend from Gemini's API to a local,
    open-source model (`intfloat/e5-small-v2` via sentence-transformers)
    after hitting Gemini's free-tier DAILY quota (1000 requests/day)
    partway through building the 2,500-pair retrieval pool.** Waiting out
    the daily reset would have cost 2-3 calendar days against a 1-week
    deadline for what should be a one-time prep step. The design intent
    (asymmetric query/document embeddings) was preserved -- e5 models use
    text prefixes ("query: " / "passage: ") instead of Gemini's `task_type`
    parameter, but the underlying idea is identical.

11. **Retrieval uses brute-force NumPy cosine similarity, not a vector
    database or FAISS.** At 2,500 vectors, a matrix multiply is fast
    enough; a vector DB would add setup complexity and a dependency with
    no real benefit at this scale.

12. **Found and fixed a keyword-matching bug where "sue" (meant to catch
    legal threats) matched as a substring of the ordinary word "issue."**
    This silently made ~30 of the first 40 sampled "risk-signal" golden-set
    candidates false positives. Fixed by switching ALL keyword matching
    (not just the one affected keyword) from substring matching to
    word-boundary regex, since the same failure mode could affect any
    keyword list.

13. **Found intermittent NaN/Inf corruption in retrieval's similarity
    computation, traced to Apple's Accelerate BLAS backend on macOS** (a
    known, maintainer-acknowledged NumPy issue, not a data or code bug).
    Confirmed it was inconsistent -- reproducible with the original real
    query, not reproducible across a 100-query stress test afterward.
    Rather than fully eliminating Accelerate (which would require a
    from-source NumPy rebuild against OpenBLAS), added a permanent
    finite-score guard in `top_k_similar` that raises loudly if it ever
    recurs, given the bug's inconsistency and the project's time
    constraints.

## Golden set

14. **Golden-set sampling is stratified and deliberately oversampled, not
    purely random**: a per-category floor (so rare categories aren't
    starved) plus a separately-sourced, keyword-matched batch of
    candidates for hard-escalate risk signals (since these occur at only
    a low natural rate and a random sample wouldn't yield enough to
    meaningfully evaluate that rule).

15. **All 240 labels were made by hand, with no AI assistance on the
    actual judgment calls** (what intent a message represents, whether it
    should escalate, why). Using an AI to help label would make the
    golden set circular -- it exists specifically to be an independent
    check on the AI system, and checking an AI against another AI's
    judgment isn't evaluation.

16. **Hand-labelled category distribution diverged substantially from
    the keyword-heuristic sampling buckets** -- e.g. `BOOKING_CHANGE` was
    keyword-sampled at 20 but only 13 real messages were actually that
    intent on close reading, while `PRAISE_FEEDBACK` was sampled at 20 but
    48 real messages turned out to belong there. This confirms the
    keyword sampler was correctly treated as a rough sort only, not
    ground truth.

17. **The golden set ended up with only 1 example each for the
    `weak_grounding` and `ambiguous_classification` escalation rules.**
    This is a real, acknowledged evaluation-coverage gap, not something to
    paper over -- any reported accuracy on those two specific rules should
    be read as anecdotal, not statistically meaningful.

18. **Added explicit labelling-guide rules for two edge cases discovered
    while reading real sampled data**: satirical/troll tweets (label
    normally, flag in `labeller_notes` rather than inventing a 9th
    category) and at least one message from someone claiming to be a
    fired *employee*, not a customer (labelled `SERVICE_COMPLAINT` with
    `intent_ambiguous=true`, since the taxonomy was built for customer
    support and this falls outside that scope by design).

## Phase 3 — Intent classification

19. **Chose Groq (`openai/gpt-oss-20b`) for the classification/generation
    model, kept Gemini for embeddings.** Gemini's `gemini-2.5-flash`
    hit an unusually low daily quota (20 requests/day) specific to this
    project/key -- not the general published free-tier limit. Rather than
    wait on an unexplained per-project restriction, switched only the
    chat/classification backend to Groq, which had no such restriction for
    this key. Embeddings stayed on the already-working local model since
    they were unaffected.

20. **Discovered Groq's free tier enforces a *tokens-per-day* (TPD)
    rolling limit (200,000/day for `openai/gpt-oss-20b`), not a
    request-count limit** -- a meaningfully different constraint than
    every quota hit so far (Gemini's were always request-count based).
    Because our classification prompt is verbose (8 full category
    definitions plus several rule paragraphs, resent in full on every
    call), a single full 240-example evaluation pass consumes a large
    fraction of one day's token budget. This directly motivated decision
    #22 below rather than treating every prompt iteration as "just rerun
    the eval."

21. **Fixed a retry-logic bug where any exception was treated as a
    retryable rate limit**, including a 404 "model not found" error --
    discovered when a real run wasted 4 minutes retrying an error that
    retrying could never fix. Fixed by only retrying on the SDK's specific
    `RateLimitError` type; any other exception now fails immediately with
    a message that doesn't misleadingly claim "rate limited."

22. **Built a targeted-recheck tool instead of re-running the full
    golden set for every prompt iteration.** Given the TPD constraint
    (#20), validating whether a specific prompt fix worked doesn't
    require re-classifying all 240 examples -- only the examples it was
    meant to fix, plus a random control sample to check for regressions
    elsewhere. This cuts the cost of testing one hypothesis from ~240
    calls to ~50.

23. **Normalized a model-specific JSON quirk**: `openai/gpt-oss-20b`
    sometimes returns the *string* `"null"` (or `"none"`, case-insensitive)
    instead of JSON's real `null` literal for the optional `second_choice`
    field. Rather than treating this as a hard validation failure,
    normalized it to Python `None` before validation and before caching --
    the model's intent was clearly "no second choice," just expressed with
    the wrong token.

24. **Added an explicit "topic specificity over surface form" rule to the
    classification prompt**, based on evidence from the model's own stated
    reasoning on misclassified examples -- it would explicitly name the
    correct specific topic (e.g. "asking about a SkyMiles program policy")
    and *still* default to a more generic category based on phrasing
    (question form -> `POLICY_QUESTION`; frustrated tone ->
    `SERVICE_COMPLAINT`). This was a targeted, evidence-backed prompt
    change, not a speculative tweak.
