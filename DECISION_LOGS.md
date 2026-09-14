Decision 5 (bundled): LLM API and embedding model

Recommendation: Google Gemini API (via Google AI Studio)

Here's the reasoning. As of now, there are a handful of genuine no-card-required free tiers — Google Gemini, Groq, OpenRouter, Mistral, Cloudflare Workers AI, and a few others. Google Gemini API offers free usage on current Flash models with no card required, and eight providers currently offer a genuine standing free tier as of a recent check. 
costbench

The reason I'd pick Gemini specifically over the others: it gives you both a chat model AND an embedding model under the same free API key. That matters a lot for us, because our architecture needs two different capabilities — a chat model for classify/generate/route/judge, and an embedding model for retrieval. If we used Groq (great for chat, very fast) we'd still need a second provider just for embeddings, since Groq doesn't offer them. Gemini gives us both in one place, which keeps our whole LLM-wrapper layer (Section 7 of the architecture) to one provider, one API key, one rate-limit to reason about — simpler, which is exactly our stated goal.

Simplest alternative: Groq (fastest inference, generous free limits) + a separate free embedding source (e.g., a local open-source embedding model via sentence-transformers, which runs on your own machine, no API needed at all).

Trade-off: Groq is faster per call and has no risk of embedding-quota exhaustion. But running a local embedding model adds a real dependency (downloading model weights, needing enough RAM) and two providers to configure instead of one. Given you're on a laptop-style setup and want reproducibility to be dead simple for a grader, I'd lean Gemini-only — but if Gemini's rate limits turn out too tight once we're running real numbers, local embeddings is the easy fallback (it costs nothing and has no rate limit at all).




Decision 2: Subsample size

This depends directly on the free-tier budget, so let's do the math together instead of picking an arbitrary number.

Per message that goes through our pipeline, we make roughly 4 LLM calls: classify, generate reply, route decision, judge score. (Retrieval doesn't call the chat model — just the embedding model, which is much cheaper/faster.)

If we process, say, 400 messages end-to-end (this comfortably covers our 150-250 golden set plus room for the two baselines to run against the same set, plus a little headroom for retries), that's about 1,600 chat calls total — very achievable on a free tier across a few days, especially since caching (Section 7) means we only ever pay for each unique input once, even if we rerun the pipeline many times while debugging.

Simplest alternative: a much smaller subsample, e.g., 100-150 messages total — just enough to barely cover the golden set minimum.

Trade-off: Fewer calls, faster to build, but it leaves us almost no room for the retrieval index itself (retrieval needs a good-sized pool of historical resolved cases to search over — separate from the golden set, which is our test set). The retrieval index should probably be a few thousand cleaned historical pairs (that's just embedding calls, which are cheap/fast, not chat calls) — the 400-message number above is specifically about the more expensive golden-set + evaluation side.

So, concretely: retrieval index ≈ 2,000-3,000 historical pairs (embeddings, cheap) · golden set = 150-250 hand-labelled (chat calls, the expensive part) · total pipeline test run ≈ 400 messages including baselines. That fits a one-week timeline and a free API comfortably.




Decision 1: Delta

Why: strong volume (42k replies, 36k unique conversations — plenty for both a retrieval pool and golden set), genuine multi-turn resolution content (not just DM-redirects), and a domain with naturally distinguishable risk tiers (routine info request vs. safety/financial/time-critical issue) — which is exactly what the routing/escalation requirement needs to be a real design exercise rather than a rubber-stamp.

Simplest alternative: AppleSupport — cleaner text, zero DM-redirects, slightly easier to build a taxonomy around.

Trade-off: Apple would be a bit easier to build (more uniform structure, cleaner single-turn pairs), but it gives us a weaker story for the routing/escalation logic — one of the three core required behaviors — since most Apple issues don't naturally separate into "auto-handle" vs. "needs a human," while Delta's do. Since escalation quality is explicitly graded and a real differentiator, I'd trade a bit of build-ease for a richer problem.

Close alternates if you want a similar profile: AmericanAir or SouthwestAir — same airline dynamics, similar volume tier, would work just as well if you specifically prefer one (e.g., you already know it better, or one has cleaner sample threads when you look).

Decision 3: intent taxonomy: what small set of categories we'll classify Delta customer messages into. This one specifically has to be built from real data, not invented from imagination — the assignment explicitly says "a small set of intents that you define from the data." So step one is getting a decent-sized batch of real Delta customer messages to read through.

Why we need more than the 4 samples we already have

4 threads isn't enough to spot real patterns — we'd just be guessing categories from a handful of anecdotes. To do this properly (what's sometimes called "open coding" — reading real examples and noticing recurring themes before naming categories), we want maybe 80-100 real Delta customer messages, ideally spread across different times/days so we're not just seeing one bad-weather-day's worth of delay complaints.

Simplest alternative: just look at 10-15 examples and guess a taxonomy from that.

Trade-off: faster, but risky — with such a small sample we might miss an entire common category (e.g., missing bag claims) or invent a category that's actually rare, and we'd only discover the mistake later when labelling the golden set. Since the taxonomy affects almost everything downstream (golden set schema, classifier, routing logic), it's worth getting right up front with a proper look at the data.

I read through all 100 real messages. A few clear patterns jump out — and one genuinely useful data-quality issue worth flagging before we lock the taxonomy.

First, an honest finding: not every message here is a "fresh complaint"

A handful of entries are clearly mid-conversation fragments, not new complaints — things like:

@Delta 276 · @Delta F8tpy7 · @Delta That's correct. · @Delta Thanks - the issue is that I still haven't received either of my checked bags...

These look like a customer replying within an ongoing thread (providing a confirmation number, answering a question Delta asked) rather than opening a new issue. Our thread-reconstruction script tries to walk back to the true first message, but it looks like it sometimes stops early or the chain is broken in the data. This is a real thing to document in the decision log: our root-finding heuristic occasionally captures a mid-thread reply instead of the true opening message, and we'll need a simple filter (e.g., minimum content/length, or "does this message contain a self-contained complaint") to catch these before they either pollute the taxonomy or confuse the golden set. This connects directly back to the cleaning step.

The real recurring patterns (this is the actual open-coding pass)

Reading through the rest, here's what genuinely repeats:

Flight disruption — delays, ground stops, cancellations, missed/tight connections. "still on the ground" · "flight delay and no one has a clue" · "stranded in southern China"
Baggage problems — lost, damaged, delayed bags, wheelchair broken twice. "they cannot find our bags" · "broke the new one I paid 3500 to get it fix"
Booking / reservation changes — seat changes, rebooking, name corrections, trip modifications. "trying to change an assigned seat" · "misspelled name on my reservation"
Refund / compensation requests — money back, credits, fee waivers. "will my credit card be refunded automatically?" · "zero compensation to your customers"
Account / loyalty program (SkyMiles/Medallion) — status questions, miles disputes, account changes. "Just lost 35K miles on an award ticket" · "change your name on your Skymiles account"
General policy / how-to questions — not tied to an active problem, just informational. "Can a small duffle bag count as a personal item?" · "what's your policy on paintball markers"
Service complaint / feedback (non-specific) — venting, no clear fixable ask, including the more serious ones. "over a 2hr wait" · the discrimination allegation about a flight attendant. This one deliberately includes both mundane gripes and serious ones — more on that below.
Praise / positive feedback — pure compliments, thank-yous. "Huge thanks to Sebastian at the call center"

Why I lumped "rude staff" and "discrimination allegation" into the same category as ordinary complaints, instead of giving discrimination its own category: it's tempting to split it out because it feels more serious, but on 100 messages I only saw it once — one category for a single occurrence would be a rule built for an anecdote, not a pattern. The better fix is: keep it inside "Service complaint / feedback," but let the escalation logic (next decision, not the taxonomy) treat certain keywords/severity signals within that category as an automatic hard-escalate. That keeps the taxonomy clean while still making sure that specific message type never gets auto-handled.

Why "flight disruption" and "baggage" are separate categories instead of one "travel problem" bucket: they need genuinely different information to resolve (flight status/rebooking options vs. baggage tracking/claims process), so collapsing them would make the "draft a reply grounded in precedent" step less useful — retrieval would be searching a muddier pool.

What I'd leave out of the taxonomy on purpose

The short fragments/continuations (the "276", "F8tpy7" type messages) shouldn't get their own intent category — they're a data-quality/cleaning problem, not a real customer intent. They'll get filtered out upstream instead (tie-in to the cleaning-step fix above).

Here's the proposed taxonomy as a clean list:

#	Intent	One-line definition
1	Flight Disruption	Delay, cancellation, ground stop, missed/tight connection
2	Baggage Problem	Lost, damaged, delayed, or mishandled baggage
3	Booking/Reservation Change	Seat change, rebooking, name correction, trip modification
4	Refund/Compensation Request	Money back, credit, fee waiver
5	Account/Loyalty Program	SkyMiles/Medallion status, miles, account changes
6	General Policy/How-To Question	Informational, no active problem
7	Service Complaint/Feedback	Dissatisfaction, no specific fixable ask (includes severity flags)
8	Praise/Positive Feedback	Compliments, thank-yous

Let's design this the same way we did the taxonomy — starting from real signals we already have, not an abstract policy invented in a vacuum.

### First: what signals do we actually have available to make this decision?

By the time the escalation step runs, we already have:
1. **The predicted intent category** (from the taxonomy we just locked in)
2. **How confident the retrieval step was** — did it find a genuinely similar past case, or nothing close?
3. **How confident the classifier was** — was the intent obvious, or a toss-up between two categories?
4. **The raw message text itself** — which might contain specific danger signals (safety, discrimination, injury, legal threats)

The escalation policy should be built *only* from these — nothing exotic, nothing requiring extra infrastructure.

### The core design principle: escalation should be a short ordered checklist, not one opaque score

**Recommendation:** a small, ordered set of rules, checked top to bottom. The first rule that matches decides the outcome, and the rule's name *is* the reason.

**Why:** this directly satisfies "decide... with a stated reason" in a way that's actually true, not just plausible-sounding. If the policy were a single machine-learned "escalate probability" score, the "reason" would have to be reverse-engineered/guessed after the fact — which risks being unconvincing or even dishonest (writing a reason that sounds right but isn't actually what drove the decision — we flagged this exact trap back in the logging discussion). With ordered rules, the reason is just: *which rule fired.*

**Simplest alternative:** train a classifier to predict escalate/don't-escalate from features, using our golden set as training data.

**Trade-off:** there's actually a real problem with this beyond just complexity — it's circular. We don't have any escalation labels *until* we build the golden set ourselves, and the golden set is also supposed to be our **test** set for evaluating the policy. If we also used it to train the policy, we'd be grading our own homework with the same paper we studied from. Rules avoid that entirely: we can design them from reasoning about the data now, then use the *whole* golden set purely for evaluation, honestly.

### The actual checklist, built from what we saw in the real Delta sample

**Step 1 — Hard-escalate triggers (checked first, always win)**

These are messages where getting it wrong isn't just "bad reply," it's actually risky — a bot should never touch these. From the real sample:

> *"perpetuating airline racism"* (discrimination allegation)
> *"risking my family safety at night"* (safety concern)
> *"they cannot find our bags"* + medical equipment: *"lost my son's wheelchair"* (serious loss/medical-adjacent)

**Rule:** if the message contains specific high-risk signals — discrimination/harassment language, safety/injury language, legal-sounding language, or mentions of medical equipment/minors — escalate immediately, regardless of anything else.
**Stated reason example:** *"High-risk keyword match: discrimination-related language detected."*

**Step 2 — Category defaults (only checked if Step 1 didn't fire)**

Not all intent categories carry equal risk. From what we saw:
- **Baggage Problem** and **Refund/Compensation Request** involve real money or real physical loss — getting these wrong costs the company money or the customer trust.
- **General Policy/How-To Question** and **Praise/Positive Feedback** are low-stakes by nature — worst case, a slightly-off FAQ answer, easily correctable.

**Rule:** Baggage Problem and Refund/Compensation Request default to *lean escalate* unless Step 3 clears them; General Policy/How-To and Praise default to *lean auto-handle*.
**Stated reason example:** *"Category defaults to human review: financial/compensation request."*

**Step 3 — Grounding confidence (the "do we actually know how to answer this" check)**

This is the one that ties directly back into our retrieval step. If retrieval couldn't find a genuinely similar past resolved case, we have no real precedent to draft from — auto-handling here would mean the reply is just improvised, which defeats the entire "grounded" requirement.

**Rule:** if the top retrieved similarity score is below a threshold, escalate — *regardless of category* — because we don't have grounding to trust an auto-reply on.
**Stated reason example:** *"No sufficiently similar historical precedent found (best match similarity 0.34)."*

*(We'll pick the actual threshold number once we have real similarity scores to look at — more below.)*

**Step 4 — Classification confidence**

**Rule:** if the classifier was genuinely torn between two categories (e.g., a message that reads like both a booking-change request and a refund request), escalate rather than guess.
**Stated reason example:** *"Intent classification ambiguous (top two categories nearly tied)."*

**Step 5 — Otherwise, auto-handle**

If nothing above fired, we have: a low-risk category (or a risk-category case with strong precedent), good retrieval grounding, and confident classification. Auto-handle.
**Stated reason example:** *"Routine case with strong historical precedent (similarity 0.81) and confident classification."*

### Why the order matters (this is worth being deliberate about)

Notice safety comes *before* everything else, including confidence. A safety-flagged message with a perfectly confident classification and strong retrieval match should still escalate — confidence in *how* to answer doesn't mean it's *appropriate* for a bot to answer. That's a real, explainable design choice for the decision log: **grounding confidence tells us if we know what to say; risk level tells us if we should be the one saying it. These are different questions, and risk always wins.**

### One honest limitation, worth stating upfront rather than discovering later

The "hard-escalate keyword" list (Step 1) is inherently incomplete — we're defining it from 100 sample messages, so it'll miss phrasings we haven't seen yet. That's fine, and it's exactly the kind of thing that belongs in the "misleading headline number" section later: if our escalation-recall number looks great, we should be honest that it's partly because our test set was drawn from the same data patterns we used to *design* the keyword list — of course it recognizes what it was built to recognize.

### On thresholds specifically (Step 3 and 4)

**Recommendation:** don't hand-pick threshold numbers right now from theory. Instead, leave them as clearly-named config values, run the pipeline against a slice of the golden set once it exists, look at the actual distribution of similarity/confidence scores for cases we know should vs. shouldn't escalate, and pick thresholds that separate them well — documenting *why* that number, not just the number.

**Simplest alternative:** just pick round numbers now (e.g., similarity < 0.5 → escalate) and move on.

**Trade-off:** picking now is faster, but arbitrary — we have no evidence 0.5 is meaningfully different from 0.4 or 0.6 for this data. Since we're going to be looking at real similarity scores anyway once retrieval is built, it costs us almost nothing to wait and set the threshold from actual evidence instead of a guess — and "we calibrated this threshold against observed score distributions" is a much stronger interview answer than "we picked 0.5 because it's the middle."

---

**Summary of the policy as five ordered checks:**

| Order | Check | If triggered → | Example reason string |
|---|---|---|---|
| 1 | High-risk keywords (safety/discrimination/legal/medical) | Escalate | "High-risk keyword match: safety concern." |
| 2 | Risk category (Baggage, Refund/Compensation) | Lean escalate | "Category defaults to human review: baggage claim." |
| 3 | Weak retrieval grounding | Escalate | "No sufficiently similar precedent found (sim=0.34)." |
| 4 | Ambiguous classification | Escalate | "Intent classification ambiguous." |
| 5 | None of the above | Auto-handle | "Routine case, strong precedent (sim=0.81)." |

Let's close out these last two — and since they depend on everything we've locked in so far (the 8-intent taxonomy, the 5-step escalation policy), the schema basically falls out of decisions we've already made rather than being invented fresh.

## Golden-set label schema

### The core question this schema has to answer

For every one of our 150-250 examples, a grader (or future-us) should be able to look at one row and know: *what should the system have done, and why, specifically enough to check whether our system got it right for the right reason* — not just right by coincidence.

Here's the schema, field by field, with the reasoning for each:

| Field | Type | Why it exists |
|---|---|---|
| `example_id` | string | Stable reference — reuse the original `tweet_id` so we can always trace back to raw data. |
| `raw_text` | string | The original, unmodified tweet. We label against the real text, not a cleaned version — cleaning is for the model pipeline, not for us as labellers. |
| `source_tweet_id` | string | Same as `example_id` in practice, but explicit — supports the "how we sampled" audit trail. |
| `true_intent` | one of our 8 categories | The core classification ground truth. |
| `intent_ambiguous` | boolean | **This one's easy to skip and shouldn't be.** Some real messages genuinely sit between two categories (we saw this ourselves reading the sample). Flagging it explicitly means we can later check: did our system's own classification-confidence signal (Step 4 of the escalation policy) actually catch the cases *we* found ambiguous? That's a much sharper test than just checking raw intent accuracy. |
| `true_escalate` | boolean | The ground-truth routing decision. |
| `true_escalation_rule` | one of: `hard_risk_keyword` / `risk_category_default` / `weak_grounding` / `ambiguous_classification` / `none_auto_handle` | **This is the field that makes the escalation evaluation actually rigorous.** A binary "should escalate: yes/no" only tells us if the final answer matched. This field lets us check something much more meaningful: did the system escalate *for the right reason*? A system that escalates a baggage complaint because it (correctly) matched the risk-category rule is behaving as designed; a system that escalates the same message because retrieval happened to fail is getting the right answer by accident. Without this field, we can't tell those apart — and that distinction is exactly the kind of thing that makes for a strong failure-analysis finding later. |
| `reply_acceptability_notes` | short freeform text | Support replies don't have one single "correct" answer, so instead of trying to write out an ideal reply per example (a huge, low-value effort), we write a short note on what a good reply *must* contain or *must* avoid — e.g., "must not promise a refund amount, should acknowledge the delay specifically." This becomes our reference when eyeballing judge scores for sanity later. |
| `sampling_stratum` | string | Which bucket (intent × risk-tier) this example was deliberately drawn from — this is what makes the "note on how you sampled" requirement checkable rather than just asserted. |
| `labeller_notes` | optional freeform | Anywhere the label wasn't obvious — "borderline between booking-change and refund, went with booking-change because no money was mentioned." This is what protects us if we (or a grader) come back to a weird-looking label six weeks later and wonder why we made that call. |

### One process decision worth being upfront about

Since you're labelling this solo, there's a real risk of **drifting** — labelling example #180 slightly differently than you would have labelled example #10, just because your sense of the categories shifted as you saw more data. The fix costs almost nothing: before labelling starts, write a **one-page labelling guide** (just the 8 intent definitions + the 5 escalation rules, each with 1-2 concrete examples from the real data we've already looked at) and label against that document, not from memory. It's also exactly the kind of artifact a grader would find credible as evidence of process, and it directly satisfies "a short note on how you sampled and labelled them."

**Simplest alternative:** skip the written guide, just label from your own judgment as you go.

**Trade-off:** faster to start, but with 200 examples labelled over several sessions across a week, definitions creep is a real risk — and it directly undermines your own golden set's reliability, which everything else depends on. The guide takes maybe 20 minutes to write and pays for itself immediately.

---

## Subsample and volume numbers

### Golden set size: **200** (within the 150-250 range, toward the higher end)

Why not just 150 (the minimum)? Once we stratify across 8 categories *and* deliberately oversample the rare-but-important risk cases (below), 150 starts to feel thin per bucket. 200 gives us breathing room without meaningfully increasing labelling effort (50 extra examples is maybe an extra hour of work).

### Stratification — and an honest oversampling call

If we sampled purely randomly, our own 100-message read-through suggests categories like "Service complaint/feedback" and "Praise" would dominate, while genuinely important categories like refunds or hard-risk-keyword cases would be rare or nearly absent — exactly the trap we flagged when designing the taxonomy sampling approach.

**Recommendation:**
- **Minimum floor of 20 examples per intent category** (8 × 20 = 160), regardless of natural frequency.
- **Remaining 40** allocated toward the operationally highest-stakes categories (Flight Disruption, Baggage, Refund/Compensation), since these matter most for the escalation story.
- **A deliberate, separately-sourced set of ≥15 examples containing hard-escalate signals** (Step 1 of the policy — safety/discrimination/legal/medical language). Based on our 100-message sample, these occur naturally at roughly a 2-3% rate — meaning a random sample of 200 would net us maybe 4-6 such cases, not nearly enough to evaluate that rule with any confidence. So instead: run a **targeted keyword search across the full Delta dataset** (not just our sample) for this specific purpose, pull real matches, and fold them into the golden set.

**This needs to be stated honestly, not hidden:** these ≥15 examples are *not* randomly sampled — they're deliberately, purposefully oversampled because the rule they test for is rare but important. That's a completely legitimate thing to do (and worth putting in the decision log exactly like that), but it would be dishonest to describe the golden set as "randomly sampled" without flagging this specific exception.

**Simplest alternative:** pure random sampling, no floor, no deliberate oversampling.

**Trade-off:** simpler to describe and defend as unbiased, but our escalation-rule accuracy numbers for the rare/high-stakes rule would be based on so few examples they'd be statistically meaningless — and that specific rule (catching genuinely dangerous cases) is arguably the most important one to prove works. Given "reliable" and "first-class evaluation" are explicit priorities, the extra targeting effort is worth it.

### Judge-human agreement check: **40 examples**

Separate from the golden set itself — this is us personally re-scoring a subset of the *judge's* outputs (after the full pipeline has run) on the same rubric, to measure judge-human agreement. 40 is enough to compute a meaningful agreement statistic without becoming its own multi-day task.

### Retrieval/precedent pool: **~2,500 historical resolved pairs**

This is separate from the golden set — it's the pool the retrieval step searches over at runtime, built from real Delta customer→resolution pairs (with DM-redirect-only replies already filtered out, per our cleaning decision). 2,500 gives reasonable density across all 8 categories without needing a large embedding budget — these are cheap embedding calls, not chat calls, so we have headroom to go bigger later if retrieval quality looks thin.

### The actual call budget — checking this against "free" reality

This is worth being concrete about, since "free API" was our hard constraint. Here's where the calls actually go:

| What | Calls per example | × examples | Subtotal |
|---|---|---|---|
| Our system: classify | 1 | 200 | 200 |
| Our system: generate reply | 1 | 200 | 200 |
| Our system: judge score | 1 | 200 | 200 |
| Simple baseline: classify + generate (no retrieval) | 2 | 200 | 400 |
| Simple baseline: judge score | 1 | 200 | 200 |
| Trivial baseline: judge score only (canned reply, majority-class intent — no generation call needed) | 1 | 200 | 200 |
| **Total chat/generation calls** | | | **≈ 1,400** |
| Retrieval index build (embeddings) | 1 | 2,500 | 2,500 |
| Retrieval queries at eval time (our system only) | 1 | 200 | 200 |
| **Total embedding calls** | | | **≈ 2,700** |

Notice **escalation itself costs zero LLM calls** — it's pure rule logic running on signals we already computed (classifier confidence, retrieval similarity, keyword match), which is one of the nice side-benefits of designing it as a checklist rather than another model call.

~1,400 chat calls + ~2,700 cheap embedding calls, spread across a week, with caching meaning we only ever pay for each unique input once even while debugging — this comfortably fits a free tier. If it turns out tighter than expected once we're running real numbers, the retrieval pool size (2,500) is the easiest one to shrink without hurting the core deliverables.
