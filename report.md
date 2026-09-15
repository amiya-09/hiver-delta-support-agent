# Report — Delta AI Support Agent

## Problem Framing

### What "good" means for this brand

Good, for a Delta customer-support agent, means four things, in this
order of priority:

1. **Correctly identifies what the customer actually needs**, even from
   noisy, real tweet text — measured against hand-labelled ground truth
   (the golden set), not just plausible-looking behavior on cherry-picked
   examples.
2. **Grounds its replies in genuine historical precedent**, not
   plausible-sounding improvisation. A reply that sounds helpful but
   invents a policy detail is worse than one that's honest about needing
   a human, because it erodes trust and can create real liability for the
   brand.
3. **Makes conservative, explainable escalation decisions.** For this
   brand specifically, the cost of these two error types is asymmetric:
   auto-handling a genuinely risky message (a discrimination complaint, a
   safety concern) is far more costly than escalating a routine one that
   didn't strictly need it. The system is deliberately biased toward the
   safer of the two mistakes.
4. **Is honest about its own limitations** — a system that reports one
   good-looking headline number while hiding known weak spots (thin
   evaluation coverage on rare rules, taxonomy ambiguities, prompt-level
   failure patterns) is not actually "good," even if the number is high.

### What we chose not to build

Deliberate scope cuts, and why:

- **Multi-language support.** Scoped to English-language tweets only.
  The dataset contains some non-English content; translating it correctly
  enough to trust for customer support is a separate, substantial problem
  we didn't have time to solve honestly.
- **Special handling for non-customer messages** (e.g. a message from
  someone claiming to be a fired employee, discovered during golden-set
  labelling). The taxonomy and escalation policy are built for
  customer-support interactions; out-of-scope message types are flagged
  as ambiguous rather than given first-class handling.
- **Dedicated satire/troll-tweet detection.** A few sampled messages
  during labelling were clearly not genuine complaints. Rather than
  building a classifier for this (rare, and hard to distinguish reliably
  from genuine hyperbole), these are documented as a known limitation.
- **An autonomous, multi-step planning agent** (e.g. a ReAct-style
  framework with tool-use discovery). The task has a fixed, known
  sequence of steps (classify → retrieve → generate → route) with no
  actual planning required, so a deterministic pipeline is simpler,
  more testable, and easier to explain than an agent framework.
- **A trained escalation classifier.** Escalation is a fixed, ordered
  rule checklist instead of a model trained on the golden set — training
  on the same data used for evaluation would make the evaluation
  circular, and a rule-based approach keeps the required "stated reason"
  literally true rather than reverse-engineered after the fact.
- **A production vector database.** At the retrieval pool's scale (2,500
  vectors), brute-force NumPy cosine similarity is fast enough; a vector
  DB would add infrastructure with no real benefit here.
- **Live handoff integration with a real ticketing/human-agent system.**
  The system produces a routing decision and a reason; actually
  connecting that to a live human-handoff workflow is out of scope.
- **Fine-tuning any model.** Both the classification/generation model and
  the embedding model are used off-the-shelf (prompted, not fine-tuned) —
  reasonable given the timeline and the fact that prompt-level fixes
  already demonstrated real, measurable improvement (see Failure Analysis).

---

## Results vs. Baselines
*[To be completed — Task 3]*

## Failure Analysis
*[To be completed — Task 4; see phase3_failure_analysis.md for the two
confirmed failure modes so far]*

## What Is Misleading About My Headline Number
*[To be completed — after baselines and failure analysis are finalized]*

## What I'd Do Next With One More Week
*[To be completed — after Phase 4 design planning, Task 5]*