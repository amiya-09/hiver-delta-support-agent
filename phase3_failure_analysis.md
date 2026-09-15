# Phase 3 Failure Analysis: Intent Classification

Baseline results (before prompt fix): 75.83% overall accuracy (240 examples,
`openai/gpt-oss-20b` via Groq). Compared against a trivial baseline (20.0%,
always predict the mode category) and a keyword-rule baseline (59.6%) --
see `data/eval/phase3_baseline_results.json`.

Three real failure patterns emerged from reading the model's own stated
`reasoning` field for every misclassified example, not just the confusion
matrix counts.

## Failure mode 1: Surface form overrides topic specificity
**Affects ~14 of ~58 total misclassifications (~24%) -- the single
largest identified cause.**

Two category pairs show the identical mechanism: when a message's
*phrasing* matches one category's definition (a question → POLICY_QUESTION;
frustrated tone → SERVICE_COMPLAINT) while its *topic* matches a more
specific category (miles/loyalty → ACCOUNT_LOYALTY; refund/credit →
REFUND_COMPENSATION), the model defaults to the generic, form-matching
category over the specific, topic-matching one.

**Real examples (ACCOUNT_LOYALTY mislabelled as POLICY_QUESTION), with the
model's own reasoning:**
- *"asking about a SkyMiles program policy — the bonus miles awarded for
  donations"* → the model explicitly names the SkyMiles topic, then still
  picks the generic category because the message is phrased as a question.
- *"asking for clarification on how to use miles for an upgrade"* → same
  pattern.

**Real examples (REFUND_COMPENSATION mislabelled as SERVICE_COMPLAINT):**
- *"the voucher cannot be applied when purchasing a new ticket, indicating
  a service issue"* → again, the model names the voucher/refund topic
  directly, then defaults to the generic complaint category because of the
  frustrated tone.

**Hypothesis:** the prompt's category definitions don't state a precedence
rule for when phrasing and topic point to different categories. Confirmed
by the model's own reasoning text, not inferred.

**Fix attempted and evaluated:** added an explicit "topic specificity over
surface form" rule to the prompt (see decision log #24). Validated via a
targeted recheck (not a full 240-example rerun -- see decision log #22)
rather than guessing from the fix alone:
- **22 of 58** previously-misclassified examples (38%) are now correct.
- **2 of 30** examples from a random control sample of previously-*correct*
  examples regressed to incorrect (a 6.7% regression rate on the sampled
  control group).
- Net estimate, extrapolating the control-sample regression rate across
  all 182 originally-correct examples: approximately **192/240 (~79.9%)**,
  up from the original 75.83% -- a real, meaningful improvement, but not
  a clean win. The fix helped considerably more than it hurt, but it did
  measurably hurt something, and that should be stated plainly rather
  than only reporting the 22 fixed cases. The 2 regressed control examples
  have not yet been individually reviewed to confirm the regression's
  own root cause (a natural next step, cheap to do since it requires no
  further API calls -- just reading the saved recheck JSON).

## Failure mode 2: Gratitude about a resolved disruption -- a genuine taxonomy ambiguity, not a model error
**Affects 4 misclassifications (FLIGHT_DISRUPTION mislabelled as
PRAISE_FEEDBACK).**

Initially hypothesized as a sarcasm-detection failure (the taxonomy's
sarcasm rule exists for a similar surface pattern). Reading the actual
reasoning disproved this -- these are not sarcastic. Real example: a
customer thanking staff for quickly rebooking them after a delay, with no
remaining complaint or ask.

**This is not a model bug.** The labelling guide's "mixed intent → pick
the primary ask" rule doesn't cover messages with *no* ask at all, only
gratitude about how a past problem was resolved. Whether the "true"
category should be the underlying event (FLIGHT_DISRUPTION) or the
expressed sentiment (PRAISE_FEEDBACK) is a genuine, reasonable disagreement
that a human labeller could also make either way -- not something a
prompt fix should paper over. Documented as a known taxonomy boundary
case rather than "fixed."

## Failure mode 3: The fix itself introduced a new, narrower failure mode

Both regressions found in the targeted recheck's 30-example control
sample reveal a real cost of the topic-specificity fix (failure mode 1),
not just its benefit.

**3a. Triggering on mere presence of a term, not genuine "core subject"
status.** Example: *"Do I not have to pay for checked baggage if I have
SkyTeam Elite status?"* (true: `POLICY_QUESTION`, correctly classified
before the fix, now flips to `ACCOUNT_LOYALTY`). The new rule says "if
the message's *core subject* is ... loyalty program mechanics, use
ACCOUNT_LOYALTY" -- but here, elite status is only a qualifying condition
for a baggage-fee policy question, not the subject of the question
itself. The model appears to trigger on the mere presence of a
loyalty-program term rather than genuinely assessing whether it's the
core subject, despite the rule's explicit wording asking for exactly
that distinction.

**3b. Apparent generalization of the rule's underlying principle beyond
its two stated cases.** Example: *"me and my son were just dragged off
the airline. No one recorded. I saw a security punch and I'm mad."*
(true: `SERVICE_COMPLAINT` -- a genuine safety/security incident,
correctly classified before the fix, now flips to `FLIGHT_DISRUPTION`).
This message has no textual connection to either of the rule's two
specific triggers (loyalty/miles, refund/credit). Working hypothesis:
the rule's underlying *principle* -- "match by topic, not surface
phrasing" -- may have generalized further than the two cases it was
written for, causing the word "airline" to be read as a topic signal
toward `FLIGHT_DISRUPTION` even though the actual content is a
security-incident complaint. This is speculative but plausible, and
worth flagging as a real risk of prompt-based fixes generally: a rule
written to correct two specific category confusions can shift model
behavior in unrelated categories through the general principle it
teaches, not just its literal, named content.

**Net effect:** the fix is still a real improvement (22 fixed vs. 2
regressed in the sampled groups, an estimated ~+4 points overall -- see
above), but it is not a clean, isolated win. This is good material for
the "misleading headline number" section: a single accuracy delta can
hide the fact that a fix's mechanism has side effects beyond its
intended, named scope.

## What this means for the headline number
The 75.83% figure is not purely a measure of model capability -- it
partly reflects an unresolved ambiguity in our own taxonomy (failure
mode 2) and a fixable, evidence-backed prompt gap (failure mode 1). A
reader should not treat 75.83% as "how good the model is" in isolation;
see the report's dedicated "what is misleading about my headline number"
section for the full discussion.
