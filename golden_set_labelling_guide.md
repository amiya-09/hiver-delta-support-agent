# Golden-Set Labelling Guide

Label every row in `candidates.csv` against THIS document, not memory or
gut feel in the moment — definitions drift over a 240-row session unless
anchored to something fixed. If a call still feels ambiguous after
checking this guide, that's what `intent_ambiguous` and `labeller_notes`
are for — don't force a clean answer where the data doesn't have one.

---

## The 8 intent categories

### 1. FLIGHT_DISRUPTION
Delay, cancellation, ground stop, missed/tight connection.
- Real example: *"Will flight 5330 depart anytime soon?? After two delays, we were supposed to depart at 1050. We're still on the ground."*
- NOT this category: a delay mentioned only in passing while the real ask is something else (e.g. "my flight was delayed AND I want a refund" → that's REFUND_COMPENSATION, with the delay as context — see "Mixed intent" below).

### 2. BAGGAGE_PROBLEM
Lost, damaged, delayed, or mishandled baggage (including mobility equipment like wheelchairs, since these are physically handled as baggage).
- Real example: *"Hours of delay, no communication and now they cannot find our bags."*
- Real example (equipment): *"losing my sons wheelchair and won't pay for it"*

### 3. BOOKING_CHANGE
Seat change, rebooking, name correction, trip modification.
- Real example: *"Trying to change an assigned seat on a flight today and unable to do so. Can you help?"*

### 4. REFUND_COMPENSATION
Money back, credit, fee waiver.
- Real example: *"will my credit card be refunded automatically?"*

### 5. ACCOUNT_LOYALTY
SkyMiles/Medallion status, miles, account changes.
- Real example: *"Just lost 35K miles on an award ticket and no waived fees."*

### 6. POLICY_QUESTION
Informational, no active problem.
- Real example: *"Can a small duffle bag count as a personal item?"*

### 7. SERVICE_COMPLAINT
Dissatisfaction with no specific fixable ask — includes severity flags
(discrimination, safety, legal threats) as well as ordinary venting.
- Real example: *"the hold up on DL74 to exit was ridiculous"*
- Real example (severe): *"I'll be filing a lawsuit against Delta for firing me after I reported racism and discrimination..."* — see "Non-customer messages" below for why this one is tricky.

### 8. PRAISE_FEEDBACK
Compliments, thank-yous, positive experiences.
- Real example: *"Huge thanks to Sebastian at the call center. Helped with a change and was great to work with!"*

---

## The 5 escalation rules

Checked in this order when you assign `true_escalate` — pick the FIRST one that applies, and put its exact name in `true_escalation_rule`:

1. **`hard_risk_keyword`** — genuine safety, discrimination, legal, or medical/minor-welfare content. This is for real risk, not just a message that happened to get sampled into the `risk_signal` bucket — read the actual content.
   - Real example: *"Just witnessed racism first hand. Fuck Delta airlines."*
   - NOT this: *"By the way, I love Delta. They never break my wheelchair."* — mentions "wheelchair" but is positive/casual, not risky. This one is `none_auto_handle`.

2. **`risk_category_default`** — BAGGAGE_PROBLEM or REFUND_COMPENSATION messages that don't hit a hard-risk keyword, but involve real money or real physical loss.
   - Real example: *"they cannot find our bags"*

3. **`weak_grounding`** — you genuinely can't picture what a reasonable, specific reply would look like (this anticipates what our retrieval step will face — if a human labeller can't imagine a good grounded answer, the system probably can't find one either).

4. **`ambiguous_classification`** — you seriously considered two different categories and neither is clearly right (this is exactly what `intent_ambiguous: true` should also be set for).

5. **`none_auto_handle`** — none of the above; auto-handle is fine.

---

## Special cases (found while reviewing real sampled data — handle consistently)

**Satirical / troll tweets.** A few sampled messages describe implausible scenarios (e.g. referencing meme figures, absurd reasons for being "kicked off a flight"). These aren't genuine complaints. Label your best-guess `true_intent` normally, but add `labeller_notes: "likely satirical, not a genuine complaint"`. Don't invent a 9th category for this — it's real signal for the failure-analysis section later, not a taxonomy gap.

**Non-customer messages.** At least one sampled message is from someone claiming Delta *fired* them (an employee, not a customer) making a discrimination/legal claim. Our taxonomy is built for customer support, not HR grievances. Label as `SERVICE_COMPLAINT`, set `intent_ambiguous: true`, and note `labeller_notes: "appears to be an employee/HR matter, not a customer support inquiry"`. This is legitimate "what we chose not to build" material for the report.

**Mixed intent (multiple issues in one message).** Pick whichever issue is the PRIMARY ask — usually the one that would determine what a reply needs to address first. Set `intent_ambiguous: true` if the two issues feel genuinely co-equal rather than one being clearly primary. Note the secondary issue in `labeller_notes`.

**Sarcasm / backhanded praise.** e.g. *"Thank you for getting me home safely. No thank you for the 2hrs diversion."* — read the actual sentiment, not just the presence of "thank you." This is SERVICE_COMPLAINT dressed in polite words, not PRAISE_FEEDBACK.

---

## Filling in each column

- `true_intent`: exactly one of the 8 names above (all caps, as written).
- `intent_ambiguous`: `TRUE` only if you seriously weighed two categories — not just "this was a little tricky."
- `true_escalate`: `TRUE` or `FALSE`.
- `true_escalation_rule`: if `true_escalate` is `TRUE`, one of the first 4 rule names above. If `FALSE`, exactly `none_auto_handle`.
- `reply_acceptability_notes`: a short note on what a good reply MUST contain or MUST avoid — not a full ideal reply. E.g. *"must not promise a specific refund amount; should acknowledge the delay specifically."*
- `labeller_notes`: anything that wasn't obvious, any special case above, or your reasoning for a close call.

Work in a few sittings rather than all 240 in one go — definition drift is real, and this guide is here so you can re-anchor each time you sit back down.
