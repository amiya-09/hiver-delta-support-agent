# Golden Set Labelling Guide

This is the reference I labelled the 240 examples against, not memory or
gut feel in the moment. My judgment was going to drift over a 240-row
session if I didn't anchor it to something fixed, so this is that
anchor. If a call still felt ambiguous after checking this, that's
exactly what `intent_ambiguous` and `labeller_notes` are for. I didn't
force a clean answer where the data genuinely didn't have one.

---

## The 8 intent categories

### 1. FLIGHT_DISRUPTION
Delay, cancellation, ground stop, missed or tight connection.
- Real example: *"Will flight 5330 depart anytime soon?? After two delays, we were supposed to depart at 1050. We're still on the ground."*
- Not this one: a delay mentioned only in passing while the real ask is something else. "My flight was delayed and I want a refund" is REFUND_COMPENSATION with the delay as context, see "Mixed intent" below.

### 2. BAGGAGE_PROBLEM
Lost, damaged, delayed, or mishandled baggage, including mobility
equipment like wheelchairs since those get handled as baggage.
- Real example: *"Hours of delay, no communication and now they cannot find our bags."*
- Equipment example: *"losing my sons wheelchair and won't pay for it"*

### 3. BOOKING_CHANGE
Seat change, rebooking, name correction, trip modification.
- Real example: *"Trying to change an assigned seat on a flight today and unable to do so. Can you help?"*

### 4. REFUND_COMPENSATION
Money back, credit, fee waiver.
- Real example: *"will my credit card be refunded automatically?"*

### 5. ACCOUNT_LOYALTY
SkyMiles or Medallion status, miles, account changes.
- Real example: *"Just lost 35K miles on an award ticket and no waived fees."*

### 6. POLICY_QUESTION
Informational, no active problem.
- Real example: *"Can a small duffle bag count as a personal item?"*

### 7. SERVICE_COMPLAINT
Dissatisfaction with no specific fixable ask. Includes the severe cases
too (discrimination, safety, legal threats), not just ordinary venting.
- Real example: *"the hold up on DL74 to exit was ridiculous"*
- Severe example: *"I'll be filing a lawsuit against Delta for firing me after I reported racism and discrimination..."* This one's actually tricky, see "Non-customer messages" below.

### 8. PRAISE_FEEDBACK
Compliments, thank-yous, positive experiences.
- Real example: *"Huge thanks to Sebastian at the call center. Helped with a change and was great to work with!"*

---

## The 5 escalation rules

I checked these in order and picked the first one that applied, putting
its exact name in `true_escalation_rule`.

1. `hard_risk_keyword`, genuine safety, discrimination, legal, or
medical/minor welfare content. This is for actual risk, not just
anything that happened to land in the `risk_signal` sample bucket, so I
had to actually read the content rather than trust the bucket.
   - Real example: *"Just witnessed racism first hand. Fuck Delta airlines."*
   - Not this one: *"By the way, I love Delta. They never break my wheelchair."* Mentions wheelchair but it's positive and casual, not risky. This one is `none_auto_handle`.

2. `risk_category_default`, BAGGAGE_PROBLEM or REFUND_COMPENSATION
messages that don't hit a hard-risk keyword but involve real money or
real physical loss.
   - Real example: *"they cannot find our bags"*

3. `weak_grounding`, I genuinely can't picture what a specific, useful
reply would look like. This is meant to anticipate what the retrieval
step will run into. If I can't imagine a grounded answer myself, the
system probably can't find one either.

4. `ambiguous_classification`, I seriously considered two different
categories and neither felt clearly right. Same situations where I set
`intent_ambiguous` to true.

5. `none_auto_handle`, none of the above, auto-handle is fine.

---

## Special cases I ran into while actually reading the data

**Satirical or troll tweets.** A few sampled messages describe things
that don't really add up, referencing meme figures or absurd reasons
for getting kicked off a flight. These aren't real complaints. I still
labelled my best guess at `true_intent` normally, but added a note like
"likely satirical, not a genuine complaint" instead of inventing a whole
new category for something this rare. It's a real finding worth
mentioning in the failure analysis later, not a gap in my taxonomy.

**Non-customer messages.** At least one sampled message is from someone
saying Delta fired them, an employee complaint, not a customer one. My
taxonomy was built for customer support, not HR issues, so I labelled it
SERVICE_COMPLAINT, set `intent_ambiguous` to true, and noted that it
looks like an employee matter rather than a support inquiry. Good
example for the "what I chose not to build" part of the report.

**Mixed intent, multiple issues in one message.** I picked whichever
issue was the primary ask, usually whatever a reply would need to
address first. If the two issues felt genuinely equal rather than one
being clearly primary, I set `intent_ambiguous` to true and noted the
secondary issue.

**Sarcasm or backhanded praise.** Something like "Thank you for getting
me home safely. No thank you for the 2hr diversion." I had to read the
actual sentiment, not just react to the word "thank you." That one's
SERVICE_COMPLAINT dressed up politely, not PRAISE_FEEDBACK.

---

## How I filled in each column

- `true_intent`: exactly one of the 8 category names, all caps.
- `intent_ambiguous`: TRUE only if I genuinely weighed two categories,
  not just "this one was a little tricky."
- `true_escalate`: TRUE or FALSE.
- `true_escalation_rule`: one of the first four rule names if
  `true_escalate` is TRUE, otherwise exactly `none_auto_handle`.
- `reply_acceptability_notes`: a short note on what a good reply must
  contain or must avoid, not a full example reply. Something like "must
  not promise a specific refund amount, should acknowledge the delay
  specifically."
- `labeller_notes`: anything that wasn't obvious, any of the special
  cases above, or my reasoning on a close call.

I worked through this in a few separate sittings instead of all 240 at
once, since my own judgment would drift over that many rows in a single
session. Coming back to this guide each time I sat down was the whole
point of writing it out properly instead of just keeping it in my head.
