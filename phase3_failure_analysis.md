# Phase 3 Failure Analysis: Intent Classification

Baseline result before any fix: 75.83% accuracy on the 240-example
golden set, against a 20.0% trivial baseline and a 59.6% keyword-rule
baseline. Three real patterns came out of reading the model's own
stated reasoning for every wrong answer, not just staring at the
confusion matrix.

## The model picks the generic category over the specific one when phrasing and topic disagree

This affects around 14 of roughly 58 total misclassifications, close to
a quarter, the single biggest cause I found.

Two category pairs show the exact same mechanism. When a message's
phrasing matches one category's definition, a question phrased like
POLICY_QUESTION, a frustrated tone phrased like SERVICE_COMPLAINT, while
its actual topic points to something more specific, the model defaults
to the generic, form-matching category instead.

Real examples where ACCOUNT_LOYALTY got mislabelled as POLICY_QUESTION,
with the model's own stated reasoning: one message asked about a
SkyMiles bonus-miles policy for donations, and the model's reasoning
literally named the SkyMiles topic, then still picked the generic
category because the message was phrased as a question. Another asked
about using miles for an upgrade, same pattern.

Real examples where REFUND_COMPENSATION got mislabelled as
SERVICE_COMPLAINT: one message said a voucher couldn't be applied to a
new ticket, and the model's own reasoning named the voucher topic
directly, then defaulted to the generic complaint category because of
the frustrated tone.

The prompt's category definitions just didn't say what to do when
phrasing and topic point to different places, and I confirmed that by
reading the model's actual stated reasoning, not by guessing.

## I tried a fix, and it helped more than it hurt, but not cleanly

I added a rule telling the model to weight topic over phrasing for
those two category pairs specifically. To test it without re-running
the whole golden set (Groq's daily token budget made that expensive to
repeat every time), I built a tool that only rechecks the previously
wrong examples plus a random sample of previously correct ones.

22 of 58 known errors got fixed. But 2 of 30 examples from the control
sample, previously correct, broke. That's a real net improvement,
about 79.9% estimated up from 75.83%, but not a clean win, and I think
it's more honest to say so than to only report the 22 fixed cases.

I went back and looked at the two regressions individually instead of
just accepting the net number. One was the message "do I not have to
pay for checked baggage if I have SkyTeam Elite status," true category
POLICY_QUESTION, correct before the fix, now flipped to ACCOUNT_LOYALTY.
My new rule says to use ACCOUNT_LOYALTY when the core subject is
loyalty program mechanics, but here the elite status is only a
qualifying condition for a baggage-fee question, not what the message
is actually about. The model triggered on the mere presence of a
loyalty term rather than genuinely checking whether it's the core
subject, even though the rule's wording explicitly asked for that
distinction.

The other regression was stranger. A message about being dragged off a
flight during a security incident, true category SERVICE_COMPLAINT,
flipped to FLIGHT_DISRUPTION. This message has no actual connection to
either of my rule's two named triggers, loyalty or refund language.
My best guess is that the general idea behind the rule, match by topic
instead of surface phrasing, generalized further than the two specific
cases I wrote it for, and the word "airline" ended up read as a signal
toward FLIGHT_DISRUPTION even though the real content is a
safety complaint. This is a real risk with prompt-based fixes generally:
a rule aimed at two specific confusions can shift behavior somewhere
else entirely, through the general principle it teaches, not just its
literal wording.

## A pattern that turned out to be a genuine gray area, not a model mistake

I originally guessed this one was a sarcasm-detection failure, since my
prompt has a rule for a similar surface pattern. Reading the actual
model reasoning disproved that. These four cases (FLIGHT_DISRUPTION
mislabelled as PRAISE_FEEDBACK) aren't sarcastic at all, they're
genuine thank-yous for how a disruption got handled, with nothing left
to complain about. My own labelling guide's rule for mixed intent,
"pick the primary ask," doesn't actually cover a message with no ask at
all, just gratitude about a resolved problem. Whether the true category
should be the underlying event or the expressed sentiment is a real,
reasonable disagreement a human labeller could land on either way. I'm
documenting this as a genuine boundary in my own taxonomy, not
something a prompt fix should try to paper over.

## What this means for the headline number

75.83% isn't purely a measure of how good the model is. Part of it
reflects a real gap in my taxonomy that a reasonable person could
disagree with me on, and part of it reflects a fixable, evidence-backed
prompt problem. Anyone reading that number in isolation would be
missing both of those things.
