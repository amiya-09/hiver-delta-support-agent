# Report: Delta AI Support Agent

## Problem Framing

### What "good" means for this brand

I'm defining good for a Delta support agent as four things, roughly in
order of how much they matter.

First, it has to actually understand what the customer is asking, even
when the message is messy real tweet text, not a clean support ticket.
I measure that against my own hand-labelled ground truth, not just
whether the output looks reasonable on a few cherry-picked examples.

Second, replies have to be grounded in how Delta has actually handled
similar situations before, not just something that sounds plausible. A
reply that invents a detail is worse than one that's honest it doesn't
know, since it can actively mislead a customer and create real problems
for the brand.

Third, the escalation decision has to be conservative in a specific
direction. Getting a low-stakes case wrong by sending it to a human
costs a little time. Getting a genuinely risky case wrong by letting a
bot handle it, a discrimination complaint or a safety issue, costs a lot
more. I designed the whole escalation logic around that asymmetry on
purpose.

Fourth, the system has to be honest about its own weak points. A system
that reports one good-looking number while quietly hiding where it
actually struggles isn't good, even if the number is high. A decent
chunk of this project ended up being about finding and reporting exactly
those weak points.

### What I chose not to build

- Multi-language support. I scoped this to English tweets only. The
  dataset has some non-English content, but translating it well enough
  to trust for real customer support felt like a separate project.
- Special handling for messages that aren't actually customer support
  situations. One example I found while labelling was someone claiming
  Delta fired them, an employee complaint, not a customer one. My
  taxonomy doesn't have a real place for that, and I decided that's fine
  rather than force a category for something this rare.
- A dedicated system for catching satirical or joke tweets. A couple of
  sampled messages clearly weren't genuine complaints. Building a
  reliable detector for that felt like more effort than it was worth for
  something this uncommon, so I just documented it as a known limitation
  instead.
- An autonomous, multi-step planning agent. The task has a fixed,
  known sequence, classify, retrieve, generate, route, with no real
  decision-making about what to do next, so a simple pipeline is easier
  to test and explain than a full agent framework would be.
- A trained model for the escalation decision. I used a fixed set of
  rules instead. Training something on the golden set would mean using
  the same data to build and evaluate it, which is circular, and rules
  mean the reason the system gives for escalating is the actual reason,
  not something reconstructed after the fact.
- A real vector database for retrieval. At 2,500 vectors, plain numpy
  cosine similarity is basically instant, so a whole database felt like
  unnecessary infrastructure.
- Any actual integration with a live human handoff system. The system
  produces a decision and a reason. Actually connecting that to a real
  ticketing workflow is out of scope.
- Fine-tuning anything. Both the chat model and the embedding model are
  used as-is. Given the timeline, prompt-level fixes were the cheaper
  lever to try first, and Failure Mode 2 below covers what actually
  happened when I tried one.

## Results vs. Baselines

I compared three systems on the same 240-example golden set: always
guessing the single most common category (the trivial baseline), a
simple keyword-rule classifier (the simple baseline, reused directly
from the tooling I built to sample the golden set in the first place),
and my actual LLM classifier.

| System | Accuracy |
|---|---|
| Trivial (always guess the mode) | 20.0% |
| Keyword rules | 59.6% |
| LLM classifier | 75.83% |

The LLM beats the keyword baseline by about 16 points, which is a real
gap, not just clearing the trivial floor that any working system should
clear easily.

One category ran against that overall trend early on: on ACCOUNT_LOYALTY
specifically, the keyword baseline did better than my LLM classifier
(0.818 F1 vs. 0.611 F1). Failure Mode 1 below explains why. After the
topic-specificity fix described in Failure Mode 2, this gap nearly
closed, ACCOUNT_LOYALTY F1 improved to 0.800, almost matching the
keyword baseline rather than losing to it by a wide margin.

Excluding the 6 examples I'd personally marked ambiguous while labelling
only moves the number from 75.83% to 75.64%, barely anything. That
turned out to matter: it means the model's mistakes aren't concentrated
in the genuinely hard cases I struggled with myself, they're a real,
systematic pattern, which Failure Mode 1 covers.

I also tried a targeted fix for that pattern and tested it cheaply (see
Failure Mode 2), which the targeted recheck estimated would bring
accuracy to about 79.9%. I later ran a full, real reclassification of
all 240 examples under the fixed prompt (needed anyway to complete the
cache for a clean reproduction, see decision log), and the real number
came in at 78.75%, close to the estimate but not identical, with real
costs of its own that I don't want to hide behind the improved number.

## Failure Analysis

Five failure modes, picked because they're the best evidenced and cover
different parts of the system, not just the ones that were easiest to
write up.

### 1. The classifier picks the generic category over the specific one when phrasing and topic disagree

Reading the model's own stated reasoning on wrong answers, not just the
confusion matrix, showed a clear pattern. When a message's topic points
to one category but how it's phrased points to a more generic one, the
model goes with the generic one. A message asking a question about
SkyMiles gets classified as POLICY_QUESTION instead of ACCOUNT_LOYALTY,
even when the model's own reasoning explicitly names SkyMiles as the
topic. A frustrated message about a refund gets SERVICE_COMPLAINT
instead of REFUND_COMPENSATION, same pattern. This is the direct cause
of the ACCOUNT_LOYALTY baseline gap above, and it accounts for around a
quarter of all the misclassifications I found.

### 2. A targeted fix for that pattern helped more than it hurt, but it wasn't clean

I added a rule telling the model to weight topic over phrasing for
exactly those two category pairs, and tested it against the 58 examples
it was meant to fix plus a random 30-example control sample, rather than
re-running the whole golden set (Groq's daily token budget made a full
rerun expensive to repeat every time I changed the prompt). 22 of the 58
known errors got fixed. But 2 of the 30 control examples that were
previously correct broke. Tracing those two by hand, one was the model
over-applying the new rule to a case where the loyalty term was only
mentioned in passing, not the actual subject of the message. The other
was a security-incident complaint that flipped to FLIGHT_DISRUPTION for
no clear reason connected to what I'd actually written, suggesting the
general idea behind the rule spread further than the two specific cases
I wrote it for. Net effect is a real improvement, about 79.9% up from
75.83%, but not without a real, measurable cost.

### 3. Retrieval similarity scores don't spread out enough to tell relevant from irrelevant

Across every query I tested by hand, the top retrieved results all
scored somewhere between 0.83 and 0.90, regardless of whether the
message was genuinely well matched or not. My generation prompt tells
the model to back off and give a generic answer when nothing retrieved
is actually relevant, but that instruction never got a real chance to
fire, since retrieval never handed it a score low enough to look
obviously weak. This looks like a real limitation of using a small,
general-purpose embedding model on tweet-length text, where a lot of
different problems share enough surface vocabulary to all look "fairly
similar."

### 4. That calibration gap produced a real, concrete case of fabrication, and the judge missed it too

One generated reply took a phone number from a retrieved example about
a completely different customer's wifi problem and handed it to someone
with an unrelated issue, with a similarity score that looked completely
normal. When I built the LLM judge later and scored the same kind of
output myself blind, this exact pattern showed up again, worse than I
expected. On the dimension I specifically wrote to catch fabricated or
misapplied details, every single disagreement between my scores and the
judge's was a full swing from 1 to 3, never a soft difference like the
other three dimensions had. Reading through those six cases, they all
share the same shape: the model states something specific and concrete
that isn't actually true for that particular customer (a reused phone
number, an invented "we're aware of the issue" claim, a fabricated
carry-on measurement, a flat "no refund" borrowed from an unrelated
case), and the judge scores all of them a perfect 3 anyway. The judge
appears to check whether a reply topically relates to something
retrieved, not whether its specific claims are actually true for this
customer, which is exactly the blind spot I wrote its instructions to
catch, and it missed it anyway.

### 5. The escalation rule for baggage and refund cases is a blunt, all-or-nothing trigger

Of the 60 golden examples tagged BAGGAGE_PROBLEM or REFUND_COMPENSATION,
only 35 actually needed escalation by my own labelling. My rule
escalates the entire category regardless, so it's wrong on 42% of that
category by itself. Something more interesting turned up comparing this
rule's behavior when fed the correct intent directly versus fed Phase
3's real (sometimes wrong) predictions: the false-positive count for
this rule actually looked better once real classification error was in
the loop, dropping from 25 down to 14. I didn't trust that at face value
and traced the actual examples instead of just taking the aggregate
number. Almost all of that apparent improvement was really Phase 3
misclassifying those messages into an unrelated category, which
accidentally moved them out of the one rule that would have wrongly
escalated them. Three of those examples are the exact same ones already
flagged in Failure Mode 1. The pipeline wasn't getting smarter as a
whole, one bug was quietly hiding another one.

## What Is Misleading About My Headline Number

If I had to give one number to describe this project, it would probably
be the 75.83% classification accuracy. That number is real, but on its
own it hides more than it shows.

It doesn't say anything about how classification errors propagate
downstream. Failure Mode 5 above is the clearest example: a Phase 3
mistake can accidentally cancel out a Phase 5 mistake, making the
combined system look better on one specific number than either
component actually is on its own. I actually tested this after fixing
Phase 3's classification: Rule 2's end-to-end false-positive rate got
worse, not better (rule-level accuracy dropped from 0.738 to 0.700),
exactly as the mechanism predicted. A single end-to-end number would
have completely hidden that this specific improvement had that specific
cost.

The escalation recall number tells a similar story from the other
direction. ESCALATE recall is a perfect 1.0, every message that should
escalate does. That sounds great in isolation, but it's largely a
product of one rule (the baggage/refund category default) firing
unconditionally on an entire category, escalating 60 messages to
correctly catch 35 of them. A perfect recall number doesn't tell you it
was bought by being wrong 42% of the time within that category.

And the biggest one: the 75.83% and 79.9% numbers, and the judge
agreement numbers, all come from a golden set and small samples I built
and scored myself. That's the whole point of having an independent
golden set, but it also means every one of these numbers inherits
whatever I personally got wrong or was inconsistent about while
labelling 240 examples by hand across several sittings. I tried to guard
against drift with a written labelling guide, but I can't claim that
guarantees perfect consistency.

## What I'd Do Next With One More Week

- Actually test the similarity-floor fix I drafted for generation, and
  properly investigate why retrieval scores cluster so tightly in the
  first place, possibly trying a different or larger embedding model.
- Rework the judge's factual-consistency instructions given the
  confirmed blind spot. Right now it seems to check topical relevance
  rather than claim-by-claim accuracy. I'd try forcing it to explicitly
  list which specific claims in the reply are or aren't supported by the
  retrieved examples, rather than giving one holistic score.
- Add more golden-set examples specifically aimed at the weak_grounding
  and ambiguous_classification escalation rules, which currently have
  only one real example each. They'd need to be chosen carefully so they
  don't also trigger the hard-risk-keyword rule first, since that rule
  runs before them and would otherwise swallow the test case.
- Reconsider the baggage/refund escalation rule given its 42%
  false-positive rate. Some kind of secondary signal beyond just the
  category, maybe combined with a severity check, might do better than
  an unconditional blanket trigger.
- Run the full 240-example evaluation for generation and judging,
  instead of the 10- and 30-example samples I used given the Groq
  budget constraints, and actually check the falsifiable prediction from
  Failure Mode 5: that fixing Phase 3's classification issue should make
  Rule 2's end-to-end false-positive rate look worse, not better.
