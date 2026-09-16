# Phase 5 Failure Analysis: Escalation Routing

I evaluated this two ways against all 240 golden-set examples, zero API
cost since routing is pure logic: once giving it the correct golden
intent directly (isolated), and once giving it Phase 3's actual
predicted intent (end-to-end), so I could see how real classification
mistakes affect it once they're in the loop. About 1 in 10 golden
examples turned out to also be sitting inside the retrieval pool itself,
since both were sampled from the same underlying data, so I had to
filter those out before trusting any similarity score, otherwise
retrieval would just find a message matching itself and score
artificially close to perfect.

## Recall on escalation is a perfect 1.0 in both modes

Every single one of the 57 golden examples that should escalate does
escalate, even with real classification errors stacked on top in the
end-to-end run. This is exactly the outcome I designed for from the
start, biased toward the safer of the two mistakes, and now it's
actually verified with real evidence instead of just argued for on
paper.

## The baggage and refund rule is a blunt, all-or-nothing trigger

Of the 60 golden examples tagged BAGGAGE_PROBLEM or REFUND_COMPENSATION,
only 35 actually needed escalation by my own labelling. My rule
escalates the whole category regardless, correctly catching the 35 but
also wrongly escalating the other 25, wrong on 42% of that category by
itself. This one rule accounts for the majority of the isolated run's
40 total false-positive escalations.

## Classification uncertainty adds a second, separate source of over-escalation

In the isolated run, `ambiguous_classification` can never fire, since
there's no real second-choice value in that mode by construction. In
the end-to-end run, 26 examples got escalated through that exact rule
that the golden labels say shouldn't have been, purely because Phase
3's classifier genuinely returned a second choice for them. Overall
escalation precision drops from 0.588 isolated to 0.509 end-to-end,
false positives growing from 40 to 55.

Something more interesting turned up comparing the two runs side by
side, back when Phase 3's classifier was still running under the
pre-fix prompt. The baggage/refund rule's false positives actually
dropped from 25 to 14 in the end-to-end run over the same set of
examples, which looked at first like the pipeline was somehow doing
better as a whole than its individual pieces. I didn't trust that and
traced the actual examples instead of taking the aggregate number at
face value. Of the 14 examples where that false positive disappeared,
only 2 shifted to the ambiguous rule as I first guessed. The other 12
shifted purely because Phase 3 misclassified the intent away from
baggage or refund entirely, moving the message out of the only two
categories that rule even watches, not because anything correctly
judged it didn't need escalation.

The wrong predicted categories for those 12: SERVICE_COMPLAINT for 5 of
them, POLICY_QUESTION for 4, one each to BOOKING_CHANGE and
PRAISE_FEEDBACK. Three of the SERVICE_COMPLAINT cases are the exact
same examples I'd already flagged as a Phase 3 classification failure.
This isn't a new problem, it's a second-order consequence of one I'd
already found. The POLICY_QUESTION cases suggest that same underlying
mechanism, surface phrasing beating topic specificity, generalizes
beyond the two category pairs my prompt fix specifically targeted.

The real conclusion at the time: the rule's apparently better precision
in the end-to-end run was two independent errors canceling out, a
classification mistake accidentally dodging an over-eager escalation
rule, not evidence the pipeline performs better together than its parts
do individually. That gave me a falsifiable prediction: if Phase 3's
classifier got more accurate on these 12 examples specifically, this
rule's real false-positive rate should surface, not improve.

I later actually fixed Phase 3's classification (see the topic
specificity rule and the real, complete reclassification described in
its own failure analysis) and reran this evaluation against the same
240 examples. The prediction held. The end-to-end `risk_category_default`
false positive count went from 14 back up to 25, exactly matching the
isolated run's number, meaning the classifier is now accurate enough on
these examples that the rule's own true false-positive rate is fully
exposed rather than partially masked. End-to-end precision dropped from
0.509 to 0.467, and end-to-end rule-level accuracy dropped from 0.738 to
0.700, both moving in exactly the direction I predicted, not the
direction that would look better on a headline number. That's a genuine
confirmed example of a single end-to-end metric hiding the fact that
one upstream error was masking a downstream one, not just a plausible
story.

## The safety keyword rule has its own steady false-positive rate

14 of 183 examples that shouldn't escalate match a risk keyword anyway,
identical in both isolated and end-to-end modes, since this rule only
looks at raw text and never touches classification output. This is the
real, measured cost of deliberately keeping that keyword list loose
("my son," "wheelchair" catching plenty of harmless mentions), a
tradeoff I made on purpose while building the golden set, now with an
actual number attached to it.

## Two of the five rules have essentially no real test coverage

The golden set only has one true example each for weak grounding and
ambiguous classification. In both evaluation modes, that single example
for each rule gets intercepted by the safety keyword rule before it
ever reaches its intended rule, since that one runs first. I have zero
real evidence either of those two rules correctly fires when genuinely
needed, only, from the end-to-end run, evidence that the ambiguous rule
fires incorrectly. This is a real gap in what I've actually tested, not
something I can currently rule out or confirm either way.

## What this means for the headline number

Rule-level accuracy, 0.808 isolated and 0.700 end-to-end (after the
Phase 3 classification fix; it was 0.738 before), looks like one
coherent number but is really hiding at least three separately caused
sources of over-escalation, plus a real blind spot on two of the five
rules entirely. Whichever of those numbers gets quoted as "how good the
escalation logic is," it would be missing that these are three
different problems that would need three different fixes, not one, and
that fixing one of them (classification) made a downstream number look
worse, not better, exactly as predicted above.
