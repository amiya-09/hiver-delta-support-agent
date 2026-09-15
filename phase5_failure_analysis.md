# Phase 5 Failure Analysis: Escalation Routing

Two evaluations run against all 240 golden-set examples (zero API cost --
routing is pure logic, retrieval uses the local embedding model):
**isolated** (routing given the golden true intent, `second_choice=None`
by construction) and **end-to-end** (routing given Phase 3's actual
predicted intent/second_choice, so classification error can compound).
23 of 240 examples (9.6%) were also present in the retrieval pool itself;
self-matches were detected and filtered before computing similarity, to
avoid artificially perfect grounding scores for those rows.

## Finding 1: Recall on ESCALATE is 1.0000 in both modes

Every one of the 57 golden examples that should escalate does escalate,
even with real classification errors compounding in the end-to-end case.
This is the core design goal (decision log #8: biased toward the safer
of the two possible mistakes) verified with real evidence, not just
argued for at design time.

## Finding 2: Rule 2 (`risk_category_default`) is a blunt, category-wide
trigger with a quantifiable false-positive cost

Of the 60 golden examples with `true_intent` in {BAGGAGE_PROBLEM,
REFUND_COMPENSATION}, only 35 (58%) actually needed escalation by human
judgment (`true_escalation_rule=risk_category_default`); the remaining
25 (42%) were judged `none_auto_handle`. Rule 2 escalates the entire
category unconditionally, correctly catching the 35 but also incorrectly
escalating the other 25. This single rule accounts for the majority of
isolated mode's 40 total false-positive escalations. This is the
documented precision/recall tradeoff from decision log #7 -- now
quantified rather than asserted.

## Finding 3: Classification uncertainty adds a second, distinct source
of over-escalation

Isolated mode's `ambiguous_classification` row is structurally always 0
(no real `second_choice` in isolated mode). End-to-end shows 26 examples
escalated via this rule that the golden standard says shouldn't have
been -- purely because Phase 3's classifier returned a genuine
`second_choice` for them. This is not the same failure as Finding 2; it's
a separate cost that only appears once real classification error is in
the loop. Overall ESCALATE precision drops from 0.588 (isolated) to
0.509 (end-to-end) -- total false positives grow from 40 to 55.

Interestingly, Rule 2's false-positive count *drops* from 25 (isolated)
to 14 (end-to-end) over the same set of examples. **Traced directly
against per-example data (not inferred from aggregates) -- this is NOT
an improvement.** Of the 14 examples where isolated mode's
`risk_category_default` false positive disappears in end-to-end mode,
only 2 shifted to `ambiguous_classification` as originally hypothesized.
**The other 12 (86%) shifted to `none_auto_handle` purely because Phase
3 misclassified the intent away from BAGGAGE_PROBLEM/REFUND_COMPENSATION
entirely** -- moving the message out of the only two categories Rule 2
watches, not because anything correctly judged it didn't need escalation.

The predicted (wrong) categories for these 12: `SERVICE_COMPLAINT` (5),
`POLICY_QUESTION` (4), `BOOKING_CHANGE` (1), `PRAISE_FEEDBACK` (1). Three
of the five `SERVICE_COMPLAINT` cases (`1064228`, `146036`, `2560926`)
are the exact same examples already documented in
`phase3_failure_analysis.md`'s Failure Mode 1
(REFUND_COMPENSATION -> SERVICE_COMPLAINT) -- this is not a new failure,
it's a second-order *consequence* of one already found and written up.
The four `POLICY_QUESTION` cases suggest Failure Mode 1's underlying
mechanism ("surface form beats topic specificity") generalizes beyond
the two category pairs the prompt fix specifically targeted.

**The conclusion this forces:** Rule 2's apparently better precision in
end-to-end mode is an artifact of two independent errors canceling out --
a classification mistake accidentally dodging an over-eager escalation
rule -- not evidence the pipeline performs better as a whole than its
parts do individually. If Phase 3's classifier were more accurate on
these 12 examples, Rule 2's real false-positive rate would surface, not
improve. This is a genuine example of why a single end-to-end metric can
be actively misleading: it can look better specifically *because* an
upstream error is masking a downstream one.

## Finding 4: Rule 1 (`hard_risk_keyword`) has an intrinsic ~7.7%
false-positive rate, independent of classification

14 of 183 true-`none_auto_handle` examples match a risk keyword anyway,
identically in both isolated and end-to-end modes (since Rule 1 only
depends on raw text, never on classification output). This is the
quantified cost of the deliberately loose keyword list from golden-set
labelling (e.g. "my son," "wheelchair" catching benign mentions) --
previously a documented design choice, now a measured 7.7% base rate.

## Finding 5: Structural zero-coverage gap for Rules 3 and 4's
true-positive behavior

The golden set has exactly one true example each for `weak_grounding`
and `ambiguous_classification` (decision log #17). In both evaluations,
that single example for each rule gets intercepted by Rule 1
(`hard_risk_keyword`) before ever reaching its intended rule, since Rule
1 is checked first. **We have zero real evidence that Rules 3 or 4
correctly fire when genuinely needed** -- only, from end-to-end,
evidence of Rule 4 firing incorrectly (Finding 3). This is a real,
structural gap in evaluation coverage, not a flaw in the rules
themselves that we can currently observe or rule out.

## What this means for the headline number

Rule-level accuracy (0.808 isolated, 0.738 end-to-end) looks like a
single coherent number but actually aggregates at least three distinct,
separately-caused sources of over-escalation (Rule 1's ~7.7% base rate,
Rule 2's ~42% within-category rate, Rule 4's classification-uncertainty
cascade) plus a structural blind spot on two of the five rules entirely.
A reader should not treat either accuracy figure as "how good the
escalation logic is" without this breakdown -- the three problems would
need three different fixes, not one.

## Next steps (not yet done)
- Consider whether Rule 2 should incorporate a secondary signal (e.g.
  combine category with a severity/keyword check) rather than being a
  blanket category-wide trigger, given the measured 42% false-positive
  rate -- this matters more now that we know some of that rate is
  currently hidden by classification errors, not fixed by them.
- The Rules 3/4 coverage gap can't be closed without additional golden-set
  examples specifically chosen to test those rules without also
  triggering Rule 1 first -- a labelling task, not a code fix.
- Since three of the twelve Finding-3 examples are already-documented
  Phase 3 failures, fixing Phase 3's classification accuracy on
  REFUND_COMPENSATION/POLICY_QUESTION confusions (already planned) should
  be expected to *increase* Rule 2's measured false-positive rate in
  end-to-end mode, not decrease it -- worth checking this prediction
  explicitly once that fix is validated, as a test of whether we actually
  understand the mechanism or just described it after the fact.
