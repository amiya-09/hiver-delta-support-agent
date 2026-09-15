# Phase Log

Running status of what's built and verified, per phase. Update the
"Status" line as work progresses; keep entries brief -- reasoning behind
decisions belongs in decision_log.md, not here.

## Phase 1 -- Data Pipeline
**Status: Done.**
Loaded raw 2,811,774-row dataset, filtered to Delta, reconstructed
customer<->brand reply pairs, cleaned. Output: 34,435 pairs
(`data/processed/delta_pairs.jsonl`). Verified against real data: brand
volume cross-checked against independent analysis (exact match), DM-redirect
rate measured at 18.4%, ~30% of pairs flagged `is_thread_root=false`
(mid-thread fragments, correctly excluded from taxonomy/golden-set
sampling). 20 unit tests passing.

## Phase 2 -- Retrieval Index
**Status: Done.**
Built a 2,500-pair retrieval pool (filtered to root, non-DM-redirect
pairs) embedded with a local `e5-small-v2` model (switched from Gemini API
after hitting its free-tier daily quota). Output:
`data/processed/retrieval_index/` (embeddings.npy + metadata.jsonl).
Verified: 4 hand-picked queries + 100 random real queries all retrieved
topically correct precedent with no cross-category contamination. Found
and worked around an intermittent Apple Accelerate BLAS bug (permanent
finite-score guard added). 42 unit tests passing (cumulative).

## Golden Set
**Status: Done.**
240 hand-labelled examples across 8 intent categories + stratified
risk-signal oversampling. Validator passes with zero problems.
- Per-intent counts: ACCOUNT_LOYALTY 24, BAGGAGE_PROBLEM 31,
  BOOKING_CHANGE 13, FLIGHT_DISRUPTION 34, POLICY_QUESTION 26,
  PRAISE_FEEDBACK 48, REFUND_COMPENSATION 29, SERVICE_COMPLAINT 35
- Escalation: 57 escalate / 183 auto-handle. Rule breakdown:
  hard_risk_keyword 20, risk_category_default 35, weak_grounding 1,
  ambiguous_classification 1, none_auto_handle 183
- Known gap: weak_grounding and ambiguous_classification have only 1
  example each -- insufficient to evaluate those two rules with
  confidence.
67 unit tests passing (cumulative, sampling/validation tooling).

## Phase 3 -- Intent Classification
**Status: Done.** 75.83% accuracy vs. golden set (20.0% trivial baseline,
59.6% keyword-rule baseline). Found 3 real failure modes (see
phase3_failure_analysis.md); a targeted prompt fix improved 22/58 known
errors but regressed 2/30 in a control sample (~79.9% estimated net,
validated via targeted recheck, not a full rerun). 97 unit tests passing
(cumulative through this phase).

## Phase 4 -- Reply Generation
**Status: Core built and reviewed. Evaluated on two small samples (10 +
30 examples), not the full 240**, due to Groq free-tier daily token
budget constraints. Found a real hallucination (a retrieved example's
phone number copied into an unrelated situation) and a retrieval
similarity-score calibration gap (see phase4_failure_analysis.md). A
proposed fix (similarity-floor instruction) is drafted, not yet tested.

## Phase 5 -- Routing / Escalation
**Status: Done, including full golden-set evaluation** (pure rule logic,
zero API cost). ESCALATE recall = 1.0000 in both isolated and end-to-end
modes. Quantified false-positive costs per rule and a confirmed
cross-phase interaction (a Phase 3 classification error accidentally
canceling out a Rule 2 over-escalation) -- see phase5_failure_analysis.md.
134 unit tests passing (cumulative through this phase).

## Phase 6 -- Evaluation Harness (metrics + LLM judge + judge-human agreement)
**Status: LLM-judge and agreement tooling built.** Original 10-example
human-scoring batch was contaminated by a sequencing flaw (human scorer
saw judge output before scoring -- documented, not hidden); redone
cleanly on a fresh, non-overlapping 30-example batch. 166 unit tests
passing (cumulative through this phase).

## Report
**Status: In progress.** Problem Framing and Results vs. Baselines
sections written. Failure Analysis, Misleading Headline Number, and Next
Steps sections still need consolidating from the phase-level
failure-analysis docs.
