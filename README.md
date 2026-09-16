# Delta AI Support Agent — Hiver SDE Intern Take-Home

An AI customer-support agent for Delta Air Lines, built from the Kaggle
"Customer Support on Twitter" dataset. Classifies intent, drafts a reply
grounded in real historical precedent, and decides whether to auto-handle
or escalate to a human — with a stated reason for every decision.

See `report.md` for the full writeup (problem framing, results vs.
baselines, failure analysis, known limitations, next steps) and
`decision_log.md` for the reasoning behind every non-obvious choice made
along the way.

---

## Setup

Requires **Python 3.11** and internet access for the one-time embedding
model download (~130MB, no API key needed for that step).

```bash
git clone https://github.com/amiya-09/hiver-delta-support-agent.git
cd hiver-delta-support-agent
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**No raw dataset download needed.** `data/processed/delta_pairs.jsonl`
(the cleaned, reconstructed customer↔brand pairs from Phase 1) is
committed directly, so nothing beyond this repo is required to reproduce
everything from Phase 2 onward.

**`GROQ_API_KEY` is optional for reproduction.** Every classification,
generation, and judge call made during this project's real development
is cached under `.cache/` and committed to the repo. The commands below
reproduce the reported results entirely from that cache — no live API
calls, no key needed. A key is only required if you want to run against
*new* inputs beyond what's already cached (a different sample, a
different seed, etc.).

---

## Reproducing the results

Run these in order. Steps 1–2 are one-time setup; steps 3 onward
reproduce each phase's reported numbers.

### 1. Rebuild the retrieval index (local, no API key, ~1 minute)

```bash
python -m src.retrieval.build_index
```

Downloads the local embedding model on first run (one-time), then embeds
a 2,500-pair retrieval pool from `delta_pairs.jsonl`. Fully deterministic
(fixed seed) — this regenerates the exact pool used throughout the project.

### 2. Run the full test suite

```bash
pytest tests/ -v
```

### 3. Phase 3 — intent classification vs. the golden set

```bash
python -m src.eval.run_classification_eval
```

Reproduces the 75.83% baseline accuracy reported in `report.md`, entirely
from `.cache/classify/`.

### 4. Baselines (trivial + keyword-rule) for comparison

```bash
python -m src.eval.run_baseline_eval
```

Zero API cost regardless of cache — reads Phase 3's saved results and
runs a rule-based classifier locally.

### 5. Phase 5 — escalation routing evaluation

```bash
python -m src.eval.run_routing_eval
```

Zero API cost — routing is pure rule logic; only uses the local retrieval
index.

### 6. Phase 4 — reply generation (both sampled batches used in the report)

```bash
python -m src.eval.run_generation_eval --sample-size 10 --seed 42
python -m src.eval.run_generation_eval --sample-size 30 --seed 99 --output data/eval/phase4_generation_results_clean.json
```

### 7. Phase 6 — LLM-as-judge scoring (both batches)

```bash
python -m src.eval.run_judge_eval
python -m src.eval.run_judge_eval --input data/eval/phase4_generation_results_clean.json --output data/eval/phase6_judge_results_clean.json
```

### 8. Judge–human agreement

```bash
python -m src.eval.compute_judge_agreement --judge-results data/eval/phase6_judge_results_clean.json --human-scores data/eval/human_scoring_template_clean_scored.csv --output data/eval/phase6_judge_agreement_clean.json
```

Uses the hand-scored CSV committed to the repo (scored blind, before any
judge output was seen — see decision log for the sequencing fix that
made this rigorous).

---

## Project structure
src/pipeline/ Phase 1: load, clean, reconstruct Delta customer<->brand pairs
src/retrieval/ Phase 2: local embedding index + similarity search
src/agent/ Phases 3-5: classify, generate, route, + end-to-end pipeline
src/eval/ Evaluation harness: metrics, baselines, judge, agreement
src/llm/ Shared API client infrastructure (caching, retry logic)
src/golden_set/ Golden-set sampling and label validation tooling
prompts/ Versioned prompt templates, one per stage
tests/ Unit tests, zero real API calls
data/golden_set/ 240 hand-labelled examples (ground truth)
data/eval/ Real evaluation outputs from this project's runs
.cache/ Cached API responses (committed, for zero-cost reproduction)

## Where to find more

- `report.md` — problem framing, results vs. baselines, failure analysis, limitations, next steps
- `decision_log.md` — every non-obvious decision made, and why
- `citations.md` — dataset, models, libraries, and AI assistance used
- `PHASE_1-DATA_PIPELINE.md` through `PHASE_6-LLM_JUDGE.md`, `GOLDEN_SET.md` — the story of each phase, what I built and what actually happened
- `phase1_failure_analysis.md` through `phase6_failure_analysis.md` — detailed, evidence-based findings per phase, with real examples
- `golden_set_labelling_guide.md` — the reference I labelled the golden set against
- `PHASE_LOG.md` — running status per phase

## Known limitations (see `report.md` for full discussion)

- Phase 4/6 evaluated on two small samples (10 + 30 examples), not the
  full 240-example golden set, due to Groq free-tier daily token budget
  constraints encountered during development.
- The golden set has only 1 example each for the `weak_grounding` and
  `ambiguous_classification` escalation rules — insufficient to evaluate
  those two rules with statistical confidence.
- A prompt fix for a known Phase 3 classification failure mode was
  validated via a targeted recheck (not a full golden-set rerun) — see
  `phase3_failure_analysis.md`.
