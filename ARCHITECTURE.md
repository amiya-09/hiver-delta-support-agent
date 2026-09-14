# Architecture Design
Think of the whole system as a conveyor belt. A tweet comes in from one end, and by the other end we've figured out what the customer wants, drafted a reply, and decided if a human needs to step in.

** High-level pipeline **
raw tweets (Kaggle CSV)
   │
   ▼
1. DATA PIPELINE (load, filter to brand, reconstruct threads)
   │
   ▼
2. DATA CLEANING (normalize text, drop noise, dedupe)
   │
   ▼
3. STORAGE (flat files: parquet/JSONL — no database)
   │
   ▼
4. SUBSAMPLE (documented strategy) ──► EVAL SET (hand-labelled, held out)
   │
   ▼
5. RETRIEVAL INDEX (embeddings over historical resolved threads)
   │
   ▼
6. AGENT (per incoming message: classify → retrieve → generate reply → route)
   │
   ▼
7. EVALUATION HARNESS (metrics + LLM judge + judge-human agreement)
   │
   ▼
8. REPORT (baselines, failure analysis, misleading-number section)

# Getting the Data Ready
We start with millions of raw tweets between customers and different brands. First thing we do is pick one brand and pull out only their conversations.
But the tricky part is that a customer tweet and the brand's reply to it are separate rows in the dataset, but they are connected by thread ID fields like ("this tweet replies to that tweet") 'in_response_to_tweet_id' / 'response_tweet_id'.
We need to stitch these back together into pairs: "Customer complained about X" -> "Brand replied with Y". 

Why does this matter so much ?
=> Because later, when we want to draft a reply "grounded in how the brand historically resolved similar issues", we need real complaint-and-solution pairs to point to.

** Simplest Alternative **
Ignore the pairing and just dump all the brand's tweets together as "here's how this brand talks." Why we don't do that: it gives us tone, not solutions. If a customer asks about a refund, we want to find a past refund complaint that got solved, not just any brand tweet.

** Trade-off **


# Cleaning the Data
Real tweets are messy — spam, duplicate complaints, empty junk after removing links and @mentions. We do a light cleanup: remove junk, remove exact duplicates, but don't over-polish it (don't fix grammar, don't translate slang). The assignment literally describes the data as "noisy and imperfect" on purpose — that's the test. We just remove the stuff that would actively confuse our system, like 10 copies of the same spam complaint skewing what looks "common."

If we look at the raw data from Customer Support on Twitter, the mess looks like:
1. Twitter noise: @AmazonHelp, #fail, links, emojis, retweet markers, etc.
** The problem **: if we leave @AmazonHelp and https://t.co/xyz123 in the text, our retrieval step (which measures "similarity of meaning") gets thrown off bytwo completely different complaints that both happen to mention @Amazonhelp will look artificially similar just because they share that token.
** What we do **: we keep two versions of every tweet -- a clean version used for matching/embedding/classifying, and the original raw text used when we actually display the message or generate a reply.
** Simpler alternative **: would be to don't bother keeping two version, just clean everything once and use the clean version everywhere.
** Trade-off **: simplet, but we lose the original text for display and generation. We want to keep the original text for display and generation, so we keep two versions.
2. Duplicate complaints/spam: the same customer tweeting the same complaint multiple times, or multiple customers tweeting
** The problem **: the same person spamming the same complaint multiple times creates two issues: (1) it skews the retrieval system to think that complaint is more common than it really is, and (2) it makes the eval set evaluation unfair because the same complaint appears multiple times.
** What we do **: near-duplicate detection -- if two tweets are from the same conversation thread are almost identical text, we keep just one.
** Simpler alternative **: exact-match deduplication only -- only remove tweets that are character-for-character identical.
** Trade-off **: exact-match is easier to implement and fully deterministic, but misses near-duplicates like same complaint with a different typo or slightly different wording. We want to remove near-duplicates, so we implement a near-duplicate detection step.
3. Redirect Replies: sometimes the brand will reply with "please DM us" or "please call us" instead of actually resolving the issue. We want to keep those, but we don't want to treat them as a solution.
** The problem **: this is the one that matters most for me, A big fraction of "brand replies" in the dataset aren't real resolutions -- they're just "please DM us" redirects. If we let our retrieval system treat those as "how the brand solved this", every generated reply will just learn to say "please DM us" instead of actually solving the customer's problem.
** What we do **: we specifically detect and set aside these redirect-only replies (they follow a predictable pattern -- short, generic, contains "DM"/"direct message"/"call us"/"contact us"). We still keep the customer's original complaint (useful for intent classification and for the eval set), but we don't use "please DM us" as the "here's how the brand/we solved this" example for retrieval or training.
4. Bot/auto replies: some brand accounts auto-reply with a canned message like "Thanks for reaching out! We'll get back to you soon." We want to keep those, but we don't want to treat them as a solution.
5. Non-English tweets: a small number, depending on brand
6. Very short/empty-after-cleaning tweets: just "@AmazonHelp" or "Thanks!" or "OK" — not useful for training a model to generate replies.
** The problem **: after we strip mentions/links/emojis, some tweets are basically empty with nothing left to classify or embed meaningfully.
** What we do **: we drop anything under a small length threshold (something like a handful of words) once cleaned, since there is no real signal left to work with.
** Simpler alternative **: keep everything, let the model deal with short inputs.
** Trade-off **: keeping them costs nothing computationally but they pollute both the eval set (nothing meaningful to label) and the retrieval index (a near-empty vector doesn't represent a real issue).

While polishing or cleaning what we deliberately do NOT clean is:
- We don't fix grammar/spelling. Because the whole point is to turn a messy real-world dataset into a working system. If we quietly clean up all the typos, we are no longer solving the actual challenge, we are dodging it.
- We don't translate non-English tweets.
- We don't remove profanity/anger/emotion. Customers being upset is a real and important signal for the routing decision later (for example: a very angry tweet might be more escalation-worthy). Cleaning that "tone" out will hurt the escalation logic.

ORDER OF OPERATIONS :
load -> filter_to_brand -> reconstruct_threads -> clean -> subsample -> save

Cleaning happens after we've filtered to one brand and reconstructed the customer->reply thread pairs, but before the retrieval index or sample the eval set. That order matters because 
-- If we cleaned before thread reconstruction, we might damage the ID fields we need for pairing.
-- If we cleaned after subsampling, we might accidentally drop a tweet that was in the eval set which would waste the labelling work.

# Storage
** Decision **: Instead of using SQL, NoSQL, or a vector database, we just store the cleaned data in flat files (JSON) for the cleaned/subsampled dataset, for the eval set, and for the evaluation outputs.
** Why **: Data volume after brand-filtering + subsampling is small enough to fit comfortably in memory, and we don't need the complexity of a database. We can just load the JSON into memory and work with it directly.
** Simpler alternative **: keep everything is a single in-memory, no intermediate files at all
** Trade-off **: Pure in-memory is simpler, but then we lose the ability to easily inspect or share the data, and we can't inspect what happened "post-cleaning, pre-retrieval" data without rerunning everything. We want to be able to inspect the data at each stage, so we keep intermediate files.

# Retrieval -- The "similar past case" search
** Decision **: This is the heart of "grounded in how the brand has historically resolved similar issues." What we do is, we embed each historical (customer complaint->brand resolution) pair using a standard sentence-embedding model and store vectors in a flat numpy array / small FAISS index and at inference time embed the incoming message and take top-k nearest historical pairs by cosine similarity.
** Why **: This is the mechanism that makes "grounded in how the brand has historically resolved similar issues" true and auditable, therefore we can always show which past thread(s) informed a given reply. We do this by converting text into number-vectors (embeddings) that capture meaning, not just exact words -- so "my package never arrived" and "where's my order" get recognized as basically the same problem even through they share almost no words.
** Simplest reasonable alternative **: TF-IDF / BM25 keyword retrieval instead of embeddings.
** Trade-off **: TF-IDF is simpler and faster to implement -- no embedding model dependency, fully deterministic, but it is weaker on semantic similarity (typo heavy tweet text). Although we can keep BM25 as a documented fallback/comparison point.
** Simplest possible alternative (rejected) **: A hosted vector DB
** Trade-off **: Adds an external service dependency and account/API key setup, which hurts the 15-minute reproducibility requirement.

# Agent Design
message 
   │
   ▼
1. What is this about? -> classify intent (refund, shipping, account, etc.)
   │
   ▼
2. Retrieve similar resolved cases
   │
   ▼
3. Generate a reply grounded in the retrieved cases
   │
   ▼
4. Routing Decision: can the generated reply be sent automatically, or does a human need to review it first?

We deliberately do these as separate, simple steps rather than one clever all-in-one AI call because if the final reply is bad, we want to know which step broke -- was the intent wrong? Did retrieval find nothing relevant? Or was the writing step the problem? If it's all one black-box call, we can't tell, fix or explain it. 
Separate steps = easier to test, easier to explain, easier to fix.

** Simplest Alternative **: A single LLM call that does classification, retrieval - aware generation and routing decision all in one.
** Trade-off **: simpler, but we lose the ability to test and debug each step separately -- if the reply is bad, we can't tell which step broke - if it's an intent classification failure, a retrieval failure, or a generation failure.

# How we talk to the AI model (prompting + calling the API)
Each of those 3-4 steps gets its own small, clear instruction (prompt) - not one giant instruction trying to do everything at once. 
Prompt template contains:
- a short fixed instruction (e.g. "classify the intent of this tweet")
- decision criteria (e.g. "if you can't tell the intent, say 'unknown'")
- structured output format (e.g. JSON with a single field "intent")
And every time we call the AI, we save the response to a local file first. Next time we need that exact same input, we reuse the saved answer instead of calling the AI again.
Caching matters here because the whole thing must be reproducible in under 15 minutes. If every rerun means paying for and waiting on hundreds of live AI calls, that's slow and it might even give slightly different answers each time. Caching makes reruns instant, free and consistent. 

** Simplest Alternative **: One giant prompt that does everything at once.
** Trade-off **: simpler, but we lose the ability to test and debug each step separately

# Building the Eval Set
Before we can say the system is "good", we need to know what "correct" looks like. So we personally label 150-250 real examples ourselves -- the true intent, whether it should've been escalated, and why.
We don't just grab random examples, we deliberately sample a mix of "easy" and "hard" cases. The easy cases are the ones that are clearly resolved by the brand's historical replies, and the hard cases are the ones that are ambiguous or tricky. This way, we can see how well the system does on both ends of the spectrum.

# Checking the AI's work (LLM interaction)
Since we can't manually read every single reply the system generates, we use another AI call to grade each reply, to check is it grounded in the past case? Is the tone right? Is it actually useful?
A separate judge prompt that scores each generated reply against explicit rubric dimensions -- (e.g. "is the reply grounded in the retrieved past cases?" and "is the tone appropriate for a customer support reply?" and "does it actually answer the customer's question?") -- each scored on a small fixed scale, not one vague overall number. Then we personally label a subset of the same outputs on the same rubric, and report agreement between judge and human.
Now we take a chunk of those graded replies and grade them ourselves too, then compare that do the AI judge and we agree mose of the time?.
The comparison number is the proof the judge can be trusted, instead of just assuming it.

# Metrics
Instead of one single "system is 85% good" number, we break it into pieces how accurate is the intent classification? How good is the escalate/don't escalate decision? How good are the replies, by the judge's breakdown?

Per-Component Metrics:
- Intent classification: accuracy + per-class confusion matrix (vs eval set)
- Routing decision: precision/recall on "should escalate" (vs eval set) where missed escalations (false negatives) and over-escalations (false positives) are reported separately
- Reply quality: judge rubric scores, broken down by dimension, plus judge-human agreement.
- Retrieval: hit rate -- Was it able to retrieve a relevant thread in top-k from the eval set where we know the "right kind" of thread exists?.

** Simplest Alternative **: One single overall metric like "accuracy" or "F1 score" that combines everything into one number.
** Trade-off **: simpler, but we lose the ability to see which part of the system is failing -- if the overall score is low, we can't tell if it's the intent classification, the retrieval, the generation, or the routing decision that is the problem. We want to know which part is failing, so we report per-component metrics.

# Failure Analysis
So the real question here is do we understand where and why the system breaks?. This step is a proof that we understand our own system's weakness well enough to prioritize fixing them.
1. FIND THE DISAGREEMENTS: We've run our system on the eval set, and now we have three comparisons
    - intent classification vs eval set labels
    - Did the routing decision (auto-handle vs escalate) match what we labelled in the eval set?
    - judge scores vs human scores
We pull out everything that disagrees or scores low. This gives us a pile of "problem cases" which we actually read through to understand why the system failed.
2. GROUP THEM BY ROOT CAUSE, NOT SURFACE SYMPTOM: A symptom is "predicted intent was wrong". A root cause is why it was wrong. Two examples can have the same symptom but completely different root causes. We group by root cause, not surface symptom, so we can prioritize the most important fixes.
3. PICK THE TOP 5 CATEGORIES: For each failure category we settle on, we write it up a consistent format:
    - ** Category Name **: What the pattern is?
    - ** A Real Example **: The actual tweet and reply that failed, so we can see it in context.
    - ** Our Hypothesis for why it happens **: What we think is going wrong, and why.

WE DON'T FORCE EXACTLY 5 CLEAN CATEGORIES. 
If we read through the failures and only find 3 real distinct patterns (with the rest being one-offs), we only report 3 plus note that the rest were one-offs rather than manufacturing 2 fake categories just to hit a round number. 

# Testing
Evaluation (eval set + judge) answers "Is the system good at its actual job?" Testing answers more mechanical questions: "does the code do what I expect it to do?" and "does the system break if I change this one thing?".
1. Level 1: Unit tests on pure functions (cleaning rules, sampling logic, JSON-schema validation of LLM output, metric calculations) are fast and there is no API calls as it runs in CLI/locally in seconds.
2. Level 2: Smoke-testing each pipeline stage with a handful of known inputs. For the AI-driven stages (classify, retrieve, generae, route), we can't check for one exact "correct" output as AI outputs vary but we can still catch obvious breakages and for that we pick a small handful of clear cut examples where we are very confident what should happen (like for example an angry refund complaint should clearly get intent "refund", a simple shipping question shouldn't get escalated), and we run them through the pipeline to make sure the outputs are reasonable and don't break the system. This stage acts like a fast trapwire to catch obvious breakages before we run the full eval set.
3. Level 3: Eval Set Regression run: The full evaluation harness itself is the integration test as rerunning it after a change and difference in metrics tells us if a change helped or hurt the system.

# Logging
Two layers of logging:
1. Per-message logs -- Structured, per message logging (JSONL, one line per processed message)
    { 
        "input message": "...",
        "intent classification": "...",
        "retrieved thread IDs": ["...", "...", "..."] + similarity scores,
        "generated reply": "...",
        "routing decision": "..." + ** reason **,
        "judge scores": "..." + rubric breakdown,
        "errors": "..." + any exceptions or failures,
        "retries": "..." + any retries if the AI call failed
    }
2. Run-level logs -- A separate log file for each run of the system, with a summary of the run, including:
    - start time, end time, duration for each stage
    - cache hits vs fresh API calls
    - number of messages that failed at each stage (classification, retrieval, generation, routing)
    - overall metrics from the eval set (intent accuracy, routing precision/recall, judge-human agreement)
    - any notable errors or exceptions that occurred during the run