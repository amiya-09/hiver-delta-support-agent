# Golden Set

## What I built

Before I could actually evaluate anything, I needed real ground truth:
150 to 250 messages I'd hand labelled myself with the correct intent
and escalation decision. This is the one part of the whole project that
can't be automated or delegated to an AI, since the entire point of
having it is to be an independent check on the system, not something
the system helped produce.

I didn't want a purely random sample, since a quick read of the data
showed some categories are naturally way more common than others, and
the genuinely risky cases (the ones that should trigger hard escalation)
are rare enough that random sampling probably wouldn't catch enough of
them to actually test that rule. So I built a rough keyword-based
sampler to pull a diverse, deliberately stratified pool of candidates
first, then read and labelled every single one by hand.

## Files

- `src/golden_set/constants.py`, the 8 intent categories and 5
  escalation rule names, imported everywhere else instead of getting
  redefined and risking drift
- `src/golden_set/sample_candidates.py`, the stratified sampler
- `src/golden_set/validate_labels.py`, checks my finished labels for
  consistency once I'm done
- `tests/test_golden_set.py`
- `golden_set_labelling_guide.md`, the reference doc I wrote for myself
  so I wouldn't drift in judgment between labelling session one and
  labelling session five
- `data/golden_set/candidates.csv`, the actual 240 labelled examples

## Edge cases

| Case | How I handle it |
|---|---|
| A keyword bucket has fewer real candidates than its target count | Use everything available, print a warning naming the shortfall |
| A message matches both a risk keyword and a normal category keyword | Goes into the risk-signal bucket only, never counted twice |
| Raw tweet text has commas, quotes, or emoji | Written with Python's actual csv module, not manual string joining, so nothing gets corrupted |
| Spreadsheet booleans come back inconsistent (TRUE, True, 1) | Validator normalizes case and format before checking |
| `true_escalate` is False but an escalation rule is filled in anyway, or the reverse | Flagged as an inconsistency, not silently accepted |
| A reply-quality note is left blank | Flagged, since an empty note defeats the point of having the field |

## What actually happened

While building the sampler I found a genuinely dumb bug that I already
wrote up in the decision log: my "sue" keyword for catching legal
threats was matching as a substring inside the ordinary word "issue,"
which meant most of my first batch of "risky" candidates were actually
just regular complaints. Fixed the matching to use word boundaries
everywhere, not just for that one keyword, since the same mistake could
have been hiding in any of my other keyword lists too.

Labelling itself took a few sittings, working through the CSV in a
spreadsheet with dropdown validation set up so I couldn't accidentally
type an invalid category. Ran into a couple of real scares along the
way, a fill-drag that accidentally duplicated an ID across sixteen rows,
and a filter-versus-edit mixup in Google Sheets, both caught before any
actual labels were affected.

Once everything was labelled and passed validation with zero problems, a
couple of things stood out. My own judgment diverged a lot from what the
keyword sampler had guessed. It sampled 20 examples as "booking change"
and only 13 actually were once I read them properly, and sampled 20 as
"praise" but I ended up finding 48 real ones across the whole set. Makes
sense, since the sampler was only ever meant to build a diverse pool,
not to actually get the label right.

The escalation side came out to 57 examples that should escalate and
183 that shouldn't, with the hard-risk-keyword rule genuinely firing on
20 real cases, not just false positives from the substring bug I'd
already fixed. The one thing I have to be upfront about: the two rarer
escalation rules, weak grounding and ambiguous classification, only got
one real example each. Not enough to say anything statistically
meaningful about either one specifically.

I also ran into two message types I hadn't planned for while reading
through the data: a couple of tweets that read like jokes rather than
real complaints, and one message from someone saying Delta had fired
them, which isn't a customer support situation at all. Rather than
invent new categories for two one-off cases, I added notes to my own
labelling guide on how to handle them consistently if I saw more.
