# Phase 1 Failure Analysis: Data Pipeline

This phase is mostly deterministic data processing, not model behavior,
so there isn't the same room for subtle errors that later phases have.
Two real findings came out of it, and I'm not padding this out further
just to match the length of later phases.

## Finding 1: My early estimate of DM-redirect frequency was way off

Before I had the full dataset filtered down, I guessed from a small,
mixed-brand sample that around 45% of brand replies were just "please
DM us" redirects with no real resolution content. Once I actually
measured it on the real, full set of 34,435 Delta pairs, the number was
18.4%. The early sample happened to include some telecom brands that
are unusually redirect-heavy, which skewed my first guess. Lesson: a
93-row sample across several different brands isn't a reliable estimate
for one specific brand at real scale.

## Finding 2: A third of "customer messages" weren't real opening complaints

Reading through early samples by hand, I noticed some "customer
messages" were actually just replies buried in the middle of a
conversation, like someone tweeting back a confirmation number rather
than a real complaint. Once I tagged this properly (`is_thread_root`)
and measured it across the full dataset, it turned out to be 30% of
everything I'd pulled out, not a rare edge case. If I hadn't caught this
before building the taxonomy and sampling the golden set, both would
have been quietly polluted with fragments instead of genuine complaints.

## What I didn't find, worth being honest about

I didn't do a rigorous check of whether my duplicate-detection (exact
match on normalized text) is too aggressive or not aggressive enough. It
dropped about 13% of replies as duplicates, which seems reasonable given
how much spam-reposting shows up in real Twitter support data, but I
never specifically verified that against a hand-checked sample the way I
did for the DM-redirect rate. That's a real gap in how carefully I
validated this specific step, not a confirmed problem, just an
unverified assumption.
