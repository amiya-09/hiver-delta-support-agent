# Phase 2 Failure Analysis: Retrieval Index

## Finding 1: A near-miss I caught before it cost anything

Gemini's embedding model distinguishes between embedding something to
be searched (a document) versus embedding the actual search query. If I
hadn't set that distinction correctly, retrieval quality would have
quietly gotten worse with no obvious error telling me so, since the
pipeline would still run fine, it would just return worse matches. I
caught this by actually reading the API's documentation on task types
before running the real 2,500-pair build, not after. This isn't really
a "failure" since nothing actually went wrong, but it's worth recording
as exactly the kind of silent quality problem that's easy to miss if
you don't specifically look for it.

## Finding 2: An intermittent numerical bug I couldn't fully pin down

NumPy on my Mac occasionally returned NaN or infinite values from a
similarity calculation that should have been completely normal. I
traced it to a known issue with Apple's Accelerate framework, which
NumPy uses as its default math backend on Mac. The honest limitation
here: I could reproduce it exactly once, with one specific real query,
but running the same kind of check 100 more times afterward, it never
happened again. I don't actually know how often this would occur in a
longer run or under different conditions. I added a check that raises
an error immediately if it recurs, but that's a safety net, not a fix,
and I don't have a real answer for how frequently this bug would
actually surface at larger scale.

## Finding 3: My retrieval quality check was informal

I validated retrieval by hand on 4 queries I picked myself plus 100
random ones, reading each result and judging whether it looked
topically related. That's useful for catching an obviously broken
system, but it's not a rigorous metric. I never built something like a
recall@k score against a set of known correct matches, so "retrieval
looked fine in my spot checks" is a weaker claim than it might sound
like. This matters because Phase 4 later found that retrieval similarity
scores cluster in a narrow band regardless of true relevance, something
my qualitative checks in this phase weren't set up to catch, since I was
judging topical relevance by eye, not looking at whether the actual
similarity numbers were meaningfully spread out.
