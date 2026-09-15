# Decision Log

Not every decision I made on this project was obvious going in, so here's
a running list of the ones I had to actually think about, roughly in the
order I made them. Some of these came from things going wrong.

## Picking a brand

I looked at AppleSupport, AmazonHelp, and a couple of telecom accounts
before settling on Delta. Amazon had the most volume by far but it's
basically a catch-all for a marketplace, not one consistent kind of
support conversation, and some of it wasn't even in English. The telecom
ones (Comcast, T-Mobile) were almost entirely "please DM us your account
number," barely any real troubleshooting content in public. Apple's
replies were clean and single-turn, but everything was roughly the same
stakes, which would've made the escalation logic kind of pointless to
design. Delta had good volume (42k+ replies), actual back-and-forth
conversations, and a real mix of "this is fine" vs "this needs a human"
situations, which is exactly what I needed since escalation is one of
the three things the assignment asks for.

I also decided to pair up customer messages with the immediate next
reply rather than trying to find one "final" resolution per thread. It's
simpler and it captures more of the back-and-forth than picking a single
endpoint would.

One thing I added almost as an afterthought that turned out to matter a
lot: a flag for whether a customer message is actually the start of a
conversation, or just a reply buried in the middle of one, like someone
tweeting back a confirmation number. About 30% of what I initially
pulled out as "customer messages" turned out to be these mid-thread
fragments. I only use the real opening messages for classification and
for building the golden set.

## Cleaning the data

I kept cleaning pretty minimal. Stripped @mentions and links so they
don't mess with matching, dropped near-duplicate spam, and excluded pure
"please DM us" replies from the retrieval pool since they're not actual
resolutions. I didn't touch spelling, didn't try to detect sarcasm, and
didn't bother with non-English tweets. The assignment is literally about
handling messy real data, so over-cleaning it felt like cheating the
point of the exercise.

Also worth noting: I originally guessed DM-redirects made up something
like 45% of replies, based on a tiny sample across several brands early
on. Once I actually measured it on the real Delta data it was 18.4%.
Good reminder not to trust small early samples too much.

## Intents and escalation

I built the 8 intent categories myself by reading through 100 real
Delta messages and sorting them by hand rather than using an existing
labelled dataset like Banking77, since that one's banking-specific
anyway and wouldn't have mapped cleanly onto airline complaints.

For escalation, I went with a fixed set of 5 ordered rules instead of
training a model on it. Two reasons: I don't have any escalation labels
until the golden set exists, and if I trained on the same data I'm
using to evaluate, that's circular and proves nothing. Rules also mean
the "reason" the system gives for escalating is the actual reason it
escalated, not something made up after the fact to sound plausible.

I put the safety/risk keyword check first in the rule order, before
anything about grounding confidence or classification confidence. The
reasoning is that knowing how to answer something well is a totally
different question from whether a bot should be the one answering it at
all. A message can be perfectly clear and well-grounded and still need
a human if it's about discrimination or safety.

## Embeddings and retrieval

Started with Gemini for both chat and embeddings since it meant one
provider, one API key. Ended up switching embeddings off Gemini after
hitting its daily quota partway through building the retrieval pool,
more on that below.

Switched to a local model (e5-small-v2, via sentence-transformers)
instead of any hosted embedding API, so there was no more quota to worry
about for that part of the pipeline. The asymmetric query-vs-document
thing I wanted still works, just via text prefixes ("query: " /
"passage: ") instead of an API parameter.

For actually searching the retrieval pool, I just did brute-force cosine
similarity in numpy rather than pulling in FAISS or a vector database.
At 2,500 vectors that's basically instant, and a whole vector DB felt
like overkill for something this small.

Found a genuinely dumb bug while building the golden set's risk-signal
sample. My keyword match for "sue" (trying to catch legal threats) was
matching as a substring inside the word "issue." About 30 of my first 40
"risk signal" candidates turned out to just be regular complaints that
happened to contain the word "issues." Fixed it by switching every
keyword match to use word boundaries instead of plain substring
matching, since the same thing could've been happening with other
keywords too.

Also ran into a weird one on the numerical side. NumPy on my Mac was
occasionally returning garbage (NaN/Inf) from a similarity calculation
that should've been completely fine. Turned out to be a known issue with
Apple's Accelerate framework as NumPy's BLAS backend, not anything wrong
with my code or data. I couldn't reliably reproduce it (it happened once,
then not again across 100 test queries), so instead of trying to force a
full NumPy rebuild against OpenBLAS, I just added a check that raises an
error immediately if it ever happens again, rather than silently
returning a corrupted ranking.

## Building the golden set

I sampled it in a stratified way rather than pure random. Made sure
every intent category had a decent floor, and specifically went out of
my way to pull in extra examples that hit the hard-escalate keywords,
since those are genuinely rare in the data and a random sample wouldn't
have given me enough to actually test that rule.

I labelled all 240 examples myself by hand. No AI assistance on the
actual judgment calls, what's a message's intent, does it need
escalation, why. If I'd let a model help with that, I'd basically be
checking my system against another AI's opinion instead of against
something independent, which defeats the whole point of having a golden
set in the first place.

Interesting thing I noticed once I'd labelled everything: my own
judgment diverged quite a bit from what the keyword-based sampler had
guessed. For example it sampled 20 examples into "booking change" but
only 13 of those actually were, once I read them properly. Sampled 20
into "praise" but ended up with 48 real praise messages once I looked
more broadly. Makes sense, the sampler was always just a rough sort to
get a diverse pool, not something meant to be accurate.

One real gap I have to be upfront about: I only ended up with one
example each testing the "weak grounding" and "ambiguous classification"
escalation rules. Not enough to say anything statistically meaningful
about those two specifically, so anything I report there should be read
as anecdotal.

While labelling I also ran into a couple of edge cases I hadn't
planned for. A couple of tweets that looked like jokes or trolling
rather than real complaints, and one message from someone who said
Delta had fired them, so an employee complaint, not a customer one.
Added notes to the labelling guide on how to handle both rather than
inventing new categories for two anecdotal cases.

## Classification (Phase 3)

Gemini's chat model (gemini-2.5-flash) hit a daily cap of 20 requests
for this specific project, way below the normal published free tier. I
never got a clear explanation why. Rather than sit around waiting on
that, I switched the classification/generation side over to Groq, which
didn't have that problem. Kept embeddings on the local model since that
part was already fine.

Once I moved to Groq I hit a different kind of limit, 200,000 tokens
per day, not a request count. Since my classification prompt has all 8
category definitions plus a few paragraphs of rules, sent in full on
every single call, one full pass over the 240-example golden set eats a
big chunk of a day's budget by itself. This is why I ended up building a
separate "targeted recheck" tool instead of just re-running the whole
eval every time I tweaked the prompt, more on that below.

Also found a bug in my own retry logic. It was treating literally any
error as a rate limit and waiting to retry, which meant when I hit a
completely unrelated 404 "model not found" error, it sat there retrying
for 4 minutes for no reason, since retrying that specific error was
never going to fix anything. Fixed it to only retry on actual rate-limit
errors and fail immediately on anything else.

Ran into a small but annoying model quirk too. The model would
sometimes return the literal string "null" instead of JSON's real null
for the optional second_choice field. Instead of treating that as
broken output, I just normalize it to a real None before validating,
since it's obviously what the model meant.

The targeted-recheck tool was probably the most useful thing I built out
of necessity. Instead of re-running all 240 examples every time I wanted
to test a prompt change, it only re-runs the examples that were
previously wrong (to see if they're fixed) plus a random sample of
previously-correct ones (to check I didn't break anything). Cuts the
cost of testing one idea from about 240 calls down to about 50.

The actual prompt fix I tried, adding a rule that says match by topic
and not just how the message is phrased, came directly from reading the
model's own stated reasoning on wrong answers. It kept explicitly naming
the right topic (for example, "this is about SkyMiles") and then picking
the wrong, more generic category anyway because of how the sentence was
phrased. So I wrote a rule specifically targeting that pattern rather
than guessing.

## Reply generation (Phase 4)

For generation I reused basically everything I'd already built and
tested for classification rather than writing it from scratch again.
Same disk caching pattern, same retry logic, same "retry once on bad
JSON, then give up loudly" behavior. Once I noticed I was about to copy
the rate limit retry code a second time I pulled it out into its own
shared module instead of duplicating it, since at that point it was
clearly going to be needed a third time too (judging, later).

I made the model return which of the retrieved examples it actually
used to write the reply, not just the reply text. The whole point of
"grounded" is that I can check it, not just take the model's word for
it, so if it names an example that isn't actually in the list I gave it,
that's a hard error, not something to quietly ignore.

I also added a check that the reply field itself isn't empty or
missing, which sounds obvious but I'd only validated the grounding field
at first and completely skipped checking the actual reply text was
there. Caught that by holding myself to the same standard I already
used for the intent field back in Phase 3.

Only tested this on a small sample (10 examples) rather than the full
golden set, partly because of the Groq budget situation and partly
because I wanted to actually read the output by hand before trusting it
at scale. Good thing I did. Every one of the 10 retrieved examples came
back with a similarity score somewhere between 0.83 and 0.90, no matter
how relevant the message actually was. That's a real problem because my
prompt tells the model to skip grounding if nothing retrieved is
actually relevant, but if retrieval never returns a genuinely low score,
that instruction never gets tested. And it showed up as an actual
mistake: one generated reply lifted a phone number straight from a
retrieved example about a completely different customer's wifi problem
and handed it to someone with an unrelated issue, all while the
similarity score looked totally normal. That's the kind of thing that
looks fine on paper and is actually a real problem in production.

## Routing and escalation (Phase 5)

I set the "weak grounding" escalation threshold to 0.80, but given what
I'd already found in Phase 4 about scores clustering so tightly, I wrote
directly into the code that this threshold probably won't fire very
often with the current embedding model. Better to be honest about that
upfront than pretend a number I basically guessed is doing more work
than it actually is.

When I went to actually test routing against the golden set, I realized
the golden set and the retrieval pool were both sampled from the same
pile of 34,000 pairs, so some golden examples are also sitting inside
the retrieval pool itself. If I queried retrieval with one of those
messages, it would just find itself and score almost perfectly, which
would make the routing numbers look better than they really are. Had to
add a check that filters out an example matching its own ID before
computing the similarity score, and it turned out about 1 in 10 golden
examples had this problem. I also had to make that filtering loop
instead of just grabbing one extra result, in case the same ID somehow
showed up more than once in the pool.

## Two things that weren't tied to a specific phase

Partway through I realized my .gitignore had a rule that was too broad.
It was meant to keep generated CSV files out of version control, but it
also happened to match the golden set file itself, the 240 examples I'd
spent real hours labelling by hand. If I hadn't caught that before
committing, that work could have just never made it into the repo at
all. Fixed the rule to be specific about which folders to ignore
instead of matching by file extension everywhere.

Similarly, near the end I realized that reproducing anything past Phase
1 technically required the original 492MB raw dataset, which I never
committed since it's too big and the assignment didn't ask for it. But
without it, someone cloning the repo fresh couldn't get past the very
first step. Fixed that by committing the cleaned, already-processed
pairs file directly instead, since that's the one thing every later
phase actually depends on and it's small enough to just include.
