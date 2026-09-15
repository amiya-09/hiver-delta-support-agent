# Citations

Anything I used that wasn't originally mine, per the assignment's rule
on citing what I borrowed.

## Dataset

- `thoughtvector/customer-support-on-twitter` on Kaggle, the raw dataset
  the entire project is built from.

## Pretrained models

- `intfloat/e5-small-v2`, via the sentence-transformers library, the
  embedding model actually used for retrieval.
- `openai/gpt-oss-20b`, served via Groq, the chat model used for
  classification, reply generation, and judging.
- I initially tried Google's `gemini-embedding-001` and `gemini-2.5-flash`
  but switched away from both after hitting quota limits (see decision
  log). Neither is part of the system that actually runs now, but
  `src/llm/gemini_client.py` is kept in the repo as a working, tested
  fallback rather than deleted.

## Libraries

pandas, numpy, sentence-transformers, groq, google-genai, pytest. All
used as-is via pip, nothing modified.

## AI assistance

I used Claude in chat to help design the architecture, review code
before running it for real, debug problems as they came up, and plan
out each phase, and Claude Code to actually write the implementation
from prompts based on that design work. Every actual engineering
decision, what to build, how, and why, is mine and is written up in
decision_log.md. I reviewed the code before ever running it against
real data or spending real API calls, and I did all 240 golden-set
labels and all of my own blind judge scoring myself, no AI involved in
either of those specific judgment calls, since that's the one part of
the project that has to be independent to mean anything. This matches
what the assignment says is allowed: "You may use AI coding assistants
freely. We will ask you to explain and modify your own code live."
