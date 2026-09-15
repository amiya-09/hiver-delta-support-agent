"""The end-to-end agent: clean -> classify -> retrieve -> generate -> route.

This is the actual agent the whole project has been building toward.
Every stage below is a previously-built, independently-tested module
(Phases 1-5); this function's only job is wiring them together in the
right order and shape.
"""
from src.agent.classify import classify_intent
from src.agent.generate import generate_reply
from src.agent.route import route_message
from src.pipeline.clean import strip_mentions_and_links
from src.retrieval.query import load_index, top_k_similar


def process_message(raw_text, retrieval_index_dir="data/processed/retrieval_index", k=3):
    """Run one customer message through the full agent pipeline.

    Returns one dict with every stage's output -- clean text, the
    classification, the retrieved precedent, the drafted reply, and the
    routing decision -- so a caller (or a human reviewing why the agent
    did what it did) can see the whole chain, not just the final answer.
    """
    # 1. Clean: strip @mentions/links so classification/retrieval/generation
    # see the same normalized text used throughout the rest of the project.
    clean_text = strip_mentions_and_links(raw_text)

    # 2. Classify: which of the 8 intents is this?
    classification = classify_intent(clean_text)

    # 3. Retrieve: find the top-k most similar past resolved complaints.
    embeddings, metadata = load_index(retrieval_index_dir)
    retrieved_examples = top_k_similar(clean_text, embeddings, metadata, k=k)
    top_similarity = retrieved_examples[0]["similarity_score"] if retrieved_examples else 0.0

    # 4. Generate: draft a reply grounded in the retrieved precedent.
    generation = generate_reply(clean_text, classification["intent"], retrieved_examples)

    # 5. Route: decide whether a human needs to review this instead of
    # auto-sending the drafted reply. Keyword matching runs on raw_text
    # (not clean_text), same as the rest of the project's risk detection.
    routing = route_message(raw_text, classification, top_similarity)

    return {
        "raw_text": raw_text,
        "clean_text": clean_text,
        "classification": classification,
        "retrieved_examples": retrieved_examples,
        "generation": generation,
        "routing": routing,
    }
