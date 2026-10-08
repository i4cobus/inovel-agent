"""What goes into the dense retriever: the positive side of a preference query only.

A dense model has no notion of negation: 「不系统」 pulls system-novels *in*.
Negative terms are the agent's job to enforce after retrieval (check_term /
check_trope), so they are stripped before any embedding is computed. The
benchmark applies the same function so it measures what the tool does.
"""

from __future__ import annotations

from src.preferences import parse_preference_query


def retrieval_query(query: str) -> str:
    """Positive terms joined by spaces; the original text when nothing parses as positive."""

    parsed = parse_preference_query(query)
    if not parsed.positive_terms:
        return query.strip()
    return " ".join(parsed.positive_terms)
