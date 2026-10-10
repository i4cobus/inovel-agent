"""A searcher that lets the book card filter before top-k is cut.

The agent can only remove candidates after retrieval; a card can remove them before. When the
query names a genre or sub-genre of the card vocabulary, the inner searcher is over-sampled and
books whose card agrees are moved ahead (sub-genre match before genre match before the rest), so
the top-k is filled with agreeing books first and back-filled with the remainder. Queries that name
no genre pass through untouched. The candidate-supply bench runs it as one more configuration;
whether it earns its place is what the bench decides.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.preferences import parse_preference_query
from src.retrieval.supply import query_element_targets, query_genre_targets


class CardFilteredSearcher:
    def __init__(self, inner: Any, cards: Mapping[str, Any], name: str | None = None, oversample: int = 5) -> None:
        self.inner, self.cards, self.oversample = inner, cards, max(1, oversample)
        self.name = name or f"{getattr(inner, 'name', 'searcher')}+card"

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        positives = parse_preference_query(query).positive_terms or [query]
        genres, subgenres = query_genre_targets(positives)
        if not genres and not subgenres:
            return self.inner.search(query, k)
        elements = query_element_targets(positives)
        rows = self.inner.search(query, k * self.oversample)
        scored = []
        for position, row in enumerate(rows):
            card = self.cards.get(str(row.get("novel_id", "")))
            match, elements_hit = "", 0
            if card is not None:
                if card.subgenre and card.subgenre in subgenres:
                    match = "subgenre"
                elif card.genre in genres:
                    match = "genre"
                elements_hit = len(elements & set(card.elements)) if elements else 0
            scored.append((0 if match == "subgenre" else 1 if match == "genre" else 2, -elements_hit, position, {**row, "card_match": match}))
        scored.sort(key=lambda item: item[:3])
        out = [item[3] for item in scored[:k]]
        for rank, row in enumerate(out, start=1):
            row["rank"] = rank
        return out
