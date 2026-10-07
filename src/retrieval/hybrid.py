"""Searchers with one shape, and reciprocal-rank fusion to combine them.

Every searcher returns the rows ``search.semantic_search`` returns (rank, score,
novel_id, title_guess, profile_text_preview), so the agent's ``search_books``
tool and the benchmark are indifferent to which configuration sits behind them.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

from src.embed import SupportsEncode, encode_queries
from src.retrieval.bm25 import BM25Index
from src.retrieval.multivector import MultiVectorIndex
from src.search import semantic_search
from src.vector_index import load_faiss_index, load_id_map


class BookSearcher(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[dict[str, Any]]: ...


class SingleVectorSearcher:
    """The v1 index: one vector per book."""

    def __init__(self, model: SupportsEncode, index: Any, id_map: dict[int, dict[str, str]], name: str = "dense_single") -> None:
        self.model, self.index, self.id_map, self.name = model, index, id_map, name

    @classmethod
    def load(cls, model: SupportsEncode, directory: Path, name: str = "dense_single") -> "SingleVectorSearcher":
        return cls(model, load_faiss_index(directory / "faiss.index"), load_id_map(directory / "novel_id_map.json"), name)

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        return semantic_search(query, self.model, self.index, self.id_map, top_k=k)


class MultiVectorSearcher:
    def __init__(self, model: SupportsEncode, index: MultiVectorIndex, name: str = "dense_multi", oversample: int = 8) -> None:
        self.model, self.index, self.name, self.oversample = model, index, name, oversample

    @classmethod
    def load(cls, model: SupportsEncode, directory: Path, name: str = "dense_multi") -> "MultiVectorSearcher":
        return cls(model, MultiVectorIndex.load(directory), name)

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        query = query.strip()
        if not query or k <= 0:
            return []
        vector = encode_queries(self.model, [query], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
        return self.index.search_vector(vector[0], k, oversample=self.oversample)


class BM25Searcher:
    def __init__(self, index: BM25Index, meta: dict[str, dict[str, str]], name: str = "bm25") -> None:
        self.index, self.meta, self.name = index, meta, name

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        rows = []
        for rank, (novel_id, score) in enumerate(self.index.search(query, k), start=1):
            meta = self.meta.get(novel_id, {})
            rows.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "novel_id": novel_id,
                    "title_guess": meta.get("title_guess", ""),
                    "profile_text_preview": meta.get("profile_text_preview", ""),
                }
            )
        return rows


def reciprocal_rank_fusion(rankings: list[list[str]], weights: list[float] | None = None, k: int = 60) -> list[tuple[str, float]]:
    """RRF: each list contributes weight / (k + rank); ids are returned by fused score."""

    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking")
    fused: dict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            fused[item] += weight / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


class HybridSearcher:
    """Fuse several searchers by RRF; each is asked for ``depth`` candidates."""

    def __init__(self, searchers: list[BookSearcher], weights: list[float] | None = None, depth: int = 100, rrf_k: int = 60, name: str | None = None) -> None:
        if not searchers:
            raise ValueError("at least one searcher")
        self.searchers = searchers
        self.weights = weights or [1.0] * len(searchers)
        self.depth = depth
        self.rrf_k = rrf_k
        self.name = name or "hybrid(" + "+".join(s.name for s in searchers) + ")"

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        if k <= 0:
            return []
        rows_by_id: dict[str, dict[str, Any]] = {}
        rankings: list[list[str]] = []
        for searcher in self.searchers:
            rows = searcher.search(query, max(self.depth, k))
            rankings.append([str(row["novel_id"]) for row in rows])
            for row in rows:
                rows_by_id.setdefault(str(row["novel_id"]), row)
        fused = reciprocal_rank_fusion(rankings, self.weights, self.rrf_k)[:k]
        output = []
        for rank, (novel_id, score) in enumerate(fused, start=1):
            base = rows_by_id[novel_id]
            output.append({**base, "rank": rank, "score": float(score), "fused_from": [s.name for s in self.searchers]})
        return output
