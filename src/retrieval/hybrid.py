"""Searchers with one shape, and reciprocal-rank fusion to combine them.

Every searcher returns the rows ``search.semantic_search`` returns (rank, score,
novel_id, title_guess, profile_text_preview), so the agent's ``search_books``
tool and the benchmark are indifferent to which configuration sits behind them.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Protocol

import faiss
import numpy as np

from src.config import DEFAULT_INDEX_DIR

from src.embed import SupportsEncode, encode_queries
from src.retrieval.bm25 import BM25Index
from src.retrieval.multivector import MultiVectorIndex
from src.search import semantic_search
from src.vector_index import load_faiss_index, load_id_map


class BookSearcher(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[dict[str, Any]]: ...


def id_selector(rows: Iterable[int]) -> Any:
    """A FAISS search parameter restricting the search to these row ids (exact on a flat index)."""

    return faiss.SearchParameters(sel=faiss.IDSelectorBatch(np.fromiter(rows, dtype=np.int64)))


class SingleVectorSearcher:
    """One vector per book (v1 whole-digest encoding, or the pooled multi-vector index)."""

    def __init__(self, model: SupportsEncode, index: Any, id_map: dict[int, dict[str, str]], name: str = "dense_single") -> None:
        self.model, self.index, self.id_map, self.name = model, index, id_map, name
        self.rows_by_novel: dict[str, int] = {str(meta.get("novel_id", "")): row for row, meta in id_map.items()}

    @classmethod
    def load(cls, model: SupportsEncode, directory: Path, name: str = "dense_single") -> "SingleVectorSearcher":
        return cls(model, load_faiss_index(directory / "faiss.index"), load_id_map(directory / "novel_id_map.json"), name)

    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[dict[str, Any]]:
        if allowed_ids is None:
            return semantic_search(query, self.model, self.index, self.id_map, top_k=k)
        query = query.strip()
        if not query or k <= 0:
            return []
        vector = encode_queries(self.model, [query], batch_size=1, normalize_embeddings=True)
        return self._search_vector(vector, k, allowed_ids)

    def similar(self, novel_id: str, k: int, allowed_ids: set[str] | None = None) -> list[dict[str, Any]]:
        """Books nearest to this book's own vector, the book itself left out."""

        row = self.rows_by_novel.get(str(novel_id))
        if row is None:
            return []
        vector = self.index.reconstruct(int(row)).reshape(1, -1)
        allowed = (allowed_ids if allowed_ids is not None else set(self.rows_by_novel)) - {str(novel_id)}
        return self._search_vector(vector, k, allowed)

    def _search_vector(self, vector: np.ndarray, k: int, allowed_ids: set[str]) -> list[dict[str, Any]]:
        rows = [self.rows_by_novel[n] for n in allowed_ids if n in self.rows_by_novel]
        if not rows or k <= 0:
            return []
        scores, ids = self.index.search(np.ascontiguousarray(vector, dtype=np.float32), min(k, len(rows)), params=id_selector(rows))
        out: list[dict[str, Any]] = []
        for score, faiss_id in zip(scores[0], ids[0]):
            meta = self.id_map.get(int(faiss_id))
            if faiss_id < 0 or meta is None:
                continue
            out.append({"rank": len(out) + 1, "score": float(score), "novel_id": meta.get("novel_id", ""), "title_guess": meta.get("title_guess", ""), "profile_text_preview": meta.get("profile_text_preview", "")})
        return out


class MultiVectorSearcher:
    def __init__(self, model: SupportsEncode, index: MultiVectorIndex, name: str = "dense_multi", oversample: int = 8) -> None:
        self.model, self.index, self.name, self.oversample = model, index, name, oversample

    @classmethod
    def load(cls, model: SupportsEncode, directory: Path, name: str = "dense_multi") -> "MultiVectorSearcher":
        return cls(model, MultiVectorIndex.load(directory), name)

    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[dict[str, Any]]:
        query = query.strip()
        if not query or k <= 0:
            return []
        vector = encode_queries(self.model, [query], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
        return self.index.search_vector(vector[0], k, oversample=self.oversample, allowed_ids=allowed_ids)


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


def load_searchers(
    directory: Path = DEFAULT_INDEX_DIR,
    model: SupportsEncode | None = None,
    label: str | None = None,
    hybrid_depth: int = 100,
) -> list[BookSearcher]:
    """Every searcher an index directory supports: dense (single or multi, autodetected), BM25, and their hybrid.

    ``model`` is required when the directory holds a dense index. The hybrid is
    added only when both a dense and a BM25 index are present.
    """

    label = label or directory.name
    searchers: list[BookSearcher] = []
    if (directory / "faiss.index").exists():
        if model is None:
            raise ValueError(f"{directory} has a dense index; pass the embedding model")
        if (directory / "sections.json").exists():
            searchers.append(MultiVectorSearcher.load(model, directory, name=f"{label}/dense_multi"))
        else:
            searchers.append(SingleVectorSearcher.load(model, directory, name=f"{label}/dense_single"))
    if (directory / "bm25.json").exists():
        meta = json.loads((directory / "book_meta.json").read_text(encoding="utf-8"))
        searchers.append(BM25Searcher(BM25Index.load(directory / "bm25.json"), meta, name=f"{label}/bm25"))
    if not searchers:
        raise FileNotFoundError(f"No index in {directory}: build one with scripts/30_build_book_indexes.py")
    if len(searchers) == 2:
        searchers.append(HybridSearcher(list(searchers), depth=hybrid_depth, name=f"{label}/hybrid_rrf"))
    return searchers


def index_metadata(directory: Path = DEFAULT_INDEX_DIR) -> dict[str, Any]:
    path = directory / "index_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

