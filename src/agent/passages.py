"""``ask_book``: retrieval inside one book over the sections the digest already holds.

No new index. A book's digest is its blurb, its chapter-title list and twelve whole chapters
(opening 4, middle 6, ending 2); the multi-vector index cut those into ~37 pieces and embedded them.
This backend rebuilds the same pieces from the book's ``sections_json`` (deterministic chunking),
takes their vectors from the multi-vector index when it is on disk (memory-mapped, so the 3 GB file
costs no RAM) or embeds them on the spot otherwise, and ranks them against the question.

What it cannot answer is as important as what it can: about 4% of a long book is in the digest, so
every result states the coverage and the tool description tells the model to say 「摘要未收录」
rather than guess. The full-book, build-on-demand index is the planned second version.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from src.embed import SupportsEncode, encode_documents, encode_queries
from src.retrieval.multivector import DEFAULT_CHUNK_CHARS, Section, chunk_section

KIND_LABELS = {"blurb": "简介", "titles": "章节目录", "opening": "开头章节", "middle": "中段章节", "ending": "结尾章节", "card": "书卡"}


class SectionsLookup(Protocol):
    def sections(self, novel_id: str) -> list[dict[str, str]] | None:
        """[{"kind": ..., "text": ...}] in digest order, or None when the book is unknown."""


@dataclass
class Passage:
    kind: str
    ordinal: int
    heading: str
    text: str
    score: float


def heading_of(text: str) -> str:
    first = text.strip().split("\n", 1)[0].strip()
    return first if len(first) <= 60 else first[:60]


def coverage_text(sections: list[dict[str, str]]) -> str:
    """「开头章节 4 章（第一章 … 第四章）、中段章节 6 章、结尾章节 2 章、章节目录」, so the model knows what was searchable."""

    parts = []
    for kind in ("opening", "middle", "ending"):
        heads = [heading_of(s["text"]) for s in sections if s.get("kind") == kind]
        if heads:
            span = f"{heads[0]}…{heads[-1]}" if len(heads) > 1 else heads[0]
            parts.append(f"{KIND_LABELS[kind]} {len(heads)} 章（{span}）")
    if any(s.get("kind") == "titles" for s in sections):
        parts.append("章节目录")
    if any(s.get("kind") == "blurb" for s in sections):
        parts.append("简介")
    return "、".join(parts)


class DigestPassages:
    """Passage retrieval over digest chunks; see the module docstring."""

    def __init__(
        self,
        sections_lookup: SectionsLookup,
        embedder: SupportsEncode,
        multi_dir: Path | None = None,
        chunk_chars: int | None = DEFAULT_CHUNK_CHARS,
        cache_size: int = 64,
    ) -> None:
        self.lookup = sections_lookup
        self.embedder = embedder
        self.chunk_chars = chunk_chars
        self.cache_size = cache_size
        self._cache: dict[str, tuple[list[tuple[str, int, str]], np.ndarray]] = {}
        self._lock = threading.Lock()
        self._multi = None
        self.source = "embed"
        if multi_dir is not None and (multi_dir / "faiss.index").exists() and (multi_dir / "sections.json").exists():
            self._multi = _MappedMulti(multi_dir)
            self.source = f"index:{multi_dir.name}"

    def chunks(self, novel_id: str) -> list[tuple[str, int, str]] | None:
        """(kind, ordinal, text) exactly as build_section_table numbered them (cards excluded)."""

        sections = self.lookup.sections(novel_id)
        if sections is None:
            return None
        out: list[tuple[str, int, str]] = []
        pieces: list[Section] = []
        for section in sections:
            sec = Section(str(section.get("kind", "")), str(section.get("text", "")))
            pieces.extend(chunk_section(sec, self.chunk_chars) if self.chunk_chars else [sec])
        for ordinal, piece in enumerate(pieces):
            out.append((piece.kind, ordinal, piece.text))
        return out

    def _vectors(self, novel_id: str, chunks: list[tuple[str, int, str]]) -> np.ndarray:
        if self._multi is not None:
            vectors = self._multi.vectors(novel_id, [(kind, ordinal) for kind, ordinal, _ in chunks])
            if vectors is not None:
                return vectors
        vectors = encode_documents(self.embedder, [text for _, _, text in chunks], batch_size=8, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def _book(self, novel_id: str) -> tuple[list[tuple[str, int, str]], np.ndarray] | None:
        with self._lock:
            cached = self._cache.get(novel_id)
        if cached is not None:
            return cached
        chunks = self.chunks(novel_id)
        if not chunks:
            return None
        vectors = self._vectors(novel_id, chunks)
        with self._lock:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[novel_id] = (chunks, vectors)
        return chunks, vectors

    def ask(self, novel_id: str, question: str, k: int = 3) -> dict[str, Any] | None:
        book = self._book(str(novel_id))
        if book is None:
            return None
        chunks, vectors = book
        query = encode_queries(self.embedder, [question.strip()], batch_size=1, normalize_embeddings=True, show_progress_bar=False)[0]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        scores = (vectors / np.where(norms == 0, 1, norms)) @ query.astype(np.float32)
        order = np.argsort(-scores)[: max(1, k)]
        passages = [Passage(chunks[i][0], chunks[i][1], heading_of(chunks[i][2]), chunks[i][2], float(scores[i])) for i in order]
        sections = self.lookup.sections(str(novel_id)) or []
        return {"coverage": coverage_text(sections), "passages": passages, "source": self.source}


class _MappedMulti:
    """The multi-vector index opened read-only and memory-mapped; vectors by (novel_id, kind, ordinal)."""

    def __init__(self, directory: Path) -> None:
        import faiss

        self.index = faiss.read_index(str(directory / "faiss.index"), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        payload = json.loads((directory / "sections.json").read_text(encoding="utf-8"))
        self.rows: dict[str, dict[tuple[str, int], int]] = {}
        for row, record in enumerate(payload["records"]):
            self.rows.setdefault(str(record["novel_id"]), {})[(str(record["kind"]), int(record["ordinal"]))] = row

    def vectors(self, novel_id: str, keys: list[tuple[str, int]]) -> np.ndarray | None:
        table = self.rows.get(str(novel_id))
        if not table or any(key not in table for key in keys):
            return None  # the index was built from another digest or chunking: embed instead
        return np.stack([self.index.reconstruct(int(table[key])) for key in keys]).astype(np.float32)
