"""Multi-vector book index: one vector per profile section, max-pooled per book.

A profile is ~8,000 characters: a header, the author's synopsis, and up to ten
chapter excerpts. One vector over all of it blends every arc into a single
average. Here the synopsis (with the header, so the title travels with it) and
each excerpt get their own vector, and a book scores as its best section. The
section texts are recovered from ``profile_text`` by the markers
``make_profile_text`` writes, so no corpus pass is needed.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.vector_index import build_faiss_index, load_faiss_index, save_faiss_index, validate_embeddings

EXCERPT_RE = re.compile(r"节选\d+：\n")
BLURB_MARKER = "\n内容简介：\n"
EXCERPTS_MARKER = "\n正文节选：\n"
SECTIONS_FILE = "sections.json"
# digest_v2 sections are whole chapters (median 3,100 characters, up to 8,000). One vector over a whole
# chapter averages several scenes; Qwen3-Embedding-0.6B/4B both degrade past ~1k tokens of Chinese
# prose. Chapters are therefore cut into pieces of at most this many characters at paragraph
# boundaries (2026-10-10 decision, with the move to the 4B model); the pieces keep the chapter's kind.
DEFAULT_CHUNK_CHARS = 1500
HEADING_MAX_CHARS = 60
SENTENCE_BREAK_RE = re.compile(r"(?<=[。！？!?…”」』])|(?<= / )")
INDEX_FILE = "faiss.index"
META_FILE = "index_metadata.json"


@dataclass(frozen=True)
class Section:
    kind: str  # "blurb" (header + synopsis) or "excerpt"
    text: str


@dataclass(frozen=True)
class SectionRecord:
    novel_id: str
    kind: str
    ordinal: int


def split_profile_sections(profile_text: str) -> list[Section]:
    """Recover the sections ``make_profile_text`` joined; a profile without markers is one section."""

    text = profile_text.strip()
    if not text:
        return []
    head, sep, tail = text.partition(EXCERPTS_MARKER)
    title_line = head.split("\n", 1)[0].strip()
    sections = [Section("blurb", head.strip())]
    if sep:
        pieces = [piece.strip() for piece in EXCERPT_RE.split(tail) if piece.strip()]
        prefix = f"{title_line}\n" if title_line.startswith("标题：") else ""
        sections.extend(Section("excerpt", prefix + piece) for piece in pieces)
    return sections


def split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Cut one over-long paragraph at sentence ends (or the ' / ' of a title list); hard cut as a last resort."""

    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for sentence in (x for x in SENTENCE_BREAK_RE.split(paragraph) if x):
        while len(sentence) > max_chars:  # a sentence longer than the budget: hard cut
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + len(sentence) > max_chars:
            pieces.append(current)
            current = ""
        current += sentence
    if current:
        pieces.append(current)
    return [piece.strip() for piece in pieces if piece.strip()]


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split ``text`` into pieces of at most ``max_chars`` at paragraph boundaries, sized about equally
    (so a 3,100-character chapter becomes two pieces of ~1,550 rather than 1,500 + 100)."""

    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    units: list[str] = []
    for paragraph in text.split("\n"):
        if paragraph.strip():
            units.extend(split_long_paragraph(paragraph.strip(), max_chars))
    total = sum(len(unit) + 1 for unit in units)
    wanted = max(1, math.ceil(total / max_chars))
    target = total / wanted
    pieces: list[str] = []
    current: list[str] = []
    length = 0
    for unit in units:
        full = current and length + len(unit) + 1 > max_chars
        balanced = current and len(pieces) < wanted - 1 and length >= target
        if full or balanced:
            pieces.append("\n".join(current))
            current, length = [], 0
        current.append(unit)
        length += len(unit) + 1
    if current:
        pieces.append("\n".join(current))
    return pieces


def chunk_section(section: Section, max_chars: int) -> list[Section]:
    """A section longer than ``max_chars`` becomes several of the same kind. Digest chapter sections start
    with the chapter title on its own line; that heading is repeated on every piece so each vector still
    knows which chapter it came from."""

    text = section.text.strip()
    if len(text) <= max_chars:
        return [section] if text else []
    first, sep, rest = text.partition("\n")
    heading = first.strip() if sep and len(first.strip()) <= HEADING_MAX_CHARS else ""
    body = rest if heading else text
    budget = max_chars - (len(heading) + 1 if heading else 0)
    if budget < max_chars // 2:  # heading too long to repeat: treat it as body
        heading, body, budget = "", text, max_chars
    pieces = chunk_text(body, budget)
    prefix = f"{heading}\n" if heading else ""
    return [Section(section.kind, prefix + piece) for piece in pieces]


def build_section_table(
    profiles: pd.DataFrame, card_texts: dict[str, str] | None = None, chunk_chars: int | None = DEFAULT_CHUNK_CHARS
) -> tuple[list[str], list[SectionRecord]]:
    """One row per section. A digest table carries its sections explicitly (``sections_json``); an old
    profile table is split by its markers. With ``card_texts`` (novel_id -> card text) a book also gets
    a "card" section, prefixed with its title so title queries still land on it. ``chunk_chars`` cuts
    sections longer than that into pieces of the same kind (None or 0 keeps sections whole)."""

    texts: list[str] = []
    records: list[SectionRecord] = []
    explicit = "sections_json" in profiles.columns
    for row in profiles.itertuples(index=False):
        novel_id = str(row.novel_id)
        if explicit:
            sections = [Section(str(s["kind"]), str(s["text"])) for s in json.loads(str(row.sections_json))]
        else:
            sections = split_profile_sections(str(row.profile_text))
        card = (card_texts or {}).get(novel_id)
        if card:
            sections.append(Section("card", f"标题：{row.title_guess}\n{card}"))
        if chunk_chars:
            sections = [piece for section in sections for piece in chunk_section(section, chunk_chars)]
        for ordinal, section in enumerate(sections):
            texts.append(section.text)
            records.append(SectionRecord(novel_id=novel_id, kind=section.kind, ordinal=ordinal))
    return texts, records


def section_stats(records: list[SectionRecord]) -> dict[str, float]:
    """How many sections each book got. A multi-vector build over profiles that
    split into one section each is a single-vector build in disguise, which is
    what happened on 2026-10-08 against a profile table written by an older
    ``make_profile_text`` with different markers."""

    counts: dict[str, int] = {}
    for record in records:
        counts[record.novel_id] = counts.get(record.novel_id, 0) + 1
    if not counts:
        return {"books": 0, "sections": 0, "mean_per_book": 0.0, "share_single": 0.0}
    return {
        "books": len(counts),
        "sections": len(records),
        "mean_per_book": round(len(records) / len(counts), 2),
        "share_single": round(sum(1 for c in counts.values() if c == 1) / len(counts), 4),
    }


def synopsis_preview(profile_text: str, preview_chars: int = 300) -> str:
    """The author's synopsis (plus author line), not the profile header.

    The first 300 characters of a profile are 标题 / 作者 / 长度 / 章节数 and then
    the synopsis; a preview cut there spends most of its budget on metadata the
    agent can read from the title anyway (seen in the first real trajectory,
    2026-10-08). Falls back to the raw start when there is no synopsis marker.
    """

    text = profile_text.strip()
    head, sep, rest = text.partition(BLURB_MARKER)
    if not sep:
        return text[:preview_chars]
    author = next((line for line in head.split("\n") if line.startswith("作者：")), "")
    body = rest.partition(EXCERPTS_MARKER)[0].strip()
    prefix = f"{author}\n" if author else ""
    return (prefix + body)[:preview_chars]


def make_book_meta(profiles: pd.DataFrame, preview_chars: int = 300) -> dict[str, dict[str, str]]:
    return {
        str(row.novel_id): {"title_guess": str(row.title_guess or ""), "profile_text_preview": synopsis_preview(str(row.profile_text), preview_chars)}
        for row in profiles.itertuples(index=False)
    }


class MultiVectorIndex:
    def __init__(self, index: Any, records: list[SectionRecord], meta: dict[str, dict[str, str]]) -> None:
        if index.ntotal != len(records):
            raise ValueError(f"index has {index.ntotal} vectors but {len(records)} section records")
        self.index = index
        self.records = records
        self.meta = meta

    @classmethod
    def build(cls, embeddings: np.ndarray, records: list[SectionRecord], meta: dict[str, dict[str, str]]) -> "MultiVectorIndex":
        return cls(build_faiss_index(validate_embeddings(embeddings)), records, meta)

    @property
    def book_count(self) -> int:
        return len(self.meta)

    def search_vector(self, query_embedding: np.ndarray, k: int, oversample: int = 8) -> list[dict[str, Any]]:
        """Top-k books by their best-matching section; ``oversample`` sections are fetched per book wanted."""

        if k <= 0 or self.index.ntotal == 0:
            return []
        vector = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        if vector.shape[1] != self.index.d:
            raise ValueError(f"Query dim {vector.shape[1]} does not match index dim {self.index.d}")
        fetch = min(max(k * oversample, k), self.index.ntotal)
        scores, rows = self.index.search(vector, fetch)
        best: dict[str, dict[str, Any]] = {}
        for score, row in zip(scores[0], rows[0]):
            if row < 0:
                continue
            record = self.records[int(row)]
            if record.novel_id in best:
                continue
            meta = self.meta.get(record.novel_id, {})
            best[record.novel_id] = {
                "score": float(score),
                "novel_id": record.novel_id,
                "title_guess": meta.get("title_guess", ""),
                "profile_text_preview": meta.get("profile_text_preview", ""),
                "section_kind": record.kind,
                "section_ordinal": record.ordinal,
            }
            if len(best) >= k:
                break
        ranked = sorted(best.values(), key=lambda item: -item["score"])
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        return ranked

    def save(self, directory: Path, metadata: dict[str, Any] | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        save_faiss_index(self.index, directory / INDEX_FILE)
        payload = {"records": [asdict(record) for record in self.records], "meta": self.meta}
        (directory / SECTIONS_FILE).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if metadata is not None:
            (directory / META_FILE).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "MultiVectorIndex":
        payload = json.loads((directory / SECTIONS_FILE).read_text(encoding="utf-8"))
        records = [SectionRecord(**record) for record in payload["records"]]
        return cls(load_faiss_index(directory / INDEX_FILE), records, payload["meta"])
