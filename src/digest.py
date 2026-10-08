"""The digest: one text view per novel, weighted to the opening and the ending.

Replaces the profile (2026-10-09). A web novel states its genre, setting,
protagonist and 金手指 in its first chapters and shows whether it collapsed in
its last ones, so the digest spends most of its budget there. Three short
middle windows catch a change of register. The chapter-title list is the
author's own free summary of the whole book: 「第312章 青儿的心意」 says more
about a romance line than any sampled window would.

Everything downstream reads the digest: the multi-vector index embeds its
sections, ``get_profile`` returns it, the book card is extracted from it.
The judge avoids the chapters it used (``digest_chapter_indices``).
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.clean import CleaningStats, clean_novel_text_with_stats
from src.config import DEFAULT_OUTPUT_PATH, PROCESSED_DATA_DIR, resolve_worker_count
from src.profile import extract_blurb, read_text_with_encoding, substantive_chapter_indices, trim_to_sentence
from src.split_chapters import split_chapters

DEFAULT_DIGEST_PATH = PROCESSED_DATA_DIR / "novel_digests.parquet"

OPENING_CHAPTERS = 4
OPENING_CHARS = 1500
ENDING_CHAPTERS = 2
ENDING_CHARS = 1500
MIDDLE_WINDOWS = 3
MIDDLE_CHARS = 600
TITLE_SAMPLES = 100
BLURB_CHARS = 500
MIN_CHAPTERS = 3

ENDING_MARKER_RE = re.compile(r"(全书完|全文完|全本完|大结局|正文完|本书完|（完）|\(完\)|完结)")
NON_NARRATIVE_TITLE_RE = re.compile(r"(番外|后记|感言|作者的话|完本|上架|请假|通知|公告|推荐)")


@dataclass(frozen=True)
class Section:
    kind: str  # blurb | opening | titles | middle | ending
    text: str


@dataclass
class Digest:
    novel_id: str
    title: str
    author: str | None
    char_count: int
    chapter_count: int
    ending_status: str  # 完结标记 | 无标记
    sections: list[Section]
    used_chapter_indices: list[int]

    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)

    def to_row(self) -> dict[str, Any]:
        return {
            "novel_id": self.novel_id,
            "title_guess": self.title,
            "author_guess": self.author,
            "char_count": self.char_count,
            "estimated_chapter_count": self.chapter_count,
            "ending_status": self.ending_status,
            "profile_text": self.text(),
            "sections_json": json.dumps([{"kind": s.kind, "text": s.text} for s in self.sections], ensure_ascii=False),
            "used_chapter_indices": json.dumps(self.used_chapter_indices),
        }


def narrative_indices(chapters: Sequence[Any]) -> list[int]:
    """Substantive chapters whose titles do not mark non-narrative matter (番外, 后记, 感言...)."""

    return [i for i in substantive_chapter_indices(chapters) if not NON_NARRATIVE_TITLE_RE.search(str(getattr(chapters[i], "title", "") or ""))]


def digest_chapter_indices(chapters: Sequence[Any]) -> dict[str, list[int]]:
    """Which chapters the digest reads, by role. Deterministic from the chapter list alone."""

    usable = narrative_indices(chapters)
    if len(usable) < MIN_CHAPTERS:
        return {"opening": [], "middle": [], "ending": []}
    opening = usable[:OPENING_CHAPTERS]
    ending = [i for i in usable[-ENDING_CHAPTERS:] if i not in opening]
    inner = [i for i in usable if i not in opening and i not in ending]
    middle: list[int] = []
    if inner:
        for fraction in (0.25, 0.5, 0.75)[:MIDDLE_WINDOWS]:
            candidate = inner[min(int(fraction * len(inner)), len(inner) - 1)]
            if candidate not in middle:
                middle.append(candidate)
    return {"opening": opening, "middle": middle, "ending": ending}


def sample_titles(chapters: Sequence[Any], indices: list[int], samples: int = TITLE_SAMPLES) -> list[str]:
    titles = [str(getattr(chapters[i], "title", "") or "").strip() for i in indices]
    titles = [t for t in titles if t]
    if len(titles) <= samples:
        return titles
    step = len(titles) / samples
    return [titles[min(int(k * step), len(titles) - 1)] for k in range(samples)]


def ending_status(text: str, tail_chars: int = 3000) -> str:
    return "完结标记" if ENDING_MARKER_RE.search(text[-tail_chars:]) else "无标记"


def make_digest(novel_id: str, title: str, author: str | None, cleaned_text: str) -> Digest:
    chapters = split_chapters(cleaned_text)
    roles = digest_chapter_indices(chapters)
    status = ending_status(cleaned_text)
    header = f"标题：{title}\n" + (f"作者：{author}\n" if author else "") + f"长度：约{len(cleaned_text)}字\n章节数：约{len(chapters)}章\n结局：{status}"
    blurb = extract_blurb(cleaned_text, max_chars=BLURB_CHARS)
    sections = [Section("blurb", header + (f"\n\n内容简介：\n{blurb}" if blurb else ""))]
    used: list[int] = []

    if roles["opening"]:
        for i in roles["opening"]:
            body = trim_to_sentence(chapters[i].text, OPENING_CHARS)
            if body:
                sections.append(Section("opening", f"{chapters[i].title}\n{body}"))
        titles = sample_titles(chapters, narrative_indices(chapters))
        if titles:
            sections.append(Section("titles", "章节目录（抽样）：\n" + " / ".join(titles)))
        for i in roles["middle"]:
            body = trim_to_sentence(chapters[i].text, MIDDLE_CHARS)
            if body:
                sections.append(Section("middle", f"{chapters[i].title}\n{body}"))
        for i in roles["ending"]:
            body = trim_to_sentence(chapters[i].text, ENDING_CHARS)
            if body:
                sections.append(Section("ending", f"{chapters[i].title}\n{body}"))
        used = sorted(set(roles["opening"] + roles["middle"] + roles["ending"]))
    else:
        # No usable chapter structure: character windows in the same proportions.
        n = len(cleaned_text)
        opening = trim_to_sentence(cleaned_text[: OPENING_CHARS * OPENING_CHAPTERS], OPENING_CHARS * OPENING_CHAPTERS)
        if opening:
            sections.append(Section("opening", f"开头：\n{opening}"))
        for fraction in (0.25, 0.5, 0.75):
            start = int(n * fraction)
            body = trim_to_sentence(cleaned_text[start : start + MIDDLE_CHARS], MIDDLE_CHARS)
            if body:
                sections.append(Section("middle", f"中段 {int(fraction * 100)}%：\n{body}"))
        tail = trim_to_sentence(cleaned_text[-ENDING_CHARS * ENDING_CHAPTERS :], ENDING_CHARS * ENDING_CHAPTERS)
        if tail:
            sections.append(Section("ending", f"结尾：\n{tail}"))
    return Digest(novel_id, title, author, len(cleaned_text), len(chapters), status, sections, used)


# ---- corpus pass -----------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerResult:
    row: dict[str, Any] | None
    reason: str  # ok | failed | missing | read_error
    stats: CleaningStats = field(default_factory=CleaningStats)


def build_digest_worker(inventory_row: dict[str, Any]) -> WorkerResult:
    if inventory_row.get("read_status") != "ok":
        return WorkerResult(None, "failed")
    path = Path(str(inventory_row.get("absolute_path", "")))
    if not path.exists():
        return WorkerResult(None, "missing")
    try:
        raw = read_text_with_encoding(path, inventory_row.get("detected_encoding"), allow_lossy=int(inventory_row.get("decode_replacement_chars", 0) or 0) > 0)
    except (OSError, UnicodeError, LookupError, ValueError):
        return WorkerResult(None, "read_error")
    cleaned, stats = clean_novel_text_with_stats(raw)
    title = str(inventory_row.get("title_guess") or inventory_row.get("file_stem") or path.stem)
    author_value = inventory_row.get("author_guess")
    author = None if author_value is None or (isinstance(author_value, float) and pd.isna(author_value)) else str(author_value)
    digest = make_digest(str(inventory_row["novel_id"]), title, author, cleaned)
    return WorkerResult(digest.to_row(), "ok", stats)


@dataclass
class DigestBuildResult:
    frame: pd.DataFrame
    skipped: dict[str, int]
    boilerplate_detected: int
    boilerplate_lines_removed: int


def build_digests(inventory_path: Path = DEFAULT_OUTPUT_PATH, limit: int | None = None, max_workers: int | None = None) -> DigestBuildResult:
    import pyarrow.parquet as pq

    wanted = ["novel_id", "absolute_path", "detected_encoding", "read_status", "decode_replacement_chars", "title_guess", "author_guess", "file_stem"]
    present = set(pq.read_schema(inventory_path).names)
    frame = pd.read_parquet(inventory_path, columns=[c for c in wanted if c in present])
    if limit is not None:
        frame = frame.head(limit)
    rows = frame.to_dict(orient="records")
    workers = resolve_worker_count(max_workers)
    if workers == 1 or len(rows) < 2:
        results = [build_digest_worker(r) for r in rows]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            results = list(pool.map(build_digest_worker, rows, chunksize=1))
    skipped = {"failed": 0, "missing": 0, "read_error": 0}
    kept = []
    detected = removed = 0
    for result in results:
        if result.row is None:
            skipped[result.reason] += 1
            continue
        kept.append(result.row)
        detected += int(bool(result.stats.zxcs_detected))
        removed += int(result.stats.zxcs_lines_removed)
    return DigestBuildResult(pd.DataFrame(kept), skipped, detected, removed)


def load_sections(frame: pd.DataFrame) -> dict[str, list[Section]]:
    """novel_id -> sections, from a digest parquet's sections_json column."""

    out: dict[str, list[Section]] = {}
    for row in frame.itertuples(index=False):
        payload = json.loads(str(getattr(row, "sections_json")))
        out[str(row.novel_id)] = [Section(str(s["kind"]), str(s["text"])) for s in payload]
    return out
