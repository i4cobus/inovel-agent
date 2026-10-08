"""Derive a one-vector-per-book index from a multi-vector index by pooling its section vectors.

Encoding a whole digest (8.7k tokens on average, 12k at most) in one pass filled the
4080 and spilled into system memory on 2026-10-08, so the single-vector index is no
longer encoded on the GPU. The section vectors already in ``multi_0p6b`` are read
back out of the flat FAISS index, averaged per book (optionally weighted by section
kind), re-normalised and written in the layout ``SingleVectorSearcher`` expects.
Different weightings cost seconds on the Mac, so they can be benchmarked side by side.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval.multivector import INDEX_FILE, META_FILE, SECTIONS_FILE, MultiVectorIndex, SectionRecord
from src.vector_index import build_faiss_index, ensure_can_write, save_faiss_index, save_id_map, save_index_metadata

# Chosen on 2026-10-09 over six weightings benchmarked on the Mac (docs/retrieval-bench.md, round 2):
# the blurb and the chapter-title list are the densest summaries of a book, the 600-char middle
# windows the noisiest. Differences between weightings are one or two anchors; this one had the
# fewest anchors missing from the top 1000 (16 of 55) and the best Recall@20 among those.
DEFAULT_SECTION_WEIGHTS: dict[str, float] = {"blurb": 3.0, "titles": 3.0, "middle": 0.5}

BM25_FILE = "bm25.json"
BOOK_META_FILE = "book_meta.json"
ID_MAP_FILE = "novel_id_map.json"


def reconstruct_vectors(index: Any) -> np.ndarray:
    """All vectors stored in a flat FAISS index, in row order."""

    if index.ntotal == 0:
        raise ValueError("index is empty")
    return np.ascontiguousarray(index.reconstruct_n(0, index.ntotal), dtype=np.float32)


def pool_book_vectors(
    vectors: np.ndarray, records: list[SectionRecord], weights: dict[str, float] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Weighted mean of each book's section vectors, L2-normalised; books keep first-seen order.

    ``weights`` maps a section kind to its weight (default 1.0). A kind weighted 0 is
    dropped; a book left with no weighted section falls back to the plain mean so it is
    never lost from the index.
    """

    if vectors.shape[0] != len(records):
        raise ValueError(f"{vectors.shape[0]} vectors but {len(records)} records")
    weights = weights or {}
    order: list[str] = []
    rows: dict[str, list[int]] = {}
    for row, record in enumerate(records):
        if record.novel_id not in rows:
            rows[record.novel_id] = []
            order.append(record.novel_id)
        rows[record.novel_id].append(row)
    pooled = np.zeros((len(order), vectors.shape[1]), dtype=np.float32)
    for book_row, novel_id in enumerate(order):
        members = rows[novel_id]
        w = np.array([float(weights.get(records[r].kind, 1.0)) for r in members], dtype=np.float32)
        if not (w > 0).any():
            w = np.ones(len(members), dtype=np.float32)
        pooled[book_row] = (vectors[members] * w[:, None]).sum(axis=0) / w.sum()
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    pooled /= np.where(norms == 0, 1.0, norms)
    return pooled, order


def parse_weights(items: list[str]) -> dict[str, float]:
    """``["blurb=2", "titles=1.5"]`` -> ``{"blurb": 2.0, "titles": 1.5}``."""

    weights: dict[str, float] = {}
    for item in items:
        kind, sep, value = item.partition("=")
        if not sep or not kind.strip():
            raise ValueError(f"weight must look like kind=number, got {item!r}")
        weights[kind.strip()] = float(value)
    return weights


def derive_single_index(
    multi_dir: Path, out_dir: Path, weights: dict[str, float] | None = None, copy_bm25: bool = True, overwrite: bool = False
) -> dict[str, Any]:
    """Write a single-vector index directory pooled from ``multi_dir``; returns the build summary."""

    multi = MultiVectorIndex.load(multi_dir)
    targets = [out_dir / INDEX_FILE, out_dir / ID_MAP_FILE, out_dir / META_FILE]
    if copy_bm25:
        targets += [out_dir / BM25_FILE, out_dir / BOOK_META_FILE]
    ensure_can_write(targets, overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled, novel_ids = pool_book_vectors(reconstruct_vectors(multi.index), multi.records, weights)
    save_faiss_index(build_faiss_index(pooled), out_dir / INDEX_FILE)
    id_map = {
        str(row): {"novel_id": novel_id, **{k: str(v) for k, v in multi.meta.get(novel_id, {}).items()}}
        for row, novel_id in enumerate(novel_ids)
    }
    save_id_map(id_map, out_dir / ID_MAP_FILE)

    metadata_path = multi_dir / META_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata.update(
        {
            "dense": "single",
            "num_vectors": len(novel_ids),
            "index_type": "IndexFlatIP",
            "pooled_from": multi_dir.as_posix(),
            "pooling": "weighted_mean",
            "section_weights": weights or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    save_index_metadata(metadata, out_dir / META_FILE)

    if copy_bm25:
        for name in (BM25_FILE, BOOK_META_FILE):
            source = multi_dir / name
            if source.exists():
                shutil.copy2(source, out_dir / name)

    summary = {
        "pooled_from": multi_dir.as_posix(),
        "books": len(novel_ids),
        "sections": len(multi.records),
        "dim": int(pooled.shape[1]),
        "section_weights": weights or {},
        "created_at": metadata["created_at"],
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
