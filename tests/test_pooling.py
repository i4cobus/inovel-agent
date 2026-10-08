import json
from pathlib import Path

import numpy as np
import pytest

from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import SingleVectorSearcher, load_searchers
from src.retrieval.multivector import MultiVectorIndex, SectionRecord
from src.retrieval.pooling import DEFAULT_SECTION_WEIGHTS, derive_single_index, parse_weights, pool_book_vectors, reconstruct_vectors


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


RECORDS = [
    SectionRecord("a", "blurb", 0),
    SectionRecord("a", "opening", 1),
    SectionRecord("b", "blurb", 0),
    SectionRecord("b", "titles", 1),
    SectionRecord("b", "ending", 2),
]
VECTORS = np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1), unit(0, 1, 0), unit(0, 1, 0)])


def test_pool_book_vectors_means_per_book_in_first_seen_order() -> None:
    pooled, ids = pool_book_vectors(VECTORS, RECORDS)
    assert ids == ["a", "b"]
    assert np.allclose(pooled[0], unit(1, 1, 0))
    assert np.allclose(pooled[1], unit(0, 2, 1))
    assert np.allclose(np.linalg.norm(pooled, axis=1), 1.0)


def test_pool_book_vectors_applies_kind_weights_and_never_drops_a_book() -> None:
    pooled, _ = pool_book_vectors(VECTORS, RECORDS, weights={"blurb": 3.0})
    assert np.allclose(pooled[0], unit(3, 1, 0))
    # Every section of book "a" weighted 0 -> plain mean rather than a zero vector.
    pooled, _ = pool_book_vectors(VECTORS, RECORDS, weights={"blurb": 0.0, "opening": 0.0})
    assert np.allclose(pooled[0], unit(1, 1, 0))
    assert np.allclose(pooled[1], unit(0, 2, 0))


def test_pool_book_vectors_rejects_mismatch() -> None:
    with pytest.raises(ValueError):
        pool_book_vectors(VECTORS[:3], RECORDS)


def test_default_weights_name_real_digest_kinds() -> None:
    assert set(DEFAULT_SECTION_WEIGHTS) <= {"blurb", "opening", "titles", "middle", "ending", "card"}
    assert all(w > 0 for w in DEFAULT_SECTION_WEIGHTS.values())


def test_parse_weights() -> None:
    assert parse_weights(["blurb=2", " titles = 1.5"]) == {"blurb": 2.0, "titles": 1.5}
    with pytest.raises(ValueError):
        parse_weights(["blurb"])


class QueryModel:
    def encode(self, texts, **kwargs):
        return np.stack([unit(0, 0, 1) for _ in texts])


def test_derive_single_index_writes_a_loadable_single_vector_directory(tmp_path: Path) -> None:
    multi_dir, out_dir = tmp_path / "multi", tmp_path / "single"
    meta = {"a": {"title_guess": "甲", "profile_text_preview": "甲书"}, "b": {"title_guess": "乙", "profile_text_preview": "乙书"}}
    MultiVectorIndex.build(VECTORS, RECORDS, meta).save(multi_dir, metadata={"model_name": "m", "dtype": "bf16", "dense": "multi"})
    BM25Index.build(["甲书 修仙", "乙书 都市"], ["a", "b"]).save(multi_dir / "bm25.json")
    (multi_dir / "book_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    assert np.allclose(reconstruct_vectors(MultiVectorIndex.load(multi_dir).index), VECTORS)
    summary = derive_single_index(multi_dir, out_dir, weights={"blurb": 2.0})
    assert summary["books"] == 2 and summary["sections"] == 5 and summary["section_weights"] == {"blurb": 2.0}
    metadata = json.loads((out_dir / "index_metadata.json").read_text(encoding="utf-8"))
    assert metadata["dense"] == "single" and metadata["model_name"] == "m" and metadata["pooled_from"] == multi_dir.as_posix()
    assert (out_dir / "bm25.json").exists() and not (out_dir / "sections.json").exists()

    searcher = SingleVectorSearcher.load(QueryModel(), out_dir)
    hits = searcher.search("无所谓", k=2)
    assert [h["novel_id"] for h in hits] == ["b", "a"]
    assert hits[0]["title_guess"] == "乙"
    assert [s.name for s in load_searchers(out_dir, QueryModel(), label="x")] == ["x/dense_single", "x/bm25", "x/hybrid_rrf"]

    with pytest.raises(FileExistsError):
        derive_single_index(multi_dir, out_dir)
    derive_single_index(multi_dir, out_dir, overwrite=True)
