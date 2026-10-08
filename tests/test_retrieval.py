import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.profile import make_profile_text
from src.retrieval.bench import BenchQuery, anchor_rank, evaluate, format_table, load_benchmark
from src.retrieval.bm25 import BM25Index, tokenize
from src.retrieval.hybrid import BM25Searcher, HybridSearcher, MultiVectorSearcher, reciprocal_rank_fusion
from src.retrieval.multivector import MultiVectorIndex, build_section_table, make_book_meta, section_stats, split_profile_sections


class CharHashModel:
    """Bag-of-characters embedding: overlap in characters is cosine similarity. Deterministic, no download."""

    dim = 64

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for char in text:
                if "一" <= char <= "鿿":
                    out[row, int(hashlib.md5(char.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


def profile(title: str, blurb: str, excerpts: list[str]) -> str:
    return make_profile_text(title_guess=title, author_guess="某人", char_count=1_000_000, chapter_count=500, blurb=blurb, chapter_excerpts=excerpts)


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"novel_id": "a", "title_guess": "《凡人修仙传》", "profile_text": profile("《凡人修仙传》", "普通少年韩立拜入宗门修仙。", ["韩立在七玄门学艺。", "炼气筑基，步步为营。"])},
            {"novel_id": "b", "title_guess": "《诡秘之主》", "profile_text": profile("《诡秘之主》", "克莱恩穿越到蒸汽时代，与邪神周旋。", ["愚者先生的塔罗会。", "序列途径与非凡特性。"])},
            {"novel_id": "c", "title_guess": "《琅琊榜》", "profile_text": profile("《琅琊榜》", "梅长苏回京复仇，朝堂权谋。", [])},
        ]
    )


# ---- BM25 ---------------------------------------------------------------------------------


def test_tokenize_drops_punctuation_and_lowercases() -> None:
    tokens = tokenize("韩立，修仙！Hello World。")
    assert "，" not in tokens and "！" not in tokens
    assert "hello" in tokens and "修仙" in tokens


def test_bm25_ranks_matching_documents_and_round_trips(tmp_path: Path) -> None:
    docs = ["韩立修仙，拜入宗门", "克莱恩与邪神周旋", "梅长苏回京复仇"]
    index = BM25Index.build(docs, ["a", "b", "c"])
    hits = index.search("修仙 宗门", k=3)
    assert hits[0][0] == "a" and len(hits) == 1  # only the document sharing a token appears
    assert index.search("不存在的词", k=3) == []
    assert index.search("修仙", k=0) == []

    index.save(tmp_path / "bm25.json")
    reloaded = BM25Index.load(tmp_path / "bm25.json")
    assert reloaded.search("修仙 宗门", k=3) == hits
    assert reloaded.avg_length == index.avg_length


def test_bm25_rejects_mismatched_ids() -> None:
    with pytest.raises(ValueError):
        BM25Index.build(["x"], ["a", "b"])


# ---- multi-vector -------------------------------------------------------------------------


def test_profile_splits_into_blurb_and_excerpts_with_title_carried() -> None:
    sections = split_profile_sections(frame().iloc[0]["profile_text"])
    assert [s.kind for s in sections] == ["blurb", "excerpt", "excerpt"]
    assert sections[0].text.startswith("标题：《凡人修仙传》") and "内容简介" in sections[0].text
    assert sections[1].text.startswith("标题：《凡人修仙传》\n") and "七玄门" in sections[1].text
    assert "节选" not in sections[1].text

    only_blurb = split_profile_sections(frame().iloc[2]["profile_text"])
    assert [s.kind for s in only_blurb] == ["blurb"]
    assert split_profile_sections("") == []
    assert split_profile_sections("无标记的文本")[0].kind == "blurb"


def test_multivector_index_max_pools_sections_per_book(tmp_path: Path) -> None:
    model = CharHashModel()
    texts, records = build_section_table(frame())
    assert len(records) == 7  # a: blurb + 2 excerpts, b: same, c: blurb only
    index = MultiVectorIndex.build(model.encode(texts), records, make_book_meta(frame()))
    assert index.book_count == 3

    searcher = MultiVectorSearcher(model, index)
    rows = searcher.search("七玄门学艺", k=3)
    assert rows[0]["novel_id"] == "a" and rows[0]["section_kind"] == "excerpt"
    assert [row["rank"] for row in rows] == [1, 2, 3]
    assert len({row["novel_id"] for row in rows}) == 3  # one row per book, never a duplicate

    index.save(tmp_path, metadata={"model_name": "fake"})
    reloaded = MultiVectorSearcher(model, MultiVectorIndex.load(tmp_path))
    assert [r["novel_id"] for r in reloaded.search("七玄门学艺", k=3)] == [r["novel_id"] for r in rows]
    assert searcher.search("   ", k=3) == []


def test_multivector_build_rejects_record_mismatch() -> None:
    with pytest.raises(ValueError):
        MultiVectorIndex.build(np.eye(2, dtype=np.float32), [], {})


# ---- fusion -------------------------------------------------------------------------------


def test_rrf_rewards_agreement_and_respects_weights() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"]])
    assert [item for item, _ in fused][:2] == ["a", "b"] or [item for item, _ in fused][:2] == ["b", "a"]
    weighted = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], weights=[1.0, 3.0])
    assert weighted[0][0] == "b"
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])


def test_hybrid_combines_bm25_and_dense_rows() -> None:
    model = CharHashModel()
    texts, records = build_section_table(frame())
    dense = MultiVectorSearcher(model, MultiVectorIndex.build(model.encode(texts), records, make_book_meta(frame())))
    bm25 = BM25Searcher(BM25Index.build(frame()["profile_text"].tolist(), frame()["novel_id"].tolist()), make_book_meta(frame()))
    hybrid = HybridSearcher([dense, bm25], depth=10)
    rows = hybrid.search("梅长苏 权谋", k=2)
    assert rows[0]["novel_id"] == "c"
    assert rows[0]["fused_from"] == ["dense_multi", "bm25"]
    assert rows[0]["title_guess"] == "《琅琊榜》" and [r["rank"] for r in rows] == [1, 2]
    assert hybrid.name == "hybrid(dense_multi+bm25)"
    assert hybrid.search("x", k=0) == []


# ---- benchmark ----------------------------------------------------------------------------


def test_benchmark_loads_the_committed_evaluation_artifacts() -> None:
    queries = load_benchmark()
    assert len(queries) == 59
    assert sum(len(q.anchors) for q in queries) == 55
    assert sum(1 for q in queries if q.anchors) == 31
    assert sum(1 for q in queries if q.strong) == 56
    assert sum(len(q.strong) for q in queries) == 627


class FixedSearcher:
    name = "fixed"

    def __init__(self, rows_by_query: dict[str, list[dict[str, Any]]]) -> None:
        self.rows_by_query = rows_by_query

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        return self.rows_by_query.get(query, [])[:k]


def test_evaluate_reads_anchor_ranks_and_recall_from_one_ranking() -> None:
    queries = [
        BenchQuery("q1", "仙侠", anchors=["凡人修仙传"], strong={"a", "z"}),
        BenchQuery("q2", "权谋", anchors=["琅琊榜", "庆余年"], strong=set()),
        BenchQuery("q3", "空", anchors=[], strong=set()),
    ]
    rows = {
        "仙侠": [{"rank": 1, "novel_id": "x", "title_guess": "《小小凡人修仙传》"}, {"rank": 2, "novel_id": "a", "title_guess": "《凡人修仙传》（校对版）"}],
        "权谋": [{"rank": 1, "novel_id": "c", "title_guess": "《琅琊榜》"}],
    }
    result = evaluate(FixedSearcher(rows), queries, depth=50, recall_k=20)
    assert result.anchor_ranks == {"q1|凡人修仙传": 2, "q2|琅琊榜": 1, "q2|庆余年": None}
    assert result.recall_at_k == {"q1": 0.5}
    metrics = result.metrics()
    assert metrics["anchor_hit@10"] == round(2 / 3, 4) and metrics["anchor_unfound"] == 1
    assert metrics["anchor_median_rank"] == 2  # unfound counts as depth + 1 = 51
    assert metrics["recall@20_macro"] == 0.5 and metrics["strong_queries"] == 1
    table = format_table([result])
    assert table.startswith("| config | anchors |") and "| fixed |" in table
    assert anchor_rank([], "x") is None


def test_section_stats_flag_a_single_section_table() -> None:
    _, records = build_section_table(frame())
    stats = section_stats(records)
    assert stats == {"books": 3, "sections": 7, "mean_per_book": 2.33, "share_single": round(1 / 3, 4)}

    old_format = pd.DataFrame([{"novel_id": "x", "title_guess": "t", "profile_text": "标题：t\n\n开篇样本：\n正文" }])
    _, records = build_section_table(old_format)
    assert section_stats(records)["share_single"] == 1.0
    assert section_stats([]) == {"books": 0, "sections": 0, "mean_per_book": 0.0, "share_single": 0.0}

