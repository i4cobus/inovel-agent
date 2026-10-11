"""The agent-layer rework of 2026-10-11: card filters, similar books, session state, ask_book over digest chunks,
set_aside, the asks_user finish flag, and concurrent task runs over forked bundles."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.agent.backends import AgentBundle, ParquetProfiles, assemble_tools
from src.agent.context import ContextBudget, build_system_prompt, compact_tool_message
from src.agent.loop import AgentConfig, AgentLoop
from src.agent.memory import UserMemory
from src.agent.passages import DigestPassages, coverage_text
from src.agent.session import SessionState
from src.agent.tools import ToolError, ToolRegistry, build_ask_book, build_search_books, build_set_aside, build_similar_books
from src.agent.trajectory import Trajectory
from src.agent_eval.runner import RunPaths, run_tasks
from src.agent_eval.tasks import Session, Task
from src.chat_transport import ChatResponse, TokenUsage, ToolCall
from src.retrieval.cards import BookCard
from src.retrieval.catalog import CardCatalog, CardFilter, FilterError, resolve_genre, resolve_style
from src.retrieval.hybrid import MultiVectorSearcher, SingleVectorSearcher
from src.retrieval.multivector import MultiVectorIndex, build_section_table, make_book_meta
from src.vector_index import build_faiss_index


class HashEmbedder:
    """Deterministic bag-of-character-bigrams vectors: texts sharing wording land close, prefixes barely matter."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            body = text.split("\n", 1)[-1] if "\n" in text else text
            for a, b in zip(body, body[1:]):
                rng = np.random.default_rng((ord(a) * 1315423911 + ord(b)) % (2**32))
                out[row] += rng.normal(size=self.dim)
            if not np.any(out[row]):
                out[row, hash(text) % self.dim] = 1.0
        return out / np.linalg.norm(out, axis=1, keepdims=True)


CARDS = {
    "a": BookCard("a", genre="仙侠", subgenre="幻想修仙", elements=["凡人流", "宗门"], style={"爽度": "低", "感情线": "无"}),
    "b": BookCard("b", genre="仙侠", subgenre="古典仙侠", elements=["系统"], style={"爽度": "高"}),
    "c": BookCard("c", genre="都市", subgenre="都市异能", elements=["异能", "系统"], style={"爽度": "高"}),
}


# ---- catalog ---------------------------------------------------------------------------------


def test_filters_resolve_through_the_vocabulary_and_its_aliases() -> None:
    assert resolve_genre("仙侠") == ("仙侠", None)
    assert resolve_genre("修仙") == ("仙侠", "幻想修仙")  # alias -> sub-genre -> its genre
    assert resolve_style("爽度=低") == ("爽度", "低") and resolve_style("感情线：无") == ("感情线", "无")
    spec = CardFilter.parse(genre="都市异能", elements=["金手指", "超能力"], style=["爽度=高"])
    assert spec.genre == "都市" and spec.subgenre == "都市异能" and spec.elements == ("系统", "异能") and spec.style == (("爽度", "高"),)
    assert spec.describe() == "题材=都市·都市异能；元素=系统、异能；风格=爽度:高"
    for bad in (dict(genre="武打"), dict(elements=["飞天"]), dict(style=["爽度=极高"]), dict(style=["节奏=快"])):
        with pytest.raises(FilterError):
            CardFilter.parse(**bad)


def test_catalog_selects_by_genre_subgenre_elements_and_style() -> None:
    catalog = CardCatalog(CARDS)
    assert catalog.select(CardFilter.parse(genre="仙侠")) == {"a", "b"}
    assert catalog.select(CardFilter.parse(genre="幻想修仙")) == {"a"}
    assert catalog.select(CardFilter.parse(elements=["系统"])) == {"b", "c"}
    assert catalog.select(CardFilter.parse(genre="仙侠", elements=["系统"], style=["爽度=高"])) == {"b"}
    assert catalog.select(CardFilter.parse(style=["感情线=无"])) == {"a"}
    assert catalog.select(CardFilter()) == {"a", "b", "c"}
    assert catalog.label("a") == "仙侠·幻想修仙" and catalog.label("zzz") == ""


# ---- searchers: filtered search and similar --------------------------------------------------


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class AxisEmbedder:
    """Queries pick an axis by their first character so the test can steer the search."""

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        return np.stack([unit(*[1.0 if i == "abcd".index(t[0]) else 0.0 for i in range(4)]) for t in texts])


def single_searcher() -> SingleVectorSearcher:
    vectors = np.stack([unit(1, 0, 0, 0), unit(0.9, 0.1, 0, 0), unit(0, 0, 1, 0), unit(0, 0, 0, 1)])
    id_map = {i: {"novel_id": n, "title_guess": f"书{n}", "profile_text_preview": "p"} for i, n in enumerate("abcd")}
    return SingleVectorSearcher(AxisEmbedder(), build_faiss_index(vectors), id_map)


def test_single_searcher_restricts_to_allowed_ids_and_finds_similar_books() -> None:
    searcher = single_searcher()
    assert [r["novel_id"] for r in searcher.search("a", 2)] == ["a", "b"]
    assert [r["novel_id"] for r in searcher.search("a", 2, allowed_ids={"c", "d", "b"})] == ["b", "c"] or [r["novel_id"] for r in searcher.search("a", 2, allowed_ids={"c", "d", "b"})][0] == "b"
    assert searcher.search("a", 2, allowed_ids=set()) == []
    similar = searcher.similar("a", 2)
    assert similar[0]["novel_id"] == "b" and "a" not in {r["novel_id"] for r in similar}
    assert [r["novel_id"] for r in searcher.similar("a", 3, allowed_ids={"c", "d"})] and "b" not in {r["novel_id"] for r in searcher.similar("a", 3, allowed_ids={"c", "d"})}
    assert searcher.similar("zzz", 2) == []


def test_multi_searcher_restricts_to_allowed_books() -> None:
    frame = pd.DataFrame(
        [
            {"novel_id": n, "title_guess": n, "profile_text": "x", "sections_json": json.dumps([{"kind": "blurb", "text": f"{n}的简介"}, {"kind": "opening", "text": f"第一章\n{n}的正文"}], ensure_ascii=False)}
            for n in ("a", "b", "c")
        ]
    )
    texts, records = build_section_table(frame, None, chunk_chars=1500)
    embedder = HashEmbedder()
    index = MultiVectorIndex.build(embedder.encode(texts), records, make_book_meta(frame))
    searcher = MultiVectorSearcher(embedder, index)
    top = searcher.search("a的简介", 3)
    assert top[0]["novel_id"] == "a"
    restricted = searcher.search("a的简介", 3, allowed_ids={"b", "c"})
    assert {r["novel_id"] for r in restricted} == {"b", "c"}
    assert searcher.search("a的简介", 3, allowed_ids={"zzz"}) == []


# ---- search_books with filters, session and set_aside ----------------------------------------


class RecordingSearcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, set[str] | None]] = []
        self.rows_by_novel = {n: i for i, n in enumerate("abc")}

    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[dict[str, Any]]:
        self.calls.append((query, k, allowed_ids))
        ids = sorted(allowed_ids) if allowed_ids is not None else ["a", "b", "c"]
        return [{"novel_id": n, "title_guess": f"书{n}", "profile_text_preview": "p", "score": 1.0} for n in ids[:k]]

    def similar(self, novel_id: str, k: int, allowed_ids: set[str] | None = None) -> list[dict[str, Any]]:
        return [{"novel_id": n, "title_guess": f"书{n}", "profile_text_preview": "p", "score": 0.5} for n in "abc" if n != novel_id][:k]


def test_search_books_filters_through_the_catalog_and_labels_rows() -> None:
    searcher, session = RecordingSearcher(), SessionState()
    registry = ToolRegistry()
    registry.register(build_search_books(searcher, ContextBudget(), catalog=CardCatalog(CARDS), session=session))
    out = registry.call("search_books", {"query": "修仙 宗门", "k": 5, "genre": "修仙", "style": ["爽度=低"]})
    assert searcher.calls[-1] == ("修仙 宗门", 5, {"a"})
    assert out["filter"] == "题材=仙侠·幻想修仙；风格=爽度:低" and out["filter_matches"] == 1
    assert out["results"][0]["genre"] == "仙侠·幻想修仙"
    with pytest.raises(ToolError, match="不认识的题材"):
        registry.call("search_books", {"query": "q", "genre": "武打"})
    with pytest.raises(ToolError, match="没有书卡同时满足"):
        registry.call("search_books", {"query": "q", "genre": "都市", "style": ["爽度=低"]})


def test_exclude_shown_skips_recommended_and_set_aside_books() -> None:
    searcher, session = RecordingSearcher(), SessionState()
    registry = ToolRegistry()
    registry.register(build_search_books(searcher, ContextBudget(), catalog=CardCatalog(CARDS), session=session))
    registry.register(build_similar_books(searcher, ContextBudget(), catalog=CardCatalog(CARDS), session=session))
    registry.register(build_set_aside(session))
    session.note_recommendations([{"novel_id": "a", "title": "书a"}])
    assert registry.call("set_aside", {"novel_ids": ["b"], "reason": "看过了"})["excluded"] == ["b"]
    out = registry.call("search_books", {"query": "q", "k": 2, "exclude_shown": True})
    assert searcher.calls[-1] == ("q", 4, None)  # over-fetched by the number avoided, then filtered
    assert [r["novel_id"] for r in out["results"]] == ["c"] and out["excluded"] == 2
    with pytest.raises(ToolError, match="排除已推荐"):  # filter {a, b} minus avoided {a, b}
        registry.call("search_books", {"query": "q", "k": 2, "genre": "仙侠", "exclude_shown": True})
    sim = registry.call("similar_books", {"novel_id": "a", "k": 2, "exclude_shown": True})
    assert [r["novel_id"] for r in sim["results"]] == ["c"]
    prompt = build_system_prompt(UserMemory(), ContextBudget(), session)
    assert "本次对话已推荐" in prompt and "1. 书a（a）" in prompt and "用户已排除" in prompt


def test_exclusion_on_top_of_a_filter_that_leaves_nothing_is_an_error() -> None:
    searcher, session = RecordingSearcher(), SessionState()
    registry = ToolRegistry()
    registry.register(build_search_books(searcher, ContextBudget(), catalog=CardCatalog(CARDS), session=session))
    session.note_recommendations([{"novel_id": "a", "title": "书a"}])
    with pytest.raises(ToolError, match="排除已推荐"):
        registry.call("search_books", {"query": "q", "genre": "幻想修仙", "exclude_shown": True})


# ---- the loop keeps the session and records asks_user ----------------------------------------


def scripted(turns: list[ChatResponse]) -> Any:
    class Model:
        def __init__(self) -> None:
            self.turns = list(turns)

        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int) -> ChatResponse:
            return self.turns.pop(0)

    return Model()


def test_loop_updates_the_session_from_search_results_and_finish() -> None:
    session = SessionState()
    registry = ToolRegistry()
    registry.register(build_search_books(RecordingSearcher(), ContextBudget(), session=session))
    model = scripted(
        [
            ChatResponse("", (ToolCall("s", "search_books", {"query": "q", "k": 2}, "{}"),), TokenUsage()),
            ChatResponse("", (ToolCall("f", "finish", {"answer": "你想看哪类？", "asks_user": True, "recommendations": [{"novel_id": "a", "title": "书a"}]}, "{}"),), TokenUsage()),
        ]
    )
    run = AgentLoop(model, registry, session=session, config=AgentConfig(max_steps=3)).run("推荐点好看的")
    assert run.trajectory.structured["asks_user"] is True
    assert session.shown == {"a": "书a", "b": "书b"} and session.recommended == [{"novel_id": "a", "title": "书a"}]
    assert session.avoid() == {"a"}


# ---- ask_book over digest chunks -------------------------------------------------------------


SECTIONS = {
    "n1": [
        {"kind": "blurb", "text": "标题：甲\n长度：约10万字\n章节数：约30章\n结局：完结"},
        {"kind": "opening", "text": "第一章 少年\n主角叫张三，捡到一枚戒指，里面住着老爷爷。"},
        {"kind": "opening", "text": "第二章 入门\n张三拜入青云宗，成为外门弟子。"},
        {"kind": "titles", "text": "章节目录（抽样）：\n第一章 少年 / 第二章 入门 / 第三十章 飞升"},
        {"kind": "middle", "text": "第十五章 秘境\n张三在秘境里得到了一株九叶灵芝。"},
        {"kind": "ending", "text": "第三十章 飞升\n张三渡过天劫，白日飞升。"},
    ]
}


class Lookup:
    def sections(self, novel_id: str) -> list[dict[str, str]] | None:
        return SECTIONS.get(novel_id)

    def get(self, novel_id: str) -> dict[str, str] | None:
        return {"title": "甲", "profile": "x"} if novel_id in SECTIONS else None


def test_digest_passages_rank_chunks_report_coverage_and_reuse_index_vectors(tmp_path: Path) -> None:
    embedder = HashEmbedder()
    passages = DigestPassages(Lookup(), embedder, multi_dir=None)
    found = passages.ask("n1", "张三渡过天劫，白日飞升。", k=2)  # shares its wording with the ending chunk
    assert found["source"] == "embed" and found["passages"][0].heading == "第三十章 飞升" and found["passages"][0].kind == "ending"
    assert found["coverage"] == "开头章节 2 章（第一章 少年…第二章 入门）、中段章节 1 章（第十五章 秘境）、结尾章节 1 章（第三十章 飞升）、章节目录、简介"
    assert passages.ask("zzz", "q") is None

    # With the multi-vector index on disk the vectors come from it (memory-mapped), not from the embedder.
    frame = pd.DataFrame([{"novel_id": "n1", "title_guess": "甲", "profile_text": "x", "sections_json": json.dumps(SECTIONS["n1"], ensure_ascii=False)}])
    texts, records = build_section_table(frame, {"n1": "题材：仙侠"}, chunk_chars=1500)  # a card section too: must not confuse the mapping
    multi_dir = tmp_path / "multi"
    MultiVectorIndex.build(embedder.encode(texts), records, make_book_meta(frame)).save(multi_dir)

    class Counting(HashEmbedder):
        batches: list[int] = []

        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            Counting.batches.append(len(texts))
            return super().encode(texts, **kwargs)

    mapped = DigestPassages(Lookup(), Counting(), multi_dir=multi_dir)
    found2 = mapped.ask("n1", "张三渡过天劫，白日飞升。", k=2)
    assert found2["source"] == "index:multi" and found2["passages"][0].heading == "第三十章 飞升"
    assert [(p.kind, p.ordinal) for p in found2["passages"]] == [(p.kind, p.ordinal) for p in found["passages"]]
    assert Counting.batches == [1]  # only the question was embedded; the chunk vectors came from the index


def test_ask_book_tool_shapes_truncates_and_redacts() -> None:
    budget = ContextBudget(passage_chars=12)
    registry = ToolRegistry()
    registry.register(build_ask_book(DigestPassages(Lookup(), HashEmbedder()), budget, Lookup()))
    out = registry.call("ask_book", {"novel_id": "n1", "question": "主角的金手指是什么", "k": 2})
    assert out["title"] == "甲" and len(out["passages"]) == 2 and set(out["passages"][0]) == {"chapter", "kind", "text", "score"}
    assert all(len(p["text"]) <= 12 + len("…[截断]") for p in out["passages"]) and "coverage" in out and out["note"]
    redacted = registry.specs["ask_book"].redact(out)
    assert all("sha256" in p["text"] for p in redacted["passages"]) and redacted["passages"][0]["chapter"] == out["passages"][0]["chapter"]
    compact = compact_tool_message(json.dumps(out, ensure_ascii=False), "ask_book")
    assert compact.startswith("[已读段落] n1|甲") and "张三" not in compact.split(":", 1)[1].replace(out["passages"][0]["chapter"], "").replace(out["passages"][1]["chapter"], "")
    with pytest.raises(ToolError, match="没有这本书"):
        registry.call("ask_book", {"novel_id": "zzz", "question": "q"})
    with pytest.raises(ToolError):
        registry.call("ask_book", {"novel_id": "n1", "question": "q", "k": 99})


def test_parquet_profiles_keep_sections_only_on_request(tmp_path: Path) -> None:
    frame = pd.DataFrame([{"novel_id": "n1", "title_guess": "甲", "profile_text": "x", "sections_json": json.dumps(SECTIONS["n1"], ensure_ascii=False), "digest_version": "digest_v2"}])
    frame.to_parquet(tmp_path / "d.parquet")
    lean = ParquetProfiles.load(tmp_path / "d.parquet")
    assert lean.sections("n1") is None and lean.get("n1")["blurb"].startswith("标题：甲")
    full = ParquetProfiles.load(tmp_path / "d.parquet", keep_sections=True)
    assert [s["kind"] for s in full.sections("n1")] == ["blurb", "opening", "opening", "titles", "middle", "ending"]


# ---- forked bundles run tasks concurrently ---------------------------------------------------


def test_forked_bundles_run_tasks_in_parallel_with_separate_memory_and_session(tmp_path: Path) -> None:
    class Transport:
        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int) -> ChatResponse:
            user = messages[-1]["content"]
            if messages[-1]["role"] == "user" and "search" not in str(messages[-2:-1]):
                pass
            if any(m.get("role") == "tool" for m in messages):
                return ChatResponse("", (ToolCall("f", "finish", {"answer": f"推荐给 {user[:3]}", "recommendations": [{"novel_id": "a", "title": "书a"}]}, "{}"),), TokenUsage())
            return ChatResponse("", (ToolCall("m", "memory_write", {"kind": "negative", "value": user[:2], "persistent": True}, "{}"), ToolCall("s", "search_books", {"query": user, "k": 1}, "{}")), TokenUsage())

    backends = {"searcher": RecordingSearcher(), "profiles": None, "catalog": None, "densities": {"a": {"系统": 0.0}}, "judge": None, "passages": None, "transport": Transport()}
    root = AgentBundle(loop=None, tools=ToolRegistry(), memory=UserMemory(), memory_path=tmp_path / "root.json", backends=backends, config=AgentConfig(max_steps=3))  # type: ignore[arg-type]
    root.reset_memory(tmp_path / "root.json")
    assert sorted(root.tools.specs) == ["check_term", "memory_read", "memory_write", "search_books", "set_aside", "similar_books"]

    tasks = [Task(task_id=f"t{i}", kind="memory", split="dev", sessions=[Session(user_message=f"用户{i}的要求"), Session(user_message=f"再来{i}")]) for i in range(4)]
    paths = RunPaths(run_dir=tmp_path / "run", local_dir=tmp_path / "local")
    records = run_tasks(root, tasks, paths, {"model": "stub"}, workers=3)
    assert [r["task_id"] for r in records] == ["t0", "t1", "t2", "t3"]
    for i, record in enumerate(records):
        states = record["memory_states"]
        assert [e["value"] for e in states[-1]["negative"]] == [f"用户{i}"[:2], f"再来{i}"[:2]]  # each task saw only its own writes
        assert record["trajectories"][0]["termination"] == "finish"
    assert root.session.recommended == []  # the root bundle was never used
    assert json.loads((paths.run_dir / "config.json").read_text(encoding="utf-8"))["workers"] == 3
