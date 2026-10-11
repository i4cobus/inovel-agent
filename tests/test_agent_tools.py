import json
from pathlib import Path
from typing import Any

import pytest

from src.agent.context import ContextBudget
from src.agent.memory import UserMemory
from src.agent.tools import (
    ToolError,
    ToolRegistry,
    build_check_term,
    build_check_trope,
    build_get_profile,
    build_memory_tools,
    build_search_books,
)
from src.agent.trajectory import Observation, Step, Trajectory, text_fingerprint


class FakeSearcher:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        self.queries.append((query, k))
        return [{"novel_id": f"n{i}", "title_guess": f"书{i}", "profile_text_preview": "简介" * 300, "score": 0.9 - i / 100} for i in range(k)]


class FakeProfiles:
    def get(self, novel_id: str) -> dict[str, str] | None:
        if novel_id == "n1":
            return {"title": "书", "profile": "正文" * 2000}
        if novel_id == "n2":
            return {"title": "书二", "profile": "第一章\n" + "开头" * 2000, "blurb": "标题：书二\n\n内容简介：\n" + "简介" * 400, "card": "题材：玄幻·东方玄幻\n元素：系统"}
        return None


class FakeTropeJudge:
    def judge(self, novel_id: str, trope: str) -> dict[str, Any]:
        return {"novel_id": novel_id, "trope": trope, "verdict": "yes", "quotes": ["原文引文"], "confidence": "high"}


def full_registry(memory: UserMemory | None = None) -> ToolRegistry:
    budget = ContextBudget()
    registry = ToolRegistry()
    registry.register(build_search_books(FakeSearcher(), budget))
    registry.register(build_get_profile(FakeProfiles(), budget))
    registry.register(build_check_term({"n1": {"系统": 12.0, "异能": 0.0}, "n2": {"系统": 0.0}}))
    registry.register(build_check_trope(FakeTropeJudge()))
    for spec in build_memory_tools(memory or UserMemory()):
        registry.register(spec)
    return registry


def test_schemas_are_openai_function_tools() -> None:
    schemas = full_registry().schemas()
    names = [schema["function"]["name"] for schema in schemas]
    assert names == ["search_books", "get_profile", "check_term", "check_trope", "memory_read", "memory_write"]
    assert all(schema["type"] == "function" for schema in schemas)
    json.dumps(schemas)  # must be serialisable as-is


def test_search_books_applies_preview_budget_and_k_bounds() -> None:
    registry = full_registry()
    searcher = FakeSearcher()
    registry = ToolRegistry()
    registry.register(build_search_books(searcher, ContextBudget()))
    rows = registry.call("search_books", {"query": "主角理性的慢热仙侠，不要系统", "k": 3})["results"]
    assert searcher.queries == [("主角理性的慢热仙侠，不要系统", 3)]  # verbatim: the agent owns the query wording (2026-10-11)
    assert len(rows) == 3
    assert len(rows[0]["preview"]) == ContextBudget().preview_chars
    assert set(rows[0]) == {"novel_id", "title", "preview", "score"}
    with pytest.raises(ToolError, match="没有书卡"):
        registry.call("search_books", {"query": "仙侠", "genre": "仙侠"})
    with pytest.raises(ToolError):
        registry.call("search_books", {"query": "仙侠", "k": 999})
    with pytest.raises(ToolError):
        registry.call("search_books", {"query": "   "})


def test_get_profile_truncates_and_rejects_unknown_books() -> None:
    registry = full_registry()
    profile = registry.call("get_profile", {"novel_id": "n1"})
    assert len(profile["opening"]) == ContextBudget().profile_chars
    assert "card" not in profile and "blurb" not in profile
    with pytest.raises(ToolError, match="没有这本书"):
        registry.call("get_profile", {"novel_id": "zzz"})


def test_get_profile_returns_card_and_synopsis_when_the_book_has_them() -> None:
    registry = full_registry()
    profile = registry.call("get_profile", {"novel_id": "n2"})
    assert profile["card"].startswith("题材：玄幻")
    assert profile["blurb"].startswith("简介") and len(profile["blurb"]) == ContextBudget().blurb_chars  # header stripped
    assert profile["opening"].startswith("第一章") and len(profile["opening"]) == ContextBudget().profile_chars
    redacted = registry.redactors()["get_profile"](profile)
    assert redacted["card"] == profile["card"] and "简介简介" not in json.dumps(redacted, ensure_ascii=False) and "开头开头" not in json.dumps(redacted, ensure_ascii=False)


def test_check_term_rule_verdicts_and_null_for_meta_labels() -> None:
    registry = full_registry()
    hit = registry.call("check_term", {"novel_id": "n1", "terms": ["系统"]})
    assert hit["violates"] is True and hit["evidence"] == {"系统": 12.0}

    clean = registry.call("check_term", {"novel_id": "n2", "terms": ["系统"]})
    assert clean["violates"] is False

    meta = registry.call("check_term", {"novel_id": "n1", "terms": ["后宫"]})
    assert meta["violates"] is None and meta["not_checkable"] == ["后宫"] and "check_trope" in meta["note"]

    with pytest.raises(ToolError):
        registry.call("check_term", {"novel_id": "n1", "terms": []})
    with pytest.raises(ToolError, match="词频表"):
        registry.call("check_term", {"novel_id": "unknown", "terms": ["系统"]})


def test_registry_validates_required_and_unknown_arguments() -> None:
    registry = full_registry()
    with pytest.raises(ToolError, match="缺少参数"):
        registry.call("get_profile", {})
    with pytest.raises(ToolError, match="不认识的参数"):
        registry.call("get_profile", {"novel_id": "n1", "extra": 1})
    with pytest.raises(ToolError, match="未知工具"):
        registry.call("nope", {})


def test_memory_tools_write_read_and_session_entries_are_not_saved(tmp_path: Path) -> None:
    memory = UserMemory()
    registry = full_registry(memory)
    registry.call("memory_write", {"kind": "negative", "value": "系统", "persistent": True})
    registry.call("memory_write", {"kind": "negative", "value": "后宫", "persistent": False})
    assert [e["value"] for e in registry.call("memory_read", {})["negative"]] == ["系统", "后宫"]

    path = tmp_path / "user.json"
    memory.save(path)
    assert UserMemory.load(path).values("negative") == ["系统"]

    with pytest.raises(ToolError):
        registry.call("memory_write", {"kind": "wrong", "value": "x"})
    with pytest.raises(ToolError):
        registry.call("memory_write", {"kind": "note", "value": "  "})


def test_memory_rewrite_moves_the_value_to_the_end_so_latest_wins() -> None:
    memory = UserMemory()
    memory.write("negative", "系统", True)
    memory.write("negative", "后宫", True)
    memory.write("negative", "系统", True)
    assert memory.values("negative") == ["后宫", "系统"]
    assert memory.forget("negative", "后宫") == 1
    assert "不要：系统" in memory.summary()


def test_redaction_strips_corpus_text_but_keeps_ids() -> None:
    registry = full_registry()
    profile = registry.call("get_profile", {"novel_id": "n1"})
    trope = registry.call("check_trope", {"novel_id": "n1", "trope": "后宫"})
    search = registry.call("search_books", {"query": "q", "k": 2})
    traj = Trajectory(task_id="t", model="m", user_message="u", final_answer="引用了原文")
    traj.steps.append(
        Step(
            index=0,
            assistant_content="",
            tool_calls=[],
            observations=[
                Observation("get_profile", "a", {"novel_id": "n1"}, result=profile),
                Observation("check_trope", "b", {}, result=trope),
                Observation("search_books", "c", {}, result=search),
                Observation("check_term", "d", {}, result={"violates": True}),
            ],
            prompt_tokens=0,
            completion_tokens=0,
            latency_s=0.0,
        )
    )
    redacted = traj.redacted(registry.redactors())
    observations = redacted["steps"][0]["observations"]
    assert observations[0]["result"]["opening"] == text_fingerprint(profile["opening"])
    assert observations[0]["result"]["novel_id"] == "n1"
    assert observations[1]["result"]["quotes"] == [text_fingerprint("原文引文")]
    assert observations[1]["result"]["verdict"] == "yes"
    assert "sha256" in observations[2]["result"]["results"][0]["preview"]
    assert observations[3]["result"] == {"violates": True}
    assert redacted["final_answer"] == text_fingerprint("引用了原文")
    dumped = json.dumps(redacted, ensure_ascii=False)
    assert "正文正文" not in dumped and "原文引文" not in dumped and "简介简介" not in dumped


def test_search_books_prefers_the_synopsis_preview_from_profiles() -> None:
    from src.profile import make_profile_text

    class Profiles:
        def get(self, novel_id: str) -> dict[str, str] | None:
            text = make_profile_text(title_guess="书", author_guess="某人", char_count=1, chapter_count=1, blurb="凡人修仙的故事。", chapter_excerpts=["节选正文"])
            return {"title": "书", "profile": text} if novel_id == "n0" else None

    registry = ToolRegistry()
    registry.register(build_search_books(FakeSearcher(), ContextBudget(), Profiles()))
    rows = registry.call("search_books", {"query": "仙侠", "k": 2})["results"]
    assert rows[0]["preview"] == "作者：某人\n凡人修仙的故事。"
    assert rows[1]["preview"].startswith("简介")  # unknown to the profile table: falls back to the stored preview


def test_compact_tool_message_keeps_ids_and_titles_only() -> None:
    from src.agent.context import compact_tool_message

    rows = json.dumps([{"novel_id": "n1", "title": "书一", "preview": "很长" * 200, "score": 0.5}], ensure_ascii=False)
    compact = compact_tool_message(rows, "search_books")
    assert compact.startswith("[已压缩的检索结果") and "n1|书一" in compact and "很长很长很长" not in compact
    profile = json.dumps({"novel_id": "n1", "title": "书一", "opening": "正文" * 500}, ensure_ascii=False)
    assert len(compact_tool_message(profile, "get_profile")) < 220
    carded = json.dumps({"novel_id": "n1", "title": "书一", "card": "题材：玄幻", "opening": "正文" * 500}, ensure_ascii=False)
    assert "题材：玄幻" in compact_tool_message(carded, "get_profile") and "正文正文" not in compact_tool_message(carded, "get_profile")
    trope = json.dumps({"verdict": "yes", "quotes": ["引文"], "confidence": "high"}, ensure_ascii=False)
    assert "引文" not in compact_tool_message(trope, "check_trope") and "yes" in compact_tool_message(trope, "check_trope")
    assert compact_tool_message("not json " * 50, "x").endswith("…[截断]")

