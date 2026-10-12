"""Sessions and the web server over a scripted model: isolation, persistence, streaming."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient

from src.agent.backends import AgentBundle
from src.agent.loop import AgentConfig, AgentLoop
from src.agent.memory import UserMemory
from src.agent.sessions import SessionBusy, SessionStore, describe_result, summarize_step
from src.agent.tools import ToolRegistry
from src.agent.trajectory import Observation, Step
from src.agent.web import create_app
from src.chat_transport import ChatResponse, TokenUsage, ToolCall


class Searcher:
    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[dict[str, Any]]:
        return [{"novel_id": n, "title_guess": f"书{n}", "profile_text_preview": "p", "score": 1.0} for n in "abc"][:k]

    def similar(self, novel_id: str, k: int, allowed_ids: set[str] | None = None) -> list[dict[str, Any]]:
        return []


class Transport:
    """Turn one: remember the first two characters as a long-term negative, search, then finish with 书a.

    Records how many user/assistant messages each call saw, so a test can check the history a session passes in."""

    def __init__(self) -> None:
        self.history_lengths: list[int] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int) -> ChatResponse:
        user = messages[-1]["content"]
        if any(m.get("role") == "tool" for m in messages):
            return ChatResponse("", (ToolCall("f", "finish", {"answer": f"推荐《书a》给「{user[:4]}」", "recommendations": [{"novel_id": "a", "title": "书a", "reason": "合适"}]}, "{}"),), TokenUsage(30, 10))
        self.history_lengths.append(sum(1 for m in messages if m["role"] in ("user", "assistant")))
        return ChatResponse("先查一下", (ToolCall("m", "memory_write", {"kind": "negative", "value": user[:2], "persistent": True}, "{}"), ToolCall("s", "search_books", {"query": user, "k": 2}, "{}")), TokenUsage(20, 5))


def make_bundle(tmp_path: Path) -> tuple[AgentBundle, Transport]:
    transport = Transport()
    backends = {"searcher": Searcher(), "profiles": None, "catalog": None, "densities": {"a": {"系统": 0.0}}, "judge": None, "passages": None, "transport": transport}
    root = AgentBundle(loop=None, tools=ToolRegistry(), memory=UserMemory(), memory_path=tmp_path / "root.json", backends=backends, config=AgentConfig(max_steps=3))  # type: ignore[arg-type]
    root.reset_memory(tmp_path / "root.json")
    root.info = {"model": "stub", "searcher": "stub", "profiles": 3, "cards": 0}
    return root, transport


# ---- sessions ------------------------------------------------------------------------------


def test_sessions_have_separate_memory_history_and_recommendations(tmp_path: Path) -> None:
    bundle, transport = make_bundle(tmp_path)
    store = SessionStore(bundle, root=tmp_path / "sessions")
    first, second = store.create(), store.create(title="第二个")
    seen: list[dict[str, Any]] = []
    turn = store.send(first.id, "仙侠，凡人流", on_step=seen.append)
    assert turn.termination == "finish" and turn.structured["recommendations"][0]["novel_id"] == "a"
    assert [s["index"] for s in seen] == [0, 1] and seen[0]["calls"][1]["name"] == "search_books" and "2 本" in seen[0]["calls"][1]["summary"]
    assert first.agent.memory.values("negative") == ["仙侠"]
    assert first.title == "仙侠，凡人流"  # titled from the first message
    assert first.state.recommended == [{"novel_id": "a", "title": "书a"}]

    store.send(second.id, "都市异能")
    assert second.agent.memory.values("negative") == ["都市"]  # the first session's entry never showed up here
    assert first.agent.memory.values("negative") == ["仙侠"]
    assert second.title == "第二个"

    store.send(first.id, "再来几本")
    assert transport.history_lengths == [1, 1, 3]  # the second turn of the first session saw its own two earlier messages, not the other session's
    assert bundle.memory.entries == [] and bundle.session.recommended == []  # the root bundle is untouched
    assert {p.name for p in first.directory.iterdir()} == {"session.json", "memory.json", "trajectories.jsonl"}


def test_sessions_are_restored_from_disk_with_state_and_memory(tmp_path: Path) -> None:
    bundle, _ = make_bundle(tmp_path)
    store = SessionStore(bundle, root=tmp_path / "sessions")
    session = store.create()
    store.send(session.id, "玄幻，不要后宫")
    session.agent.session.exclude(["b"])
    session.save()

    reopened = SessionStore(bundle, root=tmp_path / "sessions")
    restored = reopened.get(session.id)
    assert restored.title == "玄幻，不要后宫" and len(restored.turns) == 1 and restored.turns[0].answer.startswith("推荐《书a》")
    assert restored.agent.memory.values("negative") == ["玄幻"]
    assert restored.state.recommended == [{"novel_id": "a", "title": "书a"}] and restored.state.excluded == {"b"}
    assert restored.history() == [{"role": "user", "content": "玄幻，不要后宫"}, {"role": "assistant", "content": restored.turns[0].answer}]
    assert restored.agent.turn == 1
    assert reopened.list()[0]["id"] == session.id

    reopened.delete(session.id)
    assert not session.directory.exists() and reopened.list() == []


def test_history_keeps_only_the_last_turns_and_says_so(tmp_path: Path) -> None:
    bundle, transport = make_bundle(tmp_path)
    store = SessionStore(bundle, root=tmp_path / "sessions", history_turns=2)
    session = store.create()
    for i in range(4):
        store.send(session.id, f"第{i}问")
    # Messages the first model call of each turn saw: 1; 3; 5 (two kept turns); 7 (two kept turns plus the note pair), then it stays at 7
    assert transport.history_lengths == [1, 3, 5, 7]
    history = session.history(2)
    assert history[0]["content"].startswith("（更早的 2 轮对话已省略") and history[1] == {"role": "assistant", "content": "好的。"}
    assert [m["content"] for m in history if m["role"] == "user"][1:] == ["第2问", "第3问"]
    assert session.history(0) == [{"role": "user", "content": "（更早的 4 轮对话已省略；用户的长期偏好见记忆，已推荐过的书见会话状态。）"}, {"role": "assistant", "content": "好的。"}]


def test_a_running_session_rejects_a_second_message(tmp_path: Path) -> None:
    bundle, _ = make_bundle(tmp_path)
    store = SessionStore(bundle, root=tmp_path / "sessions")
    session = store.create()
    assert session.lock.acquire()
    try:
        try:
            store.send(session.id, "x")
            raise AssertionError("expected SessionBusy")
        except SessionBusy:
            pass
    finally:
        session.lock.release()


def test_step_summaries_describe_tool_results_in_one_line() -> None:
    assert describe_result("search_books", {"results": [{"title": "甲"}, {"title": "乙"}], "filter": "题材=仙侠", "filter_matches": 40, "excluded": 2}) == "2 本：《甲》、《乙》（筛选 题材=仙侠；命中 40 本；跳过已推荐 2）"
    assert describe_result("check_term", {"violates": True, "evidence": {"系统": 12.5}}) == "违反（系统 12.5）"
    assert describe_result("check_term", {"violates": None, "evidence": {}}) == "规则判不了"
    assert describe_result("check_trope", {"trope": "后宫", "verdict": "no", "confidence": "high", "reason": "单女主"}) == "后宫：no；置信 high；单女主"
    assert describe_result("memory_write", {"written": {"kind": "negative", "value": "后宫", "persistent": False}}) == "已记住（仅本次）"
    assert describe_result("ask_book", {"coverage": "开头 1-3 章", "passages": [{}, {}]}) == "开头 1-3 章；读了 2 段"
    step = Step(index=0, assistant_content="想想", tool_calls=[{"id": "c1", "name": "get_profile", "arguments": {"novel_id": "x" * 200}}], observations=[Observation(tool="get_profile", call_id="c1", arguments={}, error="没有这本书", latency_s=0.01)], prompt_tokens=1, completion_tokens=1, latency_s=0.5)
    summary = summarize_step(step)
    assert summary["content"] == "想想" and summary["calls"][0]["error"] == "没有这本书" and len(summary["calls"][0]["arguments"]["novel_id"]) == 120


# ---- web -----------------------------------------------------------------------------------


def read_events(response: Any) -> list[tuple[str, dict[str, Any]]]:
    events, event, data = [], "", ""
    for line in response.iter_lines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data += line[5:].strip()
        elif not line and data:
            events.append((event, json.loads(data)))
            event, data = "", ""
    return events


def test_web_api_creates_sessions_and_streams_steps_then_the_turn(tmp_path: Path) -> None:
    bundle, _ = make_bundle(tmp_path)
    store = SessionStore(bundle, root=tmp_path / "sessions")
    client = TestClient(create_app(store))

    assert client.get("/").text.startswith("<!doctype html>")
    assert client.get("/api/info").json()["model"] == "stub"
    created = client.post("/api/sessions", json={}).json()
    assert created["turns"] == [] and created["memory"]["negative"] == []

    with client.stream("POST", f"/api/sessions/{created['id']}/messages", json={"text": "仙侠，凡人流"}) as response:
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
        events = read_events(response)
    kinds = [kind for kind, _ in events]
    assert kinds == ["accepted", "step", "step", "turn"]
    turn = events[-1][1]
    assert turn["turn"]["answer"].startswith("推荐《书a》") and turn["session"]["memory"]["negative"][0]["value"] == "仙侠"
    assert turn["session"]["state"]["recommended"] == [{"novel_id": "a", "title": "书a"}]

    listed = client.get("/api/sessions").json()["sessions"]
    assert listed[0]["id"] == created["id"] and listed[0]["turns"] == 1 and listed[0]["title"] == "仙侠，凡人流"
    assert client.patch(f"/api/sessions/{created['id']}", json={"title": "仙侠"}).json()["title"] == "仙侠"
    assert client.post(f"/api/sessions/{created['id']}/messages", json={"text": ""}).status_code == 400
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.delete(f"/api/sessions/{created['id']}").status_code == 204
    assert client.get("/api/sessions").json()["sessions"] == []


def test_web_reports_a_model_failure_as_an_error_event(tmp_path: Path) -> None:
    bundle, _ = make_bundle(tmp_path)

    class Broken:
        def chat(self, *_: Any, **__: Any) -> ChatResponse:
            raise RuntimeError("endpoint down")

    bundle.backends["transport"] = Broken()
    store = SessionStore(bundle, root=tmp_path / "sessions")
    client = TestClient(create_app(store))
    sid = client.post("/api/sessions", json={}).json()["id"]
    with client.stream("POST", f"/api/sessions/{sid}/messages", json={"text": "x"}) as response:
        events = read_events(response)
    # The loop records a model error as a terminated turn rather than raising, so the page gets a turn with the error.
    assert events[-1][0] == "turn" and events[-1][1]["turn"]["termination"] == "model_error" and "endpoint down" in events[-1][1]["turn"]["error"]
