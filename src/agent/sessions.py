"""Conversations as sessions: each one has its own memory, its own session state and its own history.

A session is what one line of enquiry lives in. The user who wanted 凡人流仙侠 in one session opens
another to look for something else, and nothing carries over: not the memory entries, not the
numbered recommendations, not the transcript. Technically a session is an ``AgentBundle.fork``
over the shared backends (embedder, index, profiles, judge, transport) with a memory file of its
own under ``<root>/<session id>/``.

Everything is persisted so the web server can be restarted: ``session.json`` holds the turns and
the session state, ``memory.json`` the agent's memory, ``trajectories.jsonl`` the full trajectories
(local only; they may quote corpus text).
"""

from __future__ import annotations

import json
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.agent.session import SessionState
from src.agent.trajectory import Step, Trajectory
from src.config import DATA_DIR

DEFAULT_SESSIONS_DIR = DATA_DIR / "sessions"
TITLE_CHARS = 24
# Turns replayed verbatim into the next model call; older ones are dropped (memory and session state carry what matters).
HISTORY_TURNS = 8


class SessionError(Exception):
    pass


class SessionNotFound(SessionError):
    pass


class SessionBusy(SessionError):
    """A turn is already running in this session; one at a time."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _preview(value: Any, limit: int = 120) -> Any:
    """Arguments as the UI shows them: long strings cut, structure kept."""

    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 1] + "…"
    if isinstance(value, dict):
        return {k: _preview(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_preview(v, limit) for v in value[:20]]
    return value


def describe_result(tool: str, result: Any) -> str:
    """One line per tool result, for the step trace in the UI."""

    if result is None:
        return ""
    if isinstance(result, dict):
        rows = result.get("results")
        if isinstance(rows, list):
            titles = "、".join(f"《{r.get('title', '')}》" for r in rows[:8] if isinstance(r, dict))
            extra = []
            if result.get("filter"):
                extra.append(f"筛选 {result['filter']}")
            if result.get("filter_matches") is not None:
                extra.append(f"命中 {result['filter_matches']} 本")
            if result.get("excluded"):
                extra.append(f"跳过已推荐 {result['excluded']}")
            head = f"{len(rows)} 本：{titles}" if rows else "0 本"
            return head + (f"（{'；'.join(extra)}）" if extra else "")
        if tool == "ask_book":
            passages = result.get("passages") or []
            return f"{result.get('coverage', '')}；读了 {len(passages)} 段"
        if tool == "get_profile":
            return f"《{result.get('title', '')}》档案"
        if tool == "check_trope":
            parts = [f"{result.get('trope', '')}：{result.get('verdict', '')}"]
            if result.get("confidence"):
                parts.append(f"置信 {result['confidence']}")
            if result.get("reason"):
                parts.append(str(result["reason"])[:80])
            return "；".join(parts)
        if tool == "check_term":
            verdict = result.get("violates")
            label = "违反" if verdict else ("通过" if verdict is False else "规则判不了")
            evidence = "，".join(f"{k} {v}" for k, v in (result.get("evidence") or {}).items())
            return f"{label}（{evidence}）" if evidence else label
        if tool == "memory_write":
            entry = result.get("written") or {}
            return "已记住（长期）" if entry.get("persistent", True) else "已记住（仅本次）"
        if tool == "memory_read":
            return "；".join(f"{kind} {len(items)} 条" for kind, items in result.items() if isinstance(items, list) and items) or "记忆为空"
        if tool == "set_aside":
            return f"排除 {len(result.get('excluded') or [])} 本"
        if tool == "finish":
            return "结束"
    text = json.dumps(_preview(result, 60), ensure_ascii=False)
    return text[:200]


def summarize_step(step: Step) -> dict[str, Any]:
    """The step as the UI renders it: what the model said, which tools it called, what came back."""

    calls = []
    observations = {obs.call_id: obs for obs in step.observations}
    for call in step.tool_calls:
        obs = observations.get(call["id"])
        item: dict[str, Any] = {"name": call["name"], "arguments": _preview(call.get("arguments"))}
        if obs is not None:
            item["latency_s"] = obs.latency_s
            if obs.error:
                item["error"] = obs.error
            else:
                item["summary"] = describe_result(call["name"], obs.result)
        calls.append(item)
    return {
        "index": step.index,
        "content": step.assistant_content or "",
        "calls": calls,
        "prompt_tokens": step.prompt_tokens,
        "completion_tokens": step.completion_tokens,
        "latency_s": step.latency_s,
    }


@dataclass
class Turn:
    index: int
    user: str
    answer: str
    structured: dict[str, Any] | None
    termination: str
    steps: list[dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    created_at: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "user": self.user,
            "answer": self.answer,
            "structured": self.structured,
            "termination": self.termination,
            "steps": self.steps,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_s": self.elapsed_s,
            "created_at": self.created_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Turn":
        return cls(
            index=int(data["index"]),
            user=str(data["user"]),
            answer=str(data.get("answer", "")),
            structured=data.get("structured"),
            termination=str(data.get("termination", "")),
            steps=list(data.get("steps") or []),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
            created_at=str(data.get("created_at", "")),
            error=data.get("error"),
        )

    @classmethod
    def from_trajectory(cls, index: int, trajectory: Trajectory, elapsed_s: float) -> "Turn":
        return cls(
            index=index,
            user=trajectory.user_message,
            answer=trajectory.final_answer,
            structured=trajectory.structured,
            termination=trajectory.termination,
            steps=[summarize_step(step) for step in trajectory.steps],
            prompt_tokens=trajectory.prompt_tokens,
            completion_tokens=trajectory.completion_tokens,
            elapsed_s=round(elapsed_s, 2),
            created_at=_now(),
            error=trajectory.metadata.get("error"),
        )


@dataclass
class ChatSession:
    id: str
    directory: Path
    agent: Any  # an AgentBundle fork
    title: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    turns: list[Turn] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def history(self, max_turns: int = HISTORY_TURNS) -> list[dict[str, str]]:
        """The transcript the next turn is run with: user and assistant texts of the last ``max_turns`` turns.

        Tool traffic is never replayed. Older turns are dropped, not summarised: what matters from them
        is already carried separately, the preferences in memory and the recommended books in the
        session state, both rendered into the system prompt every turn. A one-line note tells the
        model that earlier turns exist so it does not treat the window as the whole conversation."""

        kept = self.turns[-max_turns:] if max_turns > 0 else []
        messages: list[dict[str, str]] = []
        dropped = len(self.turns) - len(kept)
        if dropped > 0:
            messages.append({"role": "user", "content": f"（更早的 {dropped} 轮对话已省略；用户的长期偏好见记忆，已推荐过的书见会话状态。）"})
            messages.append({"role": "assistant", "content": "好的。"})
        for turn in kept:
            messages.append({"role": "user", "content": turn.user})
            if turn.answer:
                messages.append({"role": "assistant", "content": turn.answer})
        return messages

    @property
    def state(self) -> SessionState:
        return self.agent.session

    def state_dict(self) -> dict[str, Any]:
        state = self.state
        return {"recommended": list(state.recommended), "excluded": sorted(state.excluded), "shown": dict(state.shown)}

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": len(self.turns),
            "busy": self.lock.locked(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "turns": [turn.to_dict() for turn in self.turns],
            "memory": self.agent.memory.to_dict(),
            "state": self.state_dict(),
        }

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": [turn.to_dict() for turn in self.turns],
            "state": self.state_dict(),
        }
        tmp = self.directory / "session.json.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.directory / "session.json")


class SessionStore:
    """Creates, finds and runs sessions over one shared agent bundle."""

    def __init__(self, bundle: Any, root: Path = DEFAULT_SESSIONS_DIR, keep_trajectories: bool = True, history_turns: int = HISTORY_TURNS) -> None:
        self.bundle = bundle
        self.root = root
        self.keep_trajectories = keep_trajectories
        self.history_turns = history_turns
        self.sessions: dict[str, ChatSession] = {}
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.root.glob("*/session.json")):
            try:
                self._restore(path)
            except (OSError, ValueError, KeyError) as exc:
                # A damaged session file is skipped, not fatal: the others still load.
                print(f"[sessions] skipping {path}: {type(exc).__name__}: {exc}")

    def _restore(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        directory = path.parent
        agent = self.bundle.fork(directory / "memory.json")
        session = ChatSession(
            id=str(data["id"]),
            directory=directory,
            agent=agent,
            title=str(data.get("title", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            turns=[Turn.from_dict(t) for t in data.get("turns", [])],
        )
        state = data.get("state") or {}
        agent.session.recommended = [dict(r) for r in state.get("recommended", []) if isinstance(r, dict)]
        agent.session.excluded = set(str(n) for n in state.get("excluded", []))
        agent.session.shown = {str(k): str(v) for k, v in (state.get("shown") or {}).items()}
        agent.turn = len(session.turns)
        self.sessions[session.id] = session

    def create(self, title: str = "") -> ChatSession:
        with self._lock:
            while True:
                session_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
                if session_id not in self.sessions and not (self.root / session_id).exists():
                    break
            directory = self.root / session_id
            session = ChatSession(id=session_id, directory=directory, agent=self.bundle.fork(directory / "memory.json"), title=title.strip())
            session.save()
            self.sessions[session_id] = session
            return session

    def get(self, session_id: str) -> ChatSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise SessionNotFound(session_id)
        return session

    def list(self) -> list[dict[str, Any]]:
        return [s.summary() for s in sorted(self.sessions.values(), key=lambda s: s.updated_at, reverse=True)]

    def delete(self, session_id: str) -> None:
        session = self.get(session_id)
        if session.lock.locked():
            raise SessionBusy(session_id)
        with self._lock:
            self.sessions.pop(session_id, None)
        shutil.rmtree(session.directory, ignore_errors=True)

    def rename(self, session_id: str, title: str) -> ChatSession:
        session = self.get(session_id)
        session.title = title.strip()
        session.updated_at = _now()
        session.save()
        return session

    def send(self, session_id: str, text: str, on_step: Callable[[dict[str, Any]], None] | None = None) -> Turn:
        """Run one turn in the session. ``on_step`` receives each step's summary as it completes."""

        session = self.get(session_id)
        text = text.strip()
        if not text:
            raise ValueError("empty message")
        if not session.lock.acquire(blocking=False):
            raise SessionBusy(session_id)
        try:
            index = len(session.turns) + 1
            callback = (lambda step: on_step(summarize_step(step))) if on_step is not None else None
            started = time.perf_counter()
            run = session.agent.chat(text, history=session.history(self.history_turns), task_id=f"{session.id}-t{index}", on_step=callback)
            turn = Turn.from_trajectory(index, run.trajectory, time.perf_counter() - started)
            session.turns.append(turn)
            if not session.title:
                session.title = text[:TITLE_CHARS]
            session.updated_at = _now()
            session.save()
            if self.keep_trajectories:
                with (session.directory / "trajectories.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(run.trajectory.to_dict(), ensure_ascii=False) + "\n")
            return turn
        finally:
            session.lock.release()
