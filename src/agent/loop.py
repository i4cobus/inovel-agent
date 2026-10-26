"""The agent loop: plan, call tools, observe, finish.

Deliberately small. One model turn per step; every tool call in that turn is
executed and appended as an observation. The run ends when the model calls
the built-in ``finish`` tool, whose arguments *are* the structured answer.
The first three real trajectories (2026-10-08) showed why: asked to append
a JSON block to a free-form answer, Qwen3.5-9B never did, and one run ended
on a dangling 「让我检查……」 with no tool call. A text-only turn is therefore
not an answer: it gets one nudge to continue; a second one in a row ends the
run as ``answer_without_finish``, which the evaluation scores as a format
failure rather than repairing.

Terminations, all recorded on the trajectory:

- ``finish``: the model called finish; ``structured`` holds its arguments.
- ``answer_without_finish``: two consecutive text-only turns.
- ``max_steps``: the step budget ran out.
- ``loop``: the same tool was called with the same arguments twice in a row.
- ``model_error``: the transport raised; nothing is guessed.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agent.context import ContextBudget, build_system_prompt, compact_messages, render_tool_result
from src.agent.memory import UserMemory
from src.preferences import IN_TEXT_NEGATIVES, META_LABEL_NEGATIVES, TERM_SYNONYMS, parse_preference_query
from src.agent.session import SessionState
from src.agent.tools import ToolError, ToolRegistry
from src.agent.trajectory import Observation, Step, Trajectory
from src.chat_transport import ChatModel, ChatResponse
from src.llm_json import split_first_json_object

FINISH_TOOL = "finish"
FINISH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": FINISH_TOOL,
        "description": "给出最终回答并结束。推荐书或回答问题都必须通过这个工具交付；answer 是给用户看的自然语言，recommendations / citations 是结构化附件。",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "给用户的完整回答，自然语言"},
                "recommendations": {
                    "type": "array",
                    "description": "推荐的书，没有就给空列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "novel_id": {"type": "string"},
                            "title": {"type": "string"},
                            "reason": {"type": "string", "description": "为什么推荐，一两句用户能懂的话；不写工具名和内部结果"},
                        },
                        "required": ["novel_id", "title"],
                    },
                },
                "citations": {
                    "type": "array",
                    "description": "回答引用的章节（ask_book 返回的 chapter），没有就给空列表",
                    "items": {
                        "type": "object",
                        "properties": {"novel_id": {"type": "string"}, "chapter": {"type": "string"}},
                        "required": ["novel_id", "chapter"],
                    },
                },
                "asks_user": {"type": "boolean", "description": "answer 是向用户提出的问题、需要用户回答后才能继续时为 true", "default": False},
            },
            "required": ["answer"],
        },
    },
}
NUDGE = "你刚才没有调用任何工具，也没有用 finish 结束。对话只有调用 finish 才算回答：把你刚才那段话放进 finish 的 answer（需要用户回答时 asks_user=true）；还需要更多信息就调用工具。"
MEMORY_ONLY_HINT = "记忆已写入。如果用户这次只是在陈述偏好、没有要你找书，就直接用 finish 结束：answer 里确认已记住什么，并问他现在想看什么（asks_user=true）。如果他这次还提了找书或问书的要求，继续处理，不要只回一句已记住。"
MEMORY_ONLY_TOOLS = frozenset({"memory_write", "set_aside"})  # reads do not count: a turn that starts with memory_read is usually a request
LAST_STEP = "这是最后一步，步数已用尽。现在必须调用 finish，用已经掌握的信息给出最终回答；没能确认的偏好在 answer 里说明。"


def _coerce_recommendations(value: Any) -> tuple[list[Any], str | None]:
    """finish.recommendations as a list of objects; a JSON string holding such a list is accepted (dev-flash-03 rec-064 sent one)."""
    if value is None or value == "":
        return [], None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return [], "recommendations 必须是对象数组（每项含 novel_id 和 title），不是字符串；请修正后再次 finish。"
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        return [], "recommendations 必须是对象数组（每项含 novel_id 和 title）；请修正后再次 finish。"
    return value, None


def _note_violation(violations: dict[str, list[str]], tool: str, arguments: dict[str, Any] | None, result: Any) -> None:
    """Remember a check that came back as a violation, keyed by novel_id, so a later finish can be held to it."""
    if not isinstance(result, dict) or not arguments:
        return
    novel_id = str(result.get("novel_id") or arguments.get("novel_id") or "")
    if not novel_id:
        return
    if tool == "check_term" and result.get("violates") is True:
        evidence = result.get("evidence") or {}
        terms = "、".join(f"{t} 密度 {d}" for t, d in evidence.items()) or "、".join(map(str, arguments.get("terms") or []))
        violations.setdefault(novel_id, []).append(f"check_term 判定含有 {terms}")
    elif tool == "check_trope" and result.get("verdict") == "yes":
        violations.setdefault(novel_id, []).append(f"check_trope 判定有「{result.get('trope') or arguments.get('trope')}」")


def required_in_text_negatives(user_message: str, memory: UserMemory | None) -> list[str]:
    """The in-text negative terms (check_term vocabulary) this turn must check on every pick: the ones the
    user names negatively in this message, plus every term inside a 不要 entry of the memory."""

    found: list[str] = []
    sources: list[str] = list(parse_preference_query(user_message).negative_terms) + _negative_clauses(user_message)
    if memory is not None:
        sources += memory.values("negative")
    for text in sources:
        for label in sorted(META_LABEL_NEGATIVES, key=len, reverse=True):
            text = text.replace(label, " ")  # 恋爱脑 is a trope for check_trope, not the in-text term 恋爱
        for term in sorted(IN_TEXT_NEGATIVES, key=len, reverse=True):
            if term in text and term not in found:
                found.append(term)
    return found


NEGATIVE_MARKERS = ("不要", "不看", "不想看", "不喜欢", "别推", "别来", "别给", "别带", "别有", "排除", "拒绝", "受不了", "讨厌")
CLAUSE_SPLIT = re.compile(r"[，。；！？,.;!?\n]")


def _negative_clauses(text: str) -> list[str]:
    """Clauses of the message that carry a negative marker (也不要重生的 / 这次不看穿越), which the query parser's
    token rules miss; a clause is the unit so 「不要太长的，要系统流」 keeps 系统 out."""

    return [clause for clause in CLAUSE_SPLIT.split(text or "") if any(marker in clause for marker in NEGATIVE_MARKERS)]


def _note_check(checks: dict[str, set[str]], tool: str, arguments: dict[str, Any] | None) -> None:
    """Remember which terms check_term was run with for which book."""

    if tool != "check_term" or not arguments:
        return
    novel_id = str(arguments.get("novel_id") or "")
    terms = arguments.get("terms")
    if novel_id and isinstance(terms, list):
        checks.setdefault(novel_id, set()).update(str(t).strip() for t in terms)


def _covers(checked: set[str], term: str) -> bool:
    forms = {term} | set(TERM_SYNONYMS.get(term, ()))
    return any(c == f or c in f or f in c for c in checked for f in forms)


def steps_hint(remaining: int) -> str:
    return f"[系统提示：还剩 {remaining} 步]"


@dataclass(frozen=True)
class AgentConfig:
    # 10 until 2026-10-27; test-flash-02 lost 11 constrained tasks with 1–2 picks after a refused book and 2 at the budget, so
    # the two guards below now have two more steps to be obeyed in.
    max_steps: int = 12
    # A finish call carries the whole answer plus the recommendation list; at 1,024 it was truncated in 9 of 58 dev sessions (2026-10-12).
    max_tokens: int = 4096
    budget: ContextBudget = ContextBudget()
    # A finish that recommends a book this turn's check_term / check_trope marked as violating is refused once, with the
    # evidence restated; the model may swap the book or resubmit (the second attempt is accepted). Test 2026-10-12:
    # rec-069 / rec-081 recommended books check_term had just marked `violates: true`.
    refuse_violating_recommendations: bool = True
    # A finish whose picks have not all been run through check_term with every in-text negative of this turn (the user's
    # message plus the memory's 不要 entries, exact term or its synonym) is refused once, naming the (book, term) gaps.
    # Test 2026-10-27: 25 tool_skipped, 12 of them memory tasks where the model checked a near-synonym (变身 → 魔改,
    # 修仙 → 玄幻) or checked only part of the picks.
    require_negative_checks: bool = True


@dataclass
class AgentRun:
    trajectory: Trajectory
    messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def final_answer(self) -> str:
        return self.trajectory.final_answer

    @property
    def structured(self) -> dict[str, Any] | None:
        return self.trajectory.structured


def extract_trailing_json(text: str) -> dict[str, Any] | None:
    """The last balanced JSON object in the text, or None. Fallback when finish was never called."""

    remainder = text
    found: dict[str, Any] | None = None
    while True:
        try:
            candidate, remainder = split_first_json_object(remainder)
        except ValueError:
            return found
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            found = parsed


def _assistant_message(response: ChatResponse) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": response.content or None}
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.raw_arguments or json.dumps(call.arguments or {}, ensure_ascii=False)},
            }
            for call in response.tool_calls
        ]
    return message


class AgentLoop:
    def __init__(
        self,
        model: ChatModel,
        tools: ToolRegistry,
        memory: UserMemory | None = None,
        config: AgentConfig = AgentConfig(),
        model_name: str = "",
        session: SessionState | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.memory = memory
        self.config = config
        self.model_name = model_name
        self.session = session

    def run(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None = None,
        task_id: str = "",
        on_step: Callable[[Step], None] | None = None,
    ) -> AgentRun:
        """One turn. ``on_step`` is called after each step's tool calls have been answered (the web UI streams them)."""

        budget = self.config.budget
        messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt(self.memory, budget, self.session)}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user_message})
        trajectory = Trajectory(task_id=task_id, model=self.model_name, user_message=user_message)
        schemas = self.tools.schemas() + [FINISH_SCHEMA]
        last_signature: tuple[str, str] | None = None
        last_content = ""
        text_only_streak = 0
        violations: dict[str, list[str]] = {}  # novel_id -> evidence strings from this turn's checks
        refused_once: set[str] = set()
        checks: dict[str, set[str]] = {}  # novel_id -> terms check_term ran with this turn
        required_terms = required_in_text_negatives(user_message, self.memory) if self.config.require_negative_checks else []
        coverage_refused: set[str] = set()

        tool_names: dict[str, str] = {}
        for index in range(self.config.max_steps):
            remaining = self.config.max_steps - index
            if remaining == 1:
                messages.append({"role": "user", "content": LAST_STEP})
            elif index > 0 and messages[-1].get("role") == "tool":
                previous = trajectory.steps[-1]
                if previous.tool_calls and all(c["name"] in MEMORY_ONLY_TOOLS for c in previous.tool_calls):
                    messages.append({"role": "user", "content": MEMORY_ONLY_HINT})
                else:
                    messages.append({"role": "user", "content": steps_hint(remaining)})
            started = time.perf_counter()
            try:
                response = self.model.chat(
                    compact_messages(messages, tool_names, budget.keep_recent_tool_results), schemas, self.config.max_tokens
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not guessed around
                trajectory.termination = "model_error"
                trajectory.metadata["error"] = f"{type(exc).__name__}: {exc}"
                break
            latency = time.perf_counter() - started
            messages.append(_assistant_message(response))
            if response.content:
                last_content = response.content

            step = Step(
                index=index,
                assistant_content=response.content,
                tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments, "raw_arguments": c.raw_arguments} for c in response.tool_calls],
                observations=[],
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                latency_s=round(latency, 3),
            )
            trajectory.steps.append(step)

            if not response.tool_calls:
                if on_step is not None:
                    on_step(step)
                text_only_streak += 1
                if text_only_streak >= 2:
                    trajectory.termination = "answer_without_finish"
                    break
                messages.append({"role": "user", "content": NUDGE})
                continue
            text_only_streak = 0

            finished = False
            looped = False
            for call in response.tool_calls:
                tool_names[call.id] = call.name
                signature = (call.name, json.dumps(call.arguments, ensure_ascii=False, sort_keys=True))
                observation = Observation(tool=call.name, call_id=call.id, arguments=call.arguments)
                call_started = time.perf_counter()
                if call.arguments is None:
                    observation.error = f"参数不是合法 JSON（如果是回答太长被截断，请精简 answer 后重试）：{call.raw_arguments[:200]}"
                elif call.name == FINISH_TOOL:
                    answer = str(call.arguments.get("answer", "")).strip()
                    recs, recs_error = _coerce_recommendations(call.arguments.get("recommendations"))
                    flagged = [
                        r for r in recs
                        if isinstance(r, dict) and str(r.get("novel_id")) in violations and str(r.get("novel_id")) not in refused_once
                    ] if self.config.refuse_violating_recommendations else []
                    unchecked: list[tuple[dict[str, Any], list[str]]] = []
                    if required_terms and not recs_error:
                        for r in recs:
                            nid = str(r.get("novel_id")) if isinstance(r, dict) else ""
                            if not nid or nid in coverage_refused:
                                continue
                            missing = [t for t in required_terms if not _covers(checks.get(nid, set()), t)]
                            if missing:
                                unchecked.append((r, missing))
                    if not answer:
                        observation.error = "finish 缺少 answer"
                    elif recs_error:
                        observation.error = recs_error
                    elif flagged:
                        lines = []
                        for r in flagged:
                            nid = str(r.get("novel_id"))
                            refused_once.add(nid)
                            lines.append(f"{r.get('title') or nid}：{'；'.join(violations[nid])}")
                        observation.error = (
                            "这些书在本轮核查里已判为违反用户的条件，不能这样推荐：\n" + "\n".join(lines)
                            + "\n请换书，或者在 answer 里说明原因后再次 finish；如果核查的其实是用户想要的元素，也请说明后再次 finish。"
                            "换掉之后不足 3 本，就再 search_books（换一种描述或过滤条件）或 similar_books 补一轮，核查后再交付。"
                        )
                    elif unchecked:
                        lines = []
                        for r, missing in unchecked:
                            nid = str(r.get("novel_id"))
                            coverage_refused.add(nid)
                            lines.append(f"{r.get('title') or nid}：{'、'.join(missing)}")
                        observation.error = (
                            "这些书还没有按用户的负向条件逐本核查过（条件来自用户这次说的和长期记忆里的「不要」项）：\n" + "\n".join(lines)
                            + "\n请对每本书调用 check_term，terms 里填上面列出的原词（不要换成近义词或别的标签），核完再次 finish；判为违反的换书。"
                        )
                    else:
                        observation.result = {"ok": True}
                        trajectory.final_answer = answer
                        trajectory.structured = {
                            "recommendations": recs,
                            "citations": list(call.arguments.get("citations") or []),
                            "asks_user": bool(call.arguments.get("asks_user") or False),
                        }
                        if self.session is not None:
                            self.session.note_recommendations(trajectory.structured["recommendations"])
                        finished = True
                elif signature == last_signature:
                    observation.error = "重复调用：和上一次完全相同的工具和参数。"
                    looped = True
                else:
                    try:
                        observation.result = self.tools.call(call.name, call.arguments)
                        if self.session is not None and isinstance(observation.result, dict):
                            self.session.note_rows(observation.result.get("results"))
                        _note_violation(violations, call.name, call.arguments, observation.result)
                        _note_check(checks, call.name, call.arguments)
                    except ToolError as exc:
                        observation.error = str(exc)
                    except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the run
                        observation.error = f"工具内部错误 {type(exc).__name__}: {exc}"
                observation.latency_s = round(time.perf_counter() - call_started, 3)
                last_signature = signature
                step.observations.append(observation)
                content = observation.error if observation.error is not None else render_tool_result(observation.result, budget)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
            if on_step is not None:
                on_step(step)

            if finished:
                trajectory.termination = "finish"
                break
            if looped:
                trajectory.termination = "loop"
                break
        else:
            trajectory.termination = "max_steps"

        if trajectory.termination != "finish":
            trajectory.final_answer = last_content
            trajectory.structured = extract_trailing_json(last_content) if last_content else None
        return AgentRun(trajectory=trajectory, messages=messages)
