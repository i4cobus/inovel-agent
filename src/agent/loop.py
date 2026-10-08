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
import time
from dataclasses import dataclass, field
from typing import Any

from src.agent.context import ContextBudget, build_system_prompt, render_tool_result
from src.agent.memory import UserMemory
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
                            "reason": {"type": "string", "description": "为什么推荐，以及负向约束是怎么核查的"},
                        },
                        "required": ["novel_id", "title"],
                    },
                },
                "citations": {
                    "type": "array",
                    "description": "回答引用的章节，没有就给空列表",
                    "items": {
                        "type": "object",
                        "properties": {"novel_id": {"type": "string"}, "chapter_idx": {"type": "integer"}},
                        "required": ["novel_id", "chapter_idx"],
                    },
                },
            },
            "required": ["answer"],
        },
    },
}
NUDGE = "你刚才没有调用任何工具，也没有用 finish 结束。请继续：需要更多信息就调用工具，信息够了就调用 finish 给出最终回答。"


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 10
    max_tokens: int = 1024
    budget: ContextBudget = ContextBudget()


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
    ) -> None:
        self.model = model
        self.tools = tools
        self.memory = memory
        self.config = config
        self.model_name = model_name

    def run(self, user_message: str, history: list[dict[str, Any]] | None = None, task_id: str = "") -> AgentRun:
        budget = self.config.budget
        messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt(self.memory, budget)}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user_message})
        trajectory = Trajectory(task_id=task_id, model=self.model_name, user_message=user_message)
        schemas = self.tools.schemas() + [FINISH_SCHEMA]
        last_signature: tuple[str, str] | None = None
        last_content = ""
        text_only_streak = 0

        for index in range(self.config.max_steps):
            started = time.perf_counter()
            try:
                response = self.model.chat(messages, schemas, self.config.max_tokens)
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
                signature = (call.name, json.dumps(call.arguments, ensure_ascii=False, sort_keys=True))
                observation = Observation(tool=call.name, call_id=call.id, arguments=call.arguments)
                call_started = time.perf_counter()
                if call.arguments is None:
                    observation.error = f"参数不是合法 JSON：{call.raw_arguments[:200]}"
                elif call.name == FINISH_TOOL:
                    answer = str(call.arguments.get("answer", "")).strip()
                    if not answer:
                        observation.error = "finish 缺少 answer"
                    else:
                        observation.result = {"ok": True}
                        trajectory.final_answer = answer
                        trajectory.structured = {
                            "recommendations": list(call.arguments.get("recommendations") or []),
                            "citations": list(call.arguments.get("citations") or []),
                        }
                        finished = True
                elif signature == last_signature:
                    observation.error = "重复调用：和上一次完全相同的工具和参数。"
                    looped = True
                else:
                    try:
                        observation.result = self.tools.call(call.name, call.arguments)
                    except ToolError as exc:
                        observation.error = str(exc)
                    except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the run
                        observation.error = f"工具内部错误 {type(exc).__name__}: {exc}"
                observation.latency_s = round(time.perf_counter() - call_started, 3)
                last_signature = signature
                step.observations.append(observation)
                content = observation.error if observation.error is not None else render_tool_result(observation.result, budget)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

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
