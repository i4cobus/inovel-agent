"""The agent loop: plan, call tools, observe, stop.

Deliberately small. One model turn per step; every tool call in that turn is
executed and appended as an observation; a turn with no tool calls is the final
answer. Three other ways to stop, all recorded in ``termination``:

- ``max_steps``: the step budget ran out; the last assistant text is the answer.
- ``loop``: the same tool was called with the same arguments twice in a row.
- ``model_error``: the transport raised; nothing is guessed.

The structured attachment (recommendations, citations) is the trailing JSON
object of the final answer. A missing or unparseable one is recorded as
``structured=None`` and scored as a format failure, not repaired.
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
    """The last balanced JSON object in the text, or None."""

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
        schemas = self.tools.schemas()
        last_signature: tuple[str, str] | None = None
        last_content = ""

        for index in range(self.config.max_steps):
            started = time.perf_counter()
            try:
                response = self.model.chat(messages, schemas, self.config.max_tokens)
            except Exception as exc:  # noqa: BLE001 - recorded, not guessed around
                trajectory.termination = "model_error"
                trajectory.metadata["error"] = f"{type(exc).__name__}: {exc}"
                trajectory.final_answer = last_content
                break
            latency = time.perf_counter() - started
            messages.append(_assistant_message(response))
            last_content = response.content or last_content

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
                trajectory.termination = "answer"
                break

            looped = False
            for call in response.tool_calls:
                signature = (call.name, json.dumps(call.arguments, ensure_ascii=False, sort_keys=True))
                observation = Observation(tool=call.name, call_id=call.id, arguments=call.arguments)
                call_started = time.perf_counter()
                if call.arguments is None:
                    observation.error = f"参数不是合法 JSON：{call.raw_arguments[:200]}"
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

            if looped:
                trajectory.termination = "loop"
                break
        else:
            trajectory.termination = "max_steps"

        if trajectory.termination != "model_error":
            trajectory.final_answer = last_content
        trajectory.structured = extract_trailing_json(trajectory.final_answer)
        return AgentRun(trajectory=trajectory, messages=messages)
