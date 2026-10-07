import json
from typing import Any

from src.agent.context import ContextBudget
from src.agent.loop import AgentConfig, AgentLoop, extract_trailing_json
from src.agent.memory import UserMemory
from src.agent.tools import ToolError, ToolRegistry, ToolSpec
from src.chat_transport import ChatResponse, TokenUsage, ToolCall


def call(name: str, arguments: dict[str, Any] | None, call_id: str = "c1", raw: str | None = None) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments, raw_arguments=raw if raw is not None else json.dumps(arguments))


def turn(content: str = "", calls: tuple[ToolCall, ...] = ()) -> ChatResponse:
    return ChatResponse(content=content, tool_calls=calls, usage=TokenUsage(10, 5))


class ScriptedModel:
    """Replays assistant turns in order and records what it was sent."""

    def __init__(self, turns: list[ChatResponse]) -> None:
        self.turns = list(turns)
        self.seen: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int) -> ChatResponse:
        self.seen.append([dict(m) for m in messages])
        if not self.turns:
            raise RuntimeError("script exhausted")
        return self.turns.pop(0)


def registry_with_echo() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="echo",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            handler=lambda text: {"echoed": text},
        )
    )

    def boom(**_: Any) -> Any:
        raise ToolError("没有这本书")

    registry.register(ToolSpec(name="boom", description="fails", parameters={"type": "object", "properties": {}, "required": []}, handler=boom))
    return registry


FINAL = '好的。\n{"recommendations": [{"novel_id": "n1", "title": "书", "reason": "r"}], "citations": []}'


def test_tool_call_then_answer_records_two_steps_and_structured_answer() -> None:
    model = ScriptedModel([turn(calls=(call("echo", {"text": "hi"}),)), turn(FINAL)])
    run = AgentLoop(model, registry_with_echo(), UserMemory(), AgentConfig(max_steps=5)).run("找书")

    traj = run.trajectory
    assert traj.termination == "answer"
    assert traj.step_count == 2 and traj.tool_call_count == 1
    assert traj.steps[0].observations[0].result == {"echoed": "hi"}
    assert traj.structured == {"recommendations": [{"novel_id": "n1", "title": "书", "reason": "r"}], "citations": []}
    assert traj.prompt_tokens == 20 and traj.completion_tokens == 10

    # The tool result went back to the model as a tool message tied to the call id.
    second_call_messages = model.seen[1]
    assert second_call_messages[-1]["role"] == "tool"
    assert second_call_messages[-1]["tool_call_id"] == "c1"
    assert json.loads(second_call_messages[-1]["content"]) == {"echoed": "hi"}
    assert second_call_messages[0]["role"] == "system"


def test_tool_error_becomes_an_observation_not_a_crash() -> None:
    model = ScriptedModel([turn(calls=(call("boom", {}),)), turn("抱歉")])
    run = AgentLoop(model, registry_with_echo()).run("x")
    assert run.trajectory.steps[0].observations[0].error == "没有这本书"
    assert run.trajectory.termination == "answer"
    assert run.trajectory.structured is None


def test_unknown_tool_and_bad_json_arguments_are_distinct_errors() -> None:
    model = ScriptedModel(
        [turn(calls=(call("nope", {}, "a"), call("echo", None, "b", raw="{bad json"))), turn("done")]
    )
    run = AgentLoop(model, registry_with_echo()).run("x")
    observations = run.trajectory.steps[0].observations
    assert "未知工具" in observations[0].error
    assert "参数不是合法 JSON" in observations[1].error


def test_identical_consecutive_calls_terminate_as_loop() -> None:
    model = ScriptedModel([turn(calls=(call("echo", {"text": "a"}),)), turn(calls=(call("echo", {"text": "a"}),)), turn("never")])
    run = AgentLoop(model, registry_with_echo()).run("x")
    assert run.trajectory.termination == "loop"
    assert run.trajectory.step_count == 2
    assert "重复调用" in run.trajectory.steps[1].observations[0].error


def test_step_budget_exhaustion_is_recorded() -> None:
    model = ScriptedModel([turn(calls=(call("echo", {"text": str(i)}),)) for i in range(3)])
    run = AgentLoop(model, registry_with_echo(), config=AgentConfig(max_steps=3)).run("x")
    assert run.trajectory.termination == "max_steps"
    assert run.trajectory.step_count == 3


def test_model_failure_is_recorded_and_nothing_is_invented() -> None:
    run = AgentLoop(ScriptedModel([]), registry_with_echo()).run("x")
    assert run.trajectory.termination == "model_error"
    assert run.trajectory.final_answer == ""
    assert "script exhausted" in run.trajectory.metadata["error"]


def test_memory_summary_is_injected_into_the_system_prompt() -> None:
    memory = UserMemory()
    memory.write("negative", "系统", persistent=True)
    model = ScriptedModel([turn("ok")])
    AgentLoop(model, registry_with_echo(), memory).run("x")
    assert "不要：系统" in model.seen[0][0]["content"]


def test_tool_results_are_truncated_to_the_budget() -> None:
    model = ScriptedModel([turn(calls=(call("echo", {"text": "x" * 500}),)), turn("ok")])
    config = AgentConfig(budget=ContextBudget(tool_result_chars=100))
    AgentLoop(model, registry_with_echo(), config=config).run("x")
    assert len(model.seen[1][-1]["content"]) == 100


def test_extract_trailing_json_takes_the_last_object() -> None:
    assert extract_trailing_json('先说 {"a": 1}，最后 {"b": 2}') == {"b": 2}
    assert extract_trailing_json("没有 JSON") is None
    assert extract_trailing_json('{"a": [1, 2]} 尾巴') == {"a": [1, 2]}
