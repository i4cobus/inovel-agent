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


FINISH_ARGS = {"answer": "好的。", "recommendations": [{"novel_id": "n1", "title": "书", "reason": "r"}]}


def test_tool_call_then_finish_records_two_steps_and_structured_answer() -> None:
    model = ScriptedModel([turn(calls=(call("echo", {"text": "hi"}),)), turn(calls=(call("finish", FINISH_ARGS, "f"),))])
    run = AgentLoop(model, registry_with_echo(), UserMemory(), AgentConfig(max_steps=5)).run("找书")

    traj = run.trajectory
    assert traj.termination == "finish"
    assert traj.step_count == 2 and traj.tool_call_count == 2
    assert traj.steps[0].observations[0].result == {"echoed": "hi"}
    assert traj.final_answer == "好的。"
    assert traj.structured == {"recommendations": [{"novel_id": "n1", "title": "书", "reason": "r"}], "citations": [], "asks_user": False}
    assert traj.prompt_tokens == 20 and traj.completion_tokens == 10
    assert model.seen[0][-1]["role"] == "user"

    # The tool result went back to the model as a tool message tied to the call id.
    second_call_messages = model.seen[1]
    last_tool = [m for m in second_call_messages if m["role"] == "tool"][-1]
    assert last_tool["tool_call_id"] == "c1"
    assert json.loads(last_tool["content"]) == {"echoed": "hi"}
    assert second_call_messages[0]["role"] == "system"
    assert second_call_messages[-1]["content"].startswith("[系统提示：还剩")


def test_tool_error_becomes_an_observation_not_a_crash() -> None:
    model = ScriptedModel([turn(calls=(call("boom", {}),)), turn(calls=(call("finish", {"answer": "抱歉"}),))])
    run = AgentLoop(model, registry_with_echo()).run("x")
    assert run.trajectory.steps[0].observations[0].error == "没有这本书"
    assert run.trajectory.termination == "finish"
    assert run.trajectory.structured == {"recommendations": [], "citations": [], "asks_user": False}


def test_text_only_turn_gets_one_nudge_then_ends_without_finish() -> None:
    model = ScriptedModel([turn("让我检查这些书"), turn("还是只有文字 {\"a\": 1}")])
    run = AgentLoop(model, registry_with_echo()).run("x")
    traj = run.trajectory
    assert traj.termination == "answer_without_finish" and traj.step_count == 2
    assert model.seen[1][-1] == {"role": "user", "content": __import__("src.agent.loop", fromlist=["NUDGE"]).NUDGE}
    assert traj.final_answer.startswith("还是只有文字") and traj.structured == {"a": 1}


def test_nudge_then_tool_call_resets_the_streak() -> None:
    model = ScriptedModel([turn("让我想想"), turn(calls=(call("echo", {"text": "a"}),)), turn("再想想"), turn(calls=(call("finish", {"answer": "完"}),))])
    run = AgentLoop(model, registry_with_echo()).run("x")
    assert run.trajectory.termination == "finish" and run.trajectory.step_count == 4


def test_finish_without_answer_is_an_error_and_the_run_continues() -> None:
    model = ScriptedModel([turn(calls=(call("finish", {"answer": ""}),)), turn(calls=(call("finish", {"answer": "ok"}),))])
    run = AgentLoop(model, registry_with_echo()).run("x")
    assert run.trajectory.steps[0].observations[0].error == "finish 缺少 answer"
    assert run.trajectory.termination == "finish" and run.trajectory.final_answer == "ok"


def test_finish_schema_is_offered_alongside_registered_tools() -> None:
    model = ScriptedModel([turn(calls=(call("finish", {"answer": "ok"}),))])
    loop = AgentLoop(model, registry_with_echo())
    loop.run("x")
    # schemas are passed on every call; the scripted model does not record them, so check the loop's view
    assert [s["function"]["name"] for s in loop.tools.schemas()] == ["echo", "boom"]


def test_unknown_tool_and_bad_json_arguments_are_distinct_errors() -> None:
    model = ScriptedModel(
        [turn(calls=(call("nope", {}, "a"), call("echo", None, "b", raw="{bad json"))), turn(calls=(call("finish", {"answer": "done"}),))]
    )
    run = AgentLoop(model, registry_with_echo()).run("x")
    observations = run.trajectory.steps[0].observations
    assert "未知工具" in observations[0].error
    assert "参数不是合法 JSON" in observations[1].error


def test_identical_consecutive_calls_terminate_as_loop() -> None:
    model = ScriptedModel([turn(calls=(call("echo", {"text": "a"}),)), turn(calls=(call("echo", {"text": "a"}),)), turn(calls=(call("finish", {"answer": "never"}),))])
    run = AgentLoop(model, registry_with_echo()).run("x")
    assert run.trajectory.termination == "loop"
    assert run.trajectory.step_count == 2
    assert "重复调用" in run.trajectory.steps[1].observations[0].error


def test_step_budget_exhaustion_is_recorded_after_a_forced_finish_prompt() -> None:
    from src.agent.loop import LAST_STEP, steps_hint

    model = ScriptedModel([turn(calls=(call("echo", {"text": str(i)}),)) for i in range(3)])
    run = AgentLoop(model, registry_with_echo(), config=AgentConfig(max_steps=3)).run("x")
    assert run.trajectory.termination == "max_steps"
    assert run.trajectory.step_count == 3
    assert model.seen[1][-1] == {"role": "user", "content": steps_hint(2)}
    assert model.seen[2][-1] == {"role": "user", "content": LAST_STEP}


def test_old_tool_results_are_compacted_but_the_trajectory_keeps_them() -> None:
    turns = [turn(calls=(call("echo", {"text": "x" * 599 + str(i)}, f"c{i}"),)) for i in range(4)] + [turn(calls=(call("finish", {"answer": "ok"}),))]
    model = ScriptedModel(turns)
    config = AgentConfig(max_steps=10, budget=ContextBudget(keep_recent_tool_results=2))
    run = AgentLoop(model, registry_with_echo(), config=config).run("x")
    sent = model.seen[-1]
    tool_messages = [m for m in sent if m["role"] == "tool"]
    assert len(tool_messages) == 4
    assert all(len(m["content"]) <= 400 for m in tool_messages[:2])  # compacted (generic fallback: 400 chars)
    assert all("x" * 599 in m["content"] for m in tool_messages[2:])  # recent ones verbatim
    assert run.trajectory.steps[0].observations[0].result == {"echoed": "x" * 599 + "0"}


def test_model_failure_is_recorded_and_nothing_is_invented() -> None:
    run = AgentLoop(ScriptedModel([]), registry_with_echo()).run("x")
    assert run.trajectory.termination == "model_error"
    assert run.trajectory.final_answer == "" and run.trajectory.structured is None
    assert "script exhausted" in run.trajectory.metadata["error"]


def test_memory_summary_is_injected_into_the_system_prompt() -> None:
    memory = UserMemory()
    memory.write("negative", "系统", persistent=True)
    model = ScriptedModel([turn(calls=(call("finish", {"answer": "ok"}),))])
    AgentLoop(model, registry_with_echo(), memory).run("x")
    assert "不要：系统" in model.seen[0][0]["content"]


def test_tool_results_are_truncated_to_the_budget() -> None:
    model = ScriptedModel([turn(calls=(call("echo", {"text": "x" * 500}),)), turn(calls=(call("finish", {"answer": "ok"}),))])
    config = AgentConfig(budget=ContextBudget(tool_result_chars=100))
    AgentLoop(model, registry_with_echo(), config=config).run("x")
    assert len([m for m in model.seen[1] if m["role"] == "tool"][-1]["content"]) == 100


def test_extract_trailing_json_takes_the_last_object() -> None:
    assert extract_trailing_json('先说 {"a": 1}，最后 {"b": 2}') == {"b": 2}
    assert extract_trailing_json("没有 JSON") is None
    assert extract_trailing_json('{"a": [1, 2]} 尾巴') == {"a": [1, 2]}


def registry_with_checks() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="check_term",
            description="t",
            parameters={"type": "object", "properties": {"novel_id": {"type": "string"}, "terms": {"type": "array"}}, "required": ["novel_id", "terms"]},
            handler=lambda novel_id, terms: {"novel_id": novel_id, "violates": novel_id == "bad", "evidence": {t: 3.5 for t in terms}, "not_checkable": [], "note": ""},
        )
    )
    registry.register(
        ToolSpec(
            name="check_trope",
            description="t",
            parameters={"type": "object", "properties": {"novel_id": {"type": "string"}, "trope": {"type": "string"}}, "required": ["novel_id", "trope"]},
            handler=lambda novel_id, trope: {"novel_id": novel_id, "trope": trope, "verdict": "yes" if novel_id == "harem" else "no", "quotes": []},
        )
    )
    registry.register(ToolSpec(name="memory_write", description="m", parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}, handler=lambda value: {"written": value}))
    return registry


def finish_with(*ids: str) -> ToolCall:
    return call("finish", {"answer": "推荐如下", "recommendations": [{"novel_id": i, "title": f"《{i}》"} for i in ids]}, "f")


def test_finish_recommending_a_book_check_term_marked_violating_is_refused_once_with_the_evidence() -> None:
    model = ScriptedModel([
        turn(calls=(call("check_term", {"novel_id": "bad", "terms": ["重生"]}, "c1"), call("check_term", {"novel_id": "ok", "terms": ["重生"]}, "c2"))),
        turn(calls=(finish_with("bad", "ok"),)),
        turn(calls=(finish_with("ok", "other"),)),
    ])
    run = AgentLoop(model, registry_with_checks(), UserMemory(), AgentConfig(max_steps=6)).run("不要重生")
    traj = run.trajectory
    refused = traj.steps[1].observations[0]
    assert refused.error is not None and "《bad》" in refused.error and "重生 密度 3.5" in refused.error and "《ok》" not in refused.error
    assert traj.termination == "finish"
    assert [r["novel_id"] for r in traj.structured["recommendations"]] == ["ok", "other"]
    # The refusal reached the model as the tool message for the finish call.
    assert "不能这样推荐" in [m for m in model.seen[2] if m["role"] == "tool"][-1]["content"]


def test_resubmitting_the_same_book_after_the_refusal_is_accepted_and_check_trope_yes_counts_too() -> None:
    model = ScriptedModel([
        turn(calls=(call("check_trope", {"novel_id": "harem", "trope": "后宫"}, "c1"),)),
        turn(calls=(finish_with("harem"),)),
        turn(calls=(finish_with("harem"),)),
    ])
    traj = AgentLoop(model, registry_with_checks(), UserMemory(), AgentConfig(max_steps=6)).run("不要后宫").trajectory
    assert "后宫" in traj.steps[1].observations[0].error
    assert traj.termination == "finish" and traj.step_count == 3
    assert traj.structured["recommendations"][0]["novel_id"] == "harem"


def test_the_refusal_can_be_switched_off() -> None:
    model = ScriptedModel([turn(calls=(call("check_term", {"novel_id": "bad", "terms": ["系统"]}, "c1"),)), turn(calls=(finish_with("bad"),))])
    traj = AgentLoop(model, registry_with_checks(), UserMemory(), AgentConfig(max_steps=6, refuse_violating_recommendations=False)).run("x").trajectory
    assert traj.termination == "finish" and traj.step_count == 2


def test_a_memory_only_step_is_followed_by_the_finish_hint_instead_of_the_step_counter() -> None:
    model = ScriptedModel([
        turn(calls=(call("memory_write", {"value": "不要穿越"}, "c1"),)),
        turn(calls=(call("finish", {"answer": "已记住：不要穿越。想看什么？", "asks_user": True}, "f"),)),
    ])
    run = AgentLoop(model, registry_with_checks(), UserMemory(), AgentConfig(max_steps=6)).run("以后都不要穿越")
    last_user = [m for m in model.seen[1] if m["role"] == "user"][-1]["content"]
    assert last_user.startswith("记忆已更新") and "finish" in last_user
    assert run.trajectory.termination == "finish" and run.trajectory.structured["asks_user"] is True
