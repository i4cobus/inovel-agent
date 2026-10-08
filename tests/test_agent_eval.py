import json
from pathlib import Path
from typing import Any

from src.agent.loop import AgentConfig, AgentLoop
from src.agent.memory import UserMemory
from src.agent.tools import ToolRegistry, build_check_term, build_memory_tools, build_search_books
from src.agent_eval.metrics import aggregate, format_summary, memory_labels, score_session, score_task
from src.agent_eval.runner import RunPaths, load_run, run_tasks
from src.agent_eval.tasks import Task, assign_splits, load_tasks, save_tasks, synthesize_constrained, synthesize_memory
from src.chat_transport import ChatResponse, TokenUsage, ToolCall
from src.preferences import META_LABEL_NEGATIVES, is_rule_checkable


# ---- synthesis ---------------------------------------------------------------------------


def test_constrained_tasks_are_stratified_deterministic_and_well_formed() -> None:
    tasks = synthesize_constrained(9, seed=1)
    assert [t.variant for t in tasks] == ["in_text", "meta", "mixed"] * 3
    assert synthesize_constrained(9, seed=1)[4].sessions[0].user_message == tasks[4].sessions[0].user_message
    for task in tasks:
        assert all(is_rule_checkable(t) for t in task.negatives_in_text)
        assert all(t in META_LABEL_NEGATIVES for t in task.negatives_meta)
        message = task.sessions[0].user_message
        assert all(neg in message for neg in task.negatives_in_text + task.negatives_meta)
        assert task.sessions[0].expect == {"min_recommendations": 3}


def test_memory_tasks_cycle_variants_with_expectations() -> None:
    tasks = synthesize_memory(6, seed=2)
    assert [t.variant for t in tasks] == ["persist", "accumulate", "oneoff"] * 2
    persist, accumulate, oneoff = tasks[:3]
    assert persist.sessions[0].expect["persist"] == persist.negatives_in_text
    assert persist.sessions[1].expect["must_pass"] == persist.negatives_in_text
    assert len(accumulate.sessions) == 3 and accumulate.sessions[2].expect["must_pass"] == accumulate.negatives_in_text
    assert oneoff.sessions[1].expect["must_not_persist"] == oneoff.negatives_in_text
    assert "这一次" in oneoff.sessions[0].user_message


def test_splits_and_round_trip(tmp_path: Path) -> None:
    tasks = assign_splits(synthesize_constrained(10, seed=3), n_test=4, n_dev=2, seed=3)
    assert len(tasks) == 6 and sum(t.split == "test" for t in tasks) == 4 and sum(t.split == "dev" for t in tasks) == 2
    save_tasks(tmp_path / "t.jsonl", tasks)
    reloaded = load_tasks(tmp_path / "t.jsonl")
    assert [t.to_dict() for t in reloaded] == [t.to_dict() for t in tasks]


# ---- metrics -----------------------------------------------------------------------------


def traj(termination: str, recs: list[str], searched: list[str], checks: dict[str, list[str]] | None = None, tropes: dict[str, list[str]] | None = None, steps: int = 3) -> dict[str, Any]:
    tool_calls = [{"id": "s", "name": "search_books", "arguments": {"query": "q"}}]
    for novel_id, terms in (checks or {}).items():
        tool_calls.append({"id": "c", "name": "check_term", "arguments": {"novel_id": novel_id, "terms": terms}})
    for novel_id, ts in (tropes or {}).items():
        for t in ts:
            tool_calls.append({"id": "t", "name": "check_trope", "arguments": {"novel_id": novel_id, "trope": t}})
    return {
        "termination": termination,
        "structured": {"recommendations": [{"novel_id": r, "title": r} for r in recs], "citations": []} if recs else None,
        "steps": [
            {"index": 0, "tool_calls": tool_calls, "observations": [{"tool": "search_books", "result": [{"novel_id": s} for s in searched]}], "prompt_tokens": 100, "completion_tokens": 10}
        ]
        + [{"index": i, "tool_calls": [], "observations": [], "prompt_tokens": 100, "completion_tokens": 10} for i in range(1, steps)],
        "metadata": {},
    }


DENSITIES = {"clean": {"系统": 0.0}, "dirty": {"系统": 9.0}, "grey": {"系统": 2.0}}


def test_score_session_counts_violations_coverage_and_grounding() -> None:
    good = score_session(traj("finish", ["clean"], ["clean", "dirty"], checks={"clean": ["系统"]}), ["系统"], [], DENSITIES, min_recommendations=1)
    assert good.labels == [] and good.violation_rate == 0.0 and good.term_coverage == 1.0 and good.grounded

    bad = score_session(traj("finish", ["dirty", "ghost"], ["dirty"], checks={}), ["系统"], ["后宫"], DENSITIES, min_recommendations=3)
    assert set(bad.labels) == {"ungrounded_id", "no_recommendation", "constraint_violated", "tool_skipped"}
    assert bad.violation_rate == 1.0 and bad.term_coverage == 0.0 and bad.trope_coverage == 0.0 and not bad.in_corpus

    grey = score_session(traj("finish", ["grey"], ["grey"], checks={"grey": ["系统"]}), ["系统"], [], DENSITIES)
    assert grey.violation_rate is None and grey.unchecked_rate == 1.0 and grey.labels == []

    budget = score_session(traj("max_steps", [], ["clean"]), ["系统"], [], DENSITIES, min_recommendations=1)
    assert set(budget.labels) == {"format", "budget_exhausted", "no_recommendation"}


def test_memory_labels_check_persisted_and_one_off_entries() -> None:
    state = {"negative": [{"value": "不要系统", "persistent": True}, {"value": "后宫", "persistent": False}]}
    assert memory_labels({"persist": ["系统"]}, state) == []
    assert memory_labels({"persist": ["兽人"]}, state) == ["memory_missed"]
    assert memory_labels({"must_not_persist": ["后宫"]}, state) == []
    assert memory_labels({"must_not_persist": ["系统"]}, state) == ["memory_overpersisted"]


def test_score_task_and_aggregate() -> None:
    task = Task("mem-000", "memory", "dev", sessions=[], negatives_in_text=["系统"], variant="persist")
    task_dict = task.to_dict()
    task_dict["sessions"] = [
        {"user_message": "记住", "expect": {"persist": ["系统"], "min_recommendations": 1}},
        {"user_message": "推荐", "expect": {"must_pass": ["系统"], "min_recommendations": 1}},
    ]
    trajs = [traj("finish", ["clean"], ["clean"], checks={"clean": ["系统"]}), traj("finish", ["dirty"], ["dirty"], checks={"dirty": ["系统"]})]
    states = [{"negative": [{"value": "系统", "persistent": True}]}] * 2
    row = score_task(task_dict, trajs, states, DENSITIES)
    assert row["pass"] is False and row["labels"] == ["constraint_violated"] and row["steps"] == 6

    rec_task = synthesize_constrained(1, seed=0)[0].to_dict()
    rec_task["negatives_in_text"], rec_task["negatives_meta"] = ["系统"], []
    ok = score_task(rec_task, [traj("finish", ["clean", "clean", "clean"], ["clean"], checks={"clean": ["系统"]})], [{}], DENSITIES)
    assert ok["pass"] is True

    summary = aggregate([row, ok])
    assert summary["overall"]["n"] == 2 and summary["overall"]["pass_rate"] == 0.5
    assert summary["kind"]["memory"]["labels"] == {"constraint_violated": 1}
    assert "| overall |" in format_summary(summary) and "constraint_violated=1" in format_summary(summary)


# ---- runner ------------------------------------------------------------------------------


class ScriptedAgent:
    """Mimics AgentBundle: a loop over stub tools whose model writes memory then finishes with one book."""

    def __init__(self) -> None:
        self.turn = 0
        self.memory = UserMemory()
        self.memory_path = Path("unused")
        self.tools = ToolRegistry()

        class Searcher:
            def search(self, query: str, k: int) -> list[dict[str, Any]]:
                return [{"novel_id": "clean", "title_guess": "书", "profile_text_preview": "p", "score": 1.0}]

        self.tools.register(build_search_books(Searcher(), AgentConfig().budget))
        self.tools.register(build_check_term(DENSITIES))
        self.reset_memory(self.memory_path)

        class Model:
            def __init__(inner) -> None:
                inner.n = 0

            def chat(inner, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int) -> ChatResponse:
                inner.n += 1
                if inner.n % 2 == 1:
                    calls = (
                        ToolCall("m", "memory_write", {"kind": "negative", "value": "系统", "persistent": True}, "{}"),
                        ToolCall("s", "search_books", {"query": messages[-1]["content"][:5]}, "{}"),
                        ToolCall("c", "check_term", {"novel_id": "clean", "terms": ["系统"]}, "{}"),
                    )
                    return ChatResponse("", calls, TokenUsage(50, 5))
                return ChatResponse("", (ToolCall("f", "finish", {"answer": "推荐《书》", "recommendations": [{"novel_id": "clean", "title": "书"}]}, "{}"),), TokenUsage(60, 6))

        self.loop = AgentLoop(Model(), self.tools, self.memory, AgentConfig(max_steps=4))

    def reset_memory(self, memory_path: Path) -> None:
        self.memory = UserMemory.load(memory_path) if memory_path.exists() else UserMemory()
        self.memory_path = memory_path
        self.turn = 0
        for spec in build_memory_tools(self.memory, turn_counter=lambda: self.turn):
            self.tools.specs[spec.name] = spec
        if hasattr(self, "loop"):
            self.loop.memory = self.memory

    def chat(self, user_message: str, history: list[dict[str, Any]] | None = None, task_id: str = "") -> Any:
        self.turn += 1
        run = self.loop.run(user_message, history=history, task_id=task_id)
        self.memory.save(self.memory_path)
        return run


def test_runner_writes_redacted_trajectories_and_memory_states_per_task(tmp_path: Path) -> None:
    tasks = assign_splits(synthesize_memory(2, seed=0), n_test=0, n_dev=2)
    for task in tasks:  # the stub agent always writes and checks 系统; align the synthesised terms with it
        task.negatives_in_text = ["系统"]
        for session in task.sessions:
            for key in ("persist", "must_pass", "must_not_persist"):
                if key in session.expect:
                    session.expect[key] = ["系统"]
    paths = RunPaths(run_dir=tmp_path / "run", local_dir=tmp_path / "local")
    records = run_tasks(ScriptedAgent(), tasks, paths, {"model": "stub"})
    assert [len(r["trajectories"]) for r in records] == [len(t.sessions) for t in tasks]
    assert json.loads((paths.run_dir / "config.json").read_text(encoding="utf-8"))["tasks"] == 2
    assert (paths.memory_dir / f"{tasks[0].task_id}.json").exists()

    trajectories, states = load_run(paths)
    assert len(trajectories) == sum(len(t.sessions) for t in tasks)
    assert all("sha256" in t["final_answer"] for t in trajectories)  # redacted copy
    assert set(states) == {t.task_id for t in tasks}

    densities = DENSITIES
    rows = [score_task(t.to_dict(), [x for x in trajectories if x["metadata"]["task_id"] == t.task_id], states[t.task_id], densities) for t in tasks]
    assert all(r["pass"] for r in rows), [r["labels"] for r in rows]
