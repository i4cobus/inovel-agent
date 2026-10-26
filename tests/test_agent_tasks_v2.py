"""Extension batches keep earlier ids; session_state and clarify tasks and their scoring."""

from typing import Any

from src.agent_eval.metrics import score_task, session_state_labels
from src.agent_eval.tasks import CLARIFY_PHRASES, synthesize_clarify, synthesize_constrained, synthesize_memory, synthesize_session_state


def test_start_offsets_only_the_ids() -> None:
    base = synthesize_constrained(6, seed=1)
    shifted = synthesize_constrained(6, seed=1, start=128)
    assert [t.task_id for t in shifted] == [f"rec-{128 + i:03d}" for i in range(6)]
    assert [t.sessions[0].user_message for t in shifted] == [t.sessions[0].user_message for t in base]
    assert [t.task_id for t in synthesize_memory(3, seed=1, start=31)] == ["mem-031", "mem-032", "mem-033"]


def test_session_state_tasks_cycle_three_followups() -> None:
    tasks = synthesize_session_state(6, seed=2)
    assert [t.variant for t in tasks] == ["more", "refer", "exclude"] * 2
    for task in tasks:
        assert task.kind == "session_state" and len(task.sessions) == 2
        assert task.sessions[0].expect == {"min_recommendations": 3} and "不要" not in task.sessions[0].user_message
    assert tasks[0].sessions[1].expect == {"min_recommendations": 3, "disjoint_from": 0}
    assert tasks[1].sessions[1].expect == {"refers_to": 2}
    assert tasks[2].sessions[1].expect == {"min_recommendations": 1, "exclude_index": 1}


def test_clarify_tasks_are_distinct_phrasings() -> None:
    tasks = synthesize_clarify(8, seed=5)
    assert len({t.sessions[0].user_message for t in tasks}) == 8
    assert all(t.sessions[0].expect == {"asks_user": True, "max_recommendations": 0} for t in tasks)
    assert all(t.sessions[0].user_message in CLARIFY_PHRASES for t in tasks)


def traj(recs: list[str], asks_user: bool = False, looked_at: list[str] = (), termination: str = "finish") -> dict[str, Any]:
    steps = [{"tool_calls": [{"name": "search_books", "arguments": {"query": "q"}}], "observations": [{"tool": "search_books", "result": {"results": [{"novel_id": r} for r in recs + list(looked_at)]}}], "prompt_tokens": 1, "completion_tokens": 1}]
    steps.append({"tool_calls": [{"name": "get_profile", "arguments": {"novel_id": i}} for i in looked_at], "observations": [], "prompt_tokens": 1, "completion_tokens": 1})
    return {"termination": termination, "final_answer": "a", "structured": {"recommendations": [{"novel_id": r, "title": r} for r in recs], "citations": [], "asks_user": asks_user}, "steps": steps}


def test_session_state_labels_cover_the_three_followups() -> None:
    first = traj(["a", "b", "c"])
    assert session_state_labels({"disjoint_from": 0}, traj(["d", "e", "f"]), [first]) == []
    assert session_state_labels({"disjoint_from": 0}, traj(["a", "e", "f"]), [first]) == ["repeat_recommended"]
    assert session_state_labels({"refers_to": 2}, traj([], looked_at=["b"]), [first]) == []
    assert session_state_labels({"refers_to": 2}, traj([], looked_at=["c"]), [first]) == ["reference_missed"]
    assert session_state_labels({"exclude_index": 1}, traj(["d"]), [first]) == []
    assert session_state_labels({"exclude_index": 1}, traj(["a", "d"]), [first]) == ["excluded_recommended"]


def test_score_task_handles_session_state_and_clarify() -> None:
    task = {"task_id": "ses-000", "kind": "session_state", "variant": "more", "split": "dev",
            "sessions": [{"user_message": "x", "expect": {"min_recommendations": 3}}, {"user_message": "换几本", "expect": {"min_recommendations": 3, "disjoint_from": 0}}]}
    good = score_task(task, [traj(["a", "b", "c"]), traj(["d", "e", "f"])], [{}, {}], {})
    assert good["pass"]
    bad = score_task(task, [traj(["a", "b", "c"]), traj(["a", "b"])], [{}, {}], {})
    assert set(bad["labels"]) == {"repeat_recommended", "no_recommendation"}

    clarify = {"task_id": "clr-000", "kind": "clarify", "variant": "vague", "split": "dev", "sessions": [{"user_message": "推荐点好看的。", "expect": {"asks_user": True, "max_recommendations": 0}}]}
    assert score_task(clarify, [traj([], asks_user=True)], [{}], {})["pass"]
    guessed = score_task(clarify, [traj(["a", "b", "c"], asks_user=False)], [{}], {})
    assert set(guessed["labels"]) == {"no_clarification", "recommended_without_asking"}
