"""The index-stage evaluation read from the agent's own trajectories (scripts/44, 41, 42)."""

from __future__ import annotations

import json
from pathlib import Path

from src.retrieval.supply import (
    RECOMMENDED_CONFIG,
    SEEN_CONFIG,
    PoolRow,
    SupplyQuery,
    judged_metrics,
    observed_supply,
    pool_pairs,
    recommendation_metrics,
    request_feasibility,
    session_positives,
)

TASKS = {
    "rec-1": {"task_id": "rec-1", "kind": "constrained_rec", "positives": ["仙侠", "凡人流"], "negatives_in_text": ["系统"], "negatives_meta": ["后宫"], "sessions": [{"user_message": "想看凡人流仙侠，不要系统，不要后宫。", "expect": {"min_recommendations": 3}}]},
    "mem-1": {"task_id": "mem-1", "kind": "memory", "positives": ["悬疑", "玄幻"], "negatives_in_text": ["兽人", "异能"], "negatives_meta": [], "sessions": [{"user_message": "以后不要兽人。", "expect": {"persist": ["兽人"]}}, {"user_message": "异能也不看。", "expect": {"persist": ["异能"]}}, {"user_message": "推荐几本悬疑。", "expect": {"must_pass": ["兽人", "异能"], "min_recommendations": 1}}]},
    "mem-2": {"task_id": "mem-2", "kind": "memory", "positives": ["玄幻", "武侠"], "negatives_in_text": ["穿越"], "negatives_meta": [], "sessions": [{"user_message": "推荐几本玄幻，这次不要穿越。", "expect": {"min_recommendations": 1}}, {"user_message": "来几本武侠。", "expect": {"min_recommendations": 1}}]},
    "ses-1": {"task_id": "ses-1", "kind": "session_state", "positives": ["玄幻", "群像"], "negatives_in_text": [], "negatives_meta": [], "sessions": [{"user_message": "推荐几本玄幻，要群像。", "expect": {"min_recommendations": 3}}, {"user_message": "第二本主角是谁？", "expect": {"refers_to": 2}}]},
    "qa-1": {"task_id": "qa-1", "kind": "book_qa", "positives": [], "sessions": [{"user_message": "《x》开头讲什么", "expect": {}}]},
}


def test_session_positives_follow_the_request_sessions() -> None:
    assert session_positives(TASKS["rec-1"]) == {0: ("仙侠", "凡人流")}
    assert session_positives(TASKS["mem-1"]) == {2: ("悬疑",)}  # accumulate: the one request takes the first genre
    assert session_positives(TASKS["mem-2"]) == {0: ("玄幻",), 1: ("武侠",)}
    assert session_positives(TASKS["ses-1"]) == {0: ("玄幻", "群像")}  # the follow-up is not a request
    assert session_positives(TASKS["qa-1"]) == {}


def _search(query: str, ids: list[str], **arguments) -> dict:
    return {"tool": "search_books", "arguments": {"query": query, **arguments}, "result": {"results": [{"novel_id": i, "title": f"《{i}》", "score": 0.5} for i in ids]}, "error": None}


def _record(task_id: str, session: int, observations: list[dict], recommendations: list[str] | str | None = None) -> dict:
    structured = {"recommendations": recommendations if isinstance(recommendations, str) else [{"novel_id": i, "title": f"《{i}》"} for i in recommendations or []]}
    return {"task_id": f"{task_id}/s{session}", "metadata": {"task_id": task_id, "session": session}, "steps": [{"observations": observations}], "structured": structured}


def write_trajectories(tmp_path: Path) -> Path:
    records = [
        _record("rec-1", 0, [_search("凡人流 仙侠", ["a", "b", "c"], genre="仙侠", k=10), _search("凡人修仙 宗门", ["a", "d"]), {"tool": "search_books", "arguments": {"query": "坏的"}, "result": None, "error": "boom"}], ["a", "d", "c"]),
        _record("mem-1", 0, [], None),  # memory-only session: no request, nothing taken
        _record("mem-1", 2, [_search("悬疑", ["m", "n"])], ["m"]),
        _record("ses-1", 1, [_search("玄幻 群像", ["s"])], ["s"]),  # the follow-up session is not a request
        _record("qa-1", 0, [_search("《x》", ["x"])], None),
        _record("rec-1", 0, [], '[{"novel_id": "z", "title": "《z》"}]'),  # recommendations sent as a JSON string
    ]
    path = tmp_path / "trajectories.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return path


def test_observed_supply_reads_queries_results_and_recommendations(tmp_path: Path) -> None:
    queries, rows = observed_supply([write_trajectories(tmp_path)], TASKS)
    searches = [q for q in queries if q.source == "trajectory"]
    assert [q.query_id for q in searches] == ["rec-1/s0@1", "rec-1/s0@2", "mem-1/s2@1"]
    assert searches[0].request_id == "rec-1/s0" and searches[0].request.startswith("想看凡人流") and searches[0].kind == "constrained_rec"
    assert searches[0].filters == (("genre", "仙侠"), ("k", "10")) and searches[1].filters == ()
    assert searches[0].negatives_in_text == ("系统",) and searches[0].negatives_meta == ("后宫",)
    assert searches[2].positives == ("悬疑",) and searches[2].negatives_in_text == ("兽人", "异能")
    seen = [(r.query_id, r.rank, r.novel_id) for r in rows if r.config == SEEN_CONFIG]
    assert seen == [("rec-1/s0@1", 1, "a"), ("rec-1/s0@1", 2, "b"), ("rec-1/s0@1", 3, "c"), ("rec-1/s0@2", 1, "a"), ("rec-1/s0@2", 2, "d"), ("mem-1/s2@1", 1, "m"), ("mem-1/s2@1", 2, "n")]
    recommended = [(r.query_id, r.novel_id) for r in rows if r.config == RECOMMENDED_CONFIG]
    assert recommended == [("rec-1/s0@rec", "a"), ("rec-1/s0@rec", "d"), ("rec-1/s0@rec", "c"), ("mem-1/s2@rec", "m"), ("rec-1/s0@rec", "z")]
    # pairs collapse onto the request, so the two searches and the recommendations of rec-1 share labels
    pairs = pool_pairs(rows, 10, queries)
    assert set(pairs) == {("rec-1/s0", i) for i in "abcdz"} | {("mem-1/s2", "m"), ("mem-1/s2", "n")}
    assert pairs[("rec-1/s0", "a")] == "《a》"
    # round trip keeps the new fields
    again = SupplyQuery.from_dict(json.loads(json.dumps(searches[0].to_dict(), ensure_ascii=False)))
    assert again == searches[0] and again.judge_id == "rec-1/s0" and again.judge_text == searches[0].request


def verdict(positive: int, **violations: bool) -> dict:
    return {"positive": positive, "violations": violations, "confidence": "high", "reason": ""}


def test_judged_recommendation_and_feasibility_metrics_share_the_request_verdicts(tmp_path: Path) -> None:
    queries, rows = observed_supply([write_trajectories(tmp_path)], TASKS)
    verdicts = {
        ("rec-1/s0", "a"): verdict(2, 后宫=False),
        ("rec-1/s0", "b"): verdict(1, 后宫=True),
        ("rec-1/s0", "c"): verdict(0, 后宫=False),
        ("rec-1/s0", "d"): verdict(2, 后宫=False),
        ("rec-1/s0", "z"): verdict(1, 后宫=False),
        ("mem-1/s2", "m"): verdict(2),
        ("mem-1/s2", "n"): verdict(1),
    }
    densities = {"d": {"系统": 40.0}, "a": {"系统": 0.0}, "b": {"系统": 0.0}, "c": {"系统": 0.0}}  # d violates the in-text negative
    seen = judged_metrics(rows, queries, verdicts, k=10, densities=densities)[SEEN_CONFIG]
    assert seen["queries_judged"] == 3 and seen["judged_coverage"] == 1.0
    # query 1 (a,b,c): strict 1/3, lenient 2/3; query 2 (a,d): strict 1.0; mem (m,n): strict 0.5 -> means
    assert seen["positive_precision@10"] == round((1 / 3 + 1.0 + 0.5) / 3, 4)
    assert seen["feasible@10"] == 0.0  # no query has three usable candidates
    recommended = recommendation_metrics(rows, queries, verdicts, densities=densities)
    assert recommended["all"]["recommendations"] == 5 and recommended["all"]["judged_coverage"] == 1.0
    # a ok, d violates (density), c positive 0, m ok, z lenient ok -> strict 3/5, lenient 4/5, violation 1/5, usable 3/5
    assert recommended["all"]["satisfied_strict"] == 0.6 and recommended["all"]["satisfied_lenient"] == 0.8
    assert recommended["all"]["violation_rate"] == 0.2 and recommended["all"]["usable_rate"] == 0.6
    assert recommended["all"]["sessions"] == 2 and recommended["all"]["sessions_all_usable"] == 0.5
    assert set(recommended) == {"all", "constrained_rec", "memory"} and recommended["memory"]["usable_rate"] == 1.0
    # feasibility over the union of what each request saw: rec-1 saw a,b,c,d -> lenient usable a (b violates, c 0, d density) = 1 < 3
    assert request_feasibility(rows, queries, verdicts, densities=densities, needed=3) == {"rec-1": False, "mem-1": False}
    assert request_feasibility(rows, queries, verdicts, densities=densities, needed=1) == {"rec-1": True, "mem-1": True}
    assert request_feasibility(rows, queries, verdicts, densities=densities, needed=2, strict=True) == {"rec-1": False, "mem-1": False}
    assert request_feasibility(rows, queries, {}, densities=densities) == {}  # nothing judged, nothing claimed


def test_pool_pairs_without_queries_keeps_the_query_key() -> None:
    rows = [PoolRow("q1", "cfg", 1, "a", "《a》"), PoolRow("q2", "cfg", 1, "a", "《a》")]
    assert set(pool_pairs(rows, 10)) == {("q1", "a"), ("q2", "a")}
