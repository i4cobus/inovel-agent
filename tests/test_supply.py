import json
from pathlib import Path
from typing import Any

import pytest

from src.chat_transport import TokenUsage
from src.judge import BudgetGuard, PricePerMillion
from src.retrieval.cardfilter import CardFilteredSearcher
from src.retrieval.cards import BookCard
from src.retrieval.supply import (
    PoolRow,
    SupplyQuery,
    build_pool,
    calibration_agreement,
    calibration_sample,
    candidate_ok,
    judge_free_metrics,
    judged_metrics,
    legacy_queries,
    load_pool,
    pool_pairs,
    query_element_targets,
    query_genre_targets,
    rewrite_queries,
    split_negatives,
    task_queries,
    trajectory_queries,
    weighted_kappa,
    write_pool,
)
from src.retrieval.supply_judge import (
    SupplyJudgeTask,
    build_evidence,
    build_supply_judge_prompt,
    density_flags,
    estimate_prompt_tokens,
    parse_supply_verdict,
    run_supply_judgements,
    supply_cache_key,
)


def test_task_and_legacy_queries_load_from_the_committed_files() -> None:
    tasks = task_queries()
    assert len(tasks) == 95 and all(q.source == "task" and q.positives for q in tasks)
    rec = next(q for q in tasks if q.query_id == "rec-004")
    assert rec.query == "种田 经营 慢节奏 家长里短" and rec.negatives_meta == ("压抑", "后宫") and rec.negatives_in_text == ()
    legacy = legacy_queries()
    assert len(legacy) == 59 and sum(len(q.anchors) for q in legacy) == 55 and all(q.source == "legacy" for q in legacy)


def test_split_negatives_uses_the_density_rule_vocabulary() -> None:
    in_text, meta = split_negatives(["系统", "后宫", "", "压抑"])
    assert in_text == ("系统",) and meta == ("后宫", "压抑")


def test_rewrites_and_trajectory_queries_inherit_the_task_constraints(tmp_path: Path) -> None:
    base = SupplyQuery("rec-1", "rec-1", "仙侠 凡人流", positives=("仙侠", "凡人流"), negatives_in_text=("系统",), negatives_meta=("后宫",))
    tasks = {"rec-1": base}
    rewrites = tmp_path / "rewrites.jsonl"
    rewrites.write_text(json.dumps({"task_id": "rec-1", "variant": "r1", "query": "想看凡人修仙那种慢慢变强的仙侠"}, ensure_ascii=False) + "\n" + json.dumps({"task_id": "zzz", "query": "x"}) + "\n", encoding="utf-8")
    out = rewrite_queries(rewrites, tasks)
    assert [q.query_id for q in out] == ["rec-1#r1"] and out[0].negatives_meta == ("后宫",) and out[0].source == "rewrite"
    traj = tmp_path / "traj.jsonl"
    record = {"task_id": "rec-1", "steps": [{"observations": [{"tool": "search_books", "arguments": {"query": "凡人流 仙侠 宗门"}}, {"tool": "search_books", "arguments": {"query": "凡人流 仙侠 宗门"}}, {"tool": "get_profile", "arguments": {"novel_id": "a"}}]}]}
    traj.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    out = trajectory_queries([traj], tasks)
    assert len(out) == 1 and out[0].query_id == "rec-1@1" and out[0].source == "trajectory" and out[0].negatives_in_text == ("系统",)
    # the eval runner's redacted trajectories: task_id carries the session suffix, metadata the bare id
    runner_record = {"task_id": "rec-1/s0", "metadata": {"task_id": "rec-1", "session": 0}, "steps": [{"observations": [{"tool": "search_books", "arguments": {"query": "宗门 修仙 慢热"}, "result": "[redacted]"}]}]}
    traj.write_text(json.dumps(runner_record, ensure_ascii=False) + "\n", encoding="utf-8")
    out = trajectory_queries([traj], tasks)
    assert len(out) == 1 and out[0].task_id == "rec-1" and out[0].query == "宗门 修仙 慢热"


class FixedSearcher:
    def __init__(self, name: str, rows: dict[str, list[dict[str, Any]]]) -> None:
        self.name, self.rows = name, rows

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        return self.rows.get(query, [])[:k]


def row(novel_id: str, title: str = "", kind: str = "") -> dict[str, Any]:
    return {"novel_id": novel_id, "title_guess": title or f"《{novel_id}》", "score": 0.5, "section_kind": kind}


QUERIES = [
    SupplyQuery("t1", "t1", "仙侠 凡人流", positives=("仙侠", "凡人流"), negatives_in_text=("系统",), negatives_meta=("后宫",)),
    SupplyQuery("t1#r1", "t1", "凡人修仙那种", positives=("仙侠", "凡人流"), negatives_in_text=("系统",), negatives_meta=("后宫",), source="rewrite"),
    SupplyQuery("legacy:q1", "legacy:q1", "历史 权谋", positives=("历史", "权谋"), source="legacy", anchors=("琅琊榜",)),
]
ROWS = {
    "仙侠 凡人流": [row("a", kind="blurb"), row("b", kind="card"), row("c")],
    "凡人修仙那种": [row("a"), row("d")],
    "历史 权谋": [row("x", "《琅琊榜》"), row("y")],
}
CARDS = {
    "a": BookCard("a", genre="仙侠", subgenre="古典仙侠", elements=["凡人流"]),
    "b": BookCard("b", genre="玄幻", subgenre="东方玄幻", elements=["系统"]),
    "c": BookCard("c", genre="仙侠", subgenre="幻想修仙", elements=[]),
}
DENSITIES = {"a": {"系统": 0.0}, "b": {"系统": 40.0}, "c": {"系统": 0.0}}


def test_pool_round_trip_and_pairs(tmp_path: Path) -> None:
    rows, latencies = build_pool([FixedSearcher("dense", ROWS)], QUERIES, depth=20)
    assert len(rows) == 7 and set(latencies) == {"dense"} and len(latencies["dense"]) == 3
    assert rows[1].section_kind == "card" and rows[0].rank == 1
    write_pool(rows, QUERIES, tmp_path)
    loaded_rows, loaded_queries = load_pool(tmp_path)
    assert loaded_rows == rows and loaded_queries == QUERIES
    assert len(pool_pairs(rows)) == 7 and pool_pairs(rows)[("legacy:q1", "x")] == "《琅琊榜》"


def test_vocabulary_targets_follow_aliases() -> None:
    assert query_genre_targets(["仙侠", "慢热"]) == ({"仙侠"}, set())
    assert query_genre_targets(["修真文明"]) == ({"仙侠"}, {"幻想修仙"})
    assert query_element_targets(["特种兵", "慢热"]) == {"军旅"}


def test_judge_free_metrics_cover_anchors_cards_overlap_density_and_latency() -> None:
    rows, latencies = build_pool([FixedSearcher("dense", ROWS)], QUERIES, depth=20)
    report = judge_free_metrics(rows, QUERIES, k=10, cards=CARDS, densities=DENSITIES, latencies=latencies)["dense"]
    assert report["anchor_hit@10"] == 1.0 and report["anchors"] == 1
    assert report["card_genre_consistency@10"] == round((2 / 3 + 1.0) / 2, 4)  # t1: a, c are 仙侠, b is 玄幻; t1#r1: only 'a' has a card
    assert report["genre_queries"] == 2
    assert report["card_element_consistency@10"] is not None
    assert report["rewrite_overlap@10"] == round(1 / 4, 4) and report["rewrite_pairs"] == 1  # {a,b,c} vs {a,d}
    assert report["density_clean@10"] == round((2 / 3 + 1.0) / 2, 4) and report["density_queries"] == 2  # t1: b violates 系统; t1#r1: d has no density row
    assert report["latency_ms_p50"] is not None


def test_judged_metrics_strict_and_lenient_and_feasibility() -> None:
    rows, _ = build_pool([FixedSearcher("dense", ROWS)], QUERIES, depth=20)
    verdicts = {
        ("t1", "a"): {"positive": 2, "violations": {"后宫": False}},
        ("t1", "b"): {"positive": 2, "violations": {"后宫": False}},  # density says 系统 -> not usable
        ("t1", "c"): {"positive": 1, "violations": {"后宫": True}},
    }
    report = judged_metrics(rows, QUERIES, verdicts, k=10, densities=DENSITIES)["dense"]
    assert report["queries_judged"] == 1
    assert report["positive_precision@10"] == round(2 / 3, 4) and report["positive_precision_lenient@10"] == 1.0
    assert report["feasible@10"] == 0.0 and report["feasible_lenient@10"] == 0.0  # only 'a' is usable
    assert report["judged_coverage"] == round(3 / 7, 4)
    assert candidate_ok(None, None, strict=True) is None
    assert candidate_ok({"positive": 2, "violations": {}}, None, strict=True) is True
    assert candidate_ok({"positive": 2, "violations": {}}, True, strict=True) is False


def test_calibration_sampling_is_stratified_and_agreement_is_scored() -> None:
    verdicts = {}
    for i in range(60):
        verdicts[("q", f"n{i}")] = {"positive": i % 3, "violations": {"后宫": i % 2 == 0}}
    picked = calibration_sample(verdicts, n=12, seed=1)
    assert len(picked) == 12 and len(set(picked)) == 12
    cells = {(verdicts[p]["positive"], verdicts[p]["violations"]["后宫"]) for p in picked}
    assert len(cells) == 6
    human = [{"query_id": "q", "novel_id": f"n{i}", "positive": verdicts[("q", f"n{i}")]["positive"], "violations": {"后宫": i % 2 == 0}, "needed_more": i == 0} for i in range(6)]
    human[0]["positive"] = 2 if human[0]["positive"] != 2 else 0
    report = calibration_agreement(human, verdicts)
    assert report["pairs"] == 6 and report["positive_exact"] == round(5 / 6, 4) and report["violation_agreement"] == 1.0
    assert report["needed_more_share"] == round(1 / 6, 4)
    assert weighted_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == 1.0 and weighted_kappa([], []) is None


def test_card_filtered_searcher_moves_agreeing_books_first_and_passes_through_otherwise() -> None:
    inner = FixedSearcher("dense", {"仙侠 凡人流": [row("b"), row("a"), row("c"), row("z")], "慢热 治愈": [row("b"), row("a")]})
    searcher = CardFilteredSearcher(inner, CARDS, oversample=5)
    out = searcher.search("仙侠 凡人流", 3)
    assert [r["novel_id"] for r in out] == ["a", "c", "b"]  # genre matches first, more element hits first, then the rest
    assert [r["card_match"] for r in out] == ["genre", "genre", ""] and [r["rank"] for r in out] == [1, 2, 3]
    assert searcher.name == "dense+card"
    assert [r["novel_id"] for r in searcher.search("慢热 治愈", 2)] == ["b", "a"]  # no genre named: untouched


def test_evidence_tiers_prompt_and_parsing(tmp_path: Path) -> None:
    text = "简介\n一个少年。\n\n" + "".join(f"第{i}章 标题{i}\n" + ("正文内容。" * 120) + "\n" for i in range(1, 40))
    t1 = build_evidence("T1", "题材：仙侠·古典仙侠", "作者：某\n一个少年", raw_text=text, novel_id="n1")
    t4 = build_evidence("T4", "题材：仙侠·古典仙侠", "作者：某\n一个少年", raw_text=text, novel_id="n1")
    assert "【书卡" in t1 and "【正文摘录】" not in t1 and t4.count("【正文摘录】") == 3 and len(t4) > len(t1)
    with pytest.raises(ValueError):
        build_evidence("T9", "", "")
    flags = density_flags({"系统": 40.0, "异能": 0.0}, ["系统", "异能"])
    assert flags == (("系统", "违反"), ("异能", "未见"))
    task = SupplyJudgeTask("t1", "仙侠 凡人流", "n1", "《书》", t1, "T1", positives=("仙侠", "凡人流"), negatives_meta=("后宫",), density_flags=flags)
    prompt = build_supply_judge_prompt(task)
    assert "系统：违反" in prompt and '"后宫": true|false' in prompt and "候选书名：《书》" in prompt
    assert estimate_prompt_tokens(task) > 450
    verdict = parse_supply_verdict('前言 {"positive": 2, "violations": {"后宫": "true"}, "confidence": "HIGH", "reason": "x"} 后语', ("后宫",))
    assert verdict == {"positive": 2, "violations": {"后宫": True}, "confidence": "high", "reason": "x"}
    assert parse_supply_verdict("no json", ("后宫",)) is None
    assert parse_supply_verdict('{"positive": "many"}', ()) is None
    other = SupplyJudgeTask("t1", "仙侠 凡人流", "n1", "《书》", t4, "T4", negatives_meta=("后宫",))
    assert supply_cache_key(task, "m") != supply_cache_key(other, "m") and supply_cache_key(task, "m") != supply_cache_key(task, "m2")


class FakeTransport:
    def __init__(self, reply: str, fail_for: set[str] = frozenset()) -> None:
        self.reply, self.fail_for, self.calls = reply, fail_for, 0

    def complete_with_usage(self, prompt: str, max_tokens: int) -> tuple[str, TokenUsage]:
        self.calls += 1
        if any(marker in prompt for marker in self.fail_for):
            raise RuntimeError("boom")
        return self.reply, TokenUsage(prompt_tokens=1000, completion_tokens=50)


def test_run_supply_judgements_caches_and_counts_failures(tmp_path: Path) -> None:
    tasks = [SupplyJudgeTask("q", "仙侠", f"n{i}", f"《书{i}》", "证据", "T1", negatives_meta=("后宫",)) for i in range(3)]
    transport = FakeTransport('{"positive": 1, "violations": {"后宫": false}, "confidence": "medium", "reason": "r"}', fail_for={"《书2》"})
    budget = BudgetGuard(limit_usd=10.0, prices=PricePerMillion(12.0, 36.0))
    cache = tmp_path / "cache.jsonl"
    verdicts, summary = run_supply_judgements(tasks, transport, "m", budget, cache_path=cache, workers=2)
    assert set(verdicts) == {("q", "n0"), ("q", "n1")} and summary.judged == 2 and summary.request_failed == 1
    assert summary.spent == pytest.approx(2 * (1000 * 12 + 50 * 36) / 1_000_000)
    verdicts2, summary2 = run_supply_judgements(tasks, transport, "m", budget, cache_path=cache, workers=2)
    assert summary2.cache_hits == 2 and transport.calls == 4  # the failed one is retried, the two good ones come from cache
