from pathlib import Path

import pytest

from src.agent.tools import ToolError
from src.agent.trope import CARD_ELEMENT_RULES, CARD_GENRE_RULES, CARD_SCALE_EXTREMES, CARD_STYLE_RULES, CARD_SUBGENRE_RULES, TROPE_GLOSSARY, CachedTropeJudge, build_trope_prompt, card_verdict, parse_trope_verdict
from src.retrieval.card_schema import ELEMENTS, GENRES, STYLE_OPTIONS, SUBGENRE_TO_GENRE
from src.retrieval.cards import BookCard
from src.evidence import judge_chapter_indices, sample_judge_evidence
from src.preferences import META_LABEL_NEGATIVES
from src.split_chapters import split_chapters


def test_glossary_covers_every_meta_label_the_rule_cannot_check() -> None:
    assert META_LABEL_NEGATIVES <= set(TROPE_GLOSSARY)


def test_parse_verdict_normalises_and_bounds_quotes() -> None:
    text = '<think>{x}</think>{"verdict": "YES", "confidence": "weird", "quotes": ["a", "b", "c", "d"], "reason": "r"}'
    verdict = parse_trope_verdict(text)
    assert verdict.verdict == "yes" and verdict.confidence == "low" and len(verdict.quotes) == 3
    with pytest.raises(ValueError):
        parse_trope_verdict('{"verdict": "maybe"}')


def book_text() -> str:
    return "".join(f"第{i:04d}章 标题{i}\n" + f"第{i}章的正文内容。" * 80 + "\n" for i in range(1, 60))


def test_trope_evidence_differs_from_judge_evidence() -> None:
    text = book_text()
    chapters = split_chapters(text)
    judge = judge_chapter_indices("n0", chapters, windows=6)
    tool = judge_chapter_indices("n0", chapters, windows=6, seed_salt="trope:")
    assert judge != tool
    assert sample_judge_evidence(text, "n0", seed_salt="trope:") != sample_judge_evidence(text, "n0")


class FakeTransport:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def complete(self, prompt: str, max_tokens: int) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_cached_judge_samples_asks_and_caches(tmp_path: Path) -> None:
    transport = FakeTransport(['{"verdict": "yes", "confidence": "high", "quotes": ["引文"], "reason": "r"}'])
    judge = CachedTropeJudge(transport, lambda novel_id: book_text() if novel_id == "n0" else None, "m", cache_path=tmp_path / "c.jsonl")

    first = judge.judge("n0", "后宫")
    assert first["verdict"] == "yes" and first["cached"] is False
    assert "后宫" in transport.prompts[0] and TROPE_GLOSSARY["后宫"] in transport.prompts[0]
    assert "第" in transport.prompts[0]

    second = judge.judge("n0", "后宫")
    assert second["cached"] is True and not transport.responses

    reloaded = CachedTropeJudge(FakeTransport([]), lambda _: None, "m", cache_path=tmp_path / "c.jsonl")
    assert reloaded.judge("n0", "后宫")["verdict"] == "yes"

    assert judge.judge("n0", "不存在的标签")["verdict"] == "unknown_trope"
    with pytest.raises(ToolError):
        judge.judge("missing", "后宫")


def test_unparseable_verdict_is_unclear_and_not_cached(tmp_path: Path) -> None:
    transport = FakeTransport(["我不知道", '{"verdict": "no", "confidence": "medium", "quotes": []}'])
    judge = CachedTropeJudge(transport, lambda _: book_text(), "m", cache_path=tmp_path / "c.jsonl")
    assert judge.judge("n0", "爽文")["verdict"] == "unclear"
    assert not (tmp_path / "c.jsonl").exists()
    assert judge.judge("n0", "爽文")["verdict"] == "no"
    assert len((tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_prompt_forbids_prior_knowledge_and_ends_with_the_evidence() -> None:
    prompt = build_trope_prompt("后宫", "定义", "证据文本")
    assert "先验知识" in prompt
    assert prompt.endswith("【采样文本】\n证据文本")


def test_card_rules_only_name_vocabulary_labels() -> None:
    for trope, (scale, values) in CARD_STYLE_RULES.items():
        assert trope in TROPE_GLOSSARY and set(values) <= set(STYLE_OPTIONS[scale])
    for scale, yes, no in CARD_SCALE_EXTREMES.values():
        assert {yes, no} <= set(STYLE_OPTIONS[scale])
    assert all(set(v) <= set(GENRES) for v in CARD_GENRE_RULES.values())
    assert all(set(v) <= set(ELEMENTS) for v in CARD_ELEMENT_RULES.values())
    assert all(set(v) <= set(SUBGENRE_TO_GENRE) for v in CARD_SUBGENRE_RULES.values())


def test_card_verdict_answers_from_scales_genre_and_elements_and_stays_silent_otherwise() -> None:
    card = BookCard("n", genre="玄幻", subgenre="东方玄幻", elements=["系统", "升级流"], style={"感情线": "多女主", "基调": "热血", "爽度": "中", "主角起点": ""})
    assert card_verdict(card, "后宫")["verdict"] == "yes" and card_verdict(card, "后宫")["source"] == "card"
    assert card_verdict(card, "压抑") == {"verdict": "no", "quotes": [], "confidence": "medium", "reason": "书卡：基调 热血", "source": "card"}
    assert card_verdict(card, "金手指")["verdict"] == "yes" and "系统" in card_verdict(card, "金手指")["reason"]
    assert card_verdict(card, "玄幻")["verdict"] == "yes"
    assert card_verdict(card, "言情")["verdict"] == "no"  # genre 玄幻 and 感情线 多女主 -> not a romance
    assert card_verdict(card, "爽文") is None  # 爽度 中 is the wide middle
    assert card_verdict(card, "开局无敌") is None  # scale not filled
    assert card_verdict(card, "灵异") is None  # a missing element is not evidence
    assert card_verdict(card, "种马") is None and card_verdict(None, "后宫") is None
    romance = BookCard("r", genre="言情", subgenre="现代言情", style={"感情线": "单女主主线"})
    assert card_verdict(romance, "言情")["verdict"] == "yes"
    assert card_verdict(BookCard("g", genre="悬疑灵异", subgenre="灵异民俗"), "灵异")["verdict"] == "yes"


def test_cached_judge_prefers_the_card_and_reads_text_only_when_it_is_silent(tmp_path: Path) -> None:
    transport = FakeTransport(['{"verdict": "no", "confidence": "high", "quotes": [], "reason": "没有"}'])
    cards = {"n0": BookCard("n0", genre="玄幻", subgenre="东方玄幻", style={"感情线": "无"})}
    judge = CachedTropeJudge(transport, lambda novel_id: book_text() if novel_id == "n0" else None, "m", cache_path=tmp_path / "c.jsonl", cards=cards)
    from_card = judge.judge("n0", "后宫")
    assert from_card["verdict"] == "no" and from_card["source"] == "card" and transport.prompts == []
    from_text = judge.judge("n0", "种马")
    assert from_text["source"] == "text" and len(transport.prompts) == 1
