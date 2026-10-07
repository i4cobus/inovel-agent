from pathlib import Path

import pytest

from src.agent.tools import ToolError
from src.agent.trope import TROPE_GLOSSARY, CachedTropeJudge, build_trope_prompt, parse_trope_verdict
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
