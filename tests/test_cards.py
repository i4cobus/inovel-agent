import json
from pathlib import Path

import pandas as pd

from src.agent.trope import TROPE_GLOSSARY
from src.retrieval.cards import BookCard, CardBuilder, build_card_prompt, cards_to_frame, load_cards, parse_card, validate_tropes
from src.retrieval.multivector import build_section_table

RESPONSE = json.dumps(
    {
        "genre": "仙侠",
        "subgenres": ["凡人流"],
        "protagonist": "资质普通的少年",
        "setting": "修仙界",
        "pacing": "慢热",
        "tone": "沉稳",
        "one_liner": "普通少年一步步修仙。",
        "keywords": ["凡人流", "宗门", "炼气"],
        "tropes": {"后宫": "no", "爽文": "unclear", "金手指": "YES", "不存在": "yes"},
    },
    ensure_ascii=False,
)


def test_parse_card_normalises_fields_and_fills_every_trope() -> None:
    card = parse_card("n1", "<think>x</think>" + RESPONSE, model="m")
    assert card.genre == "仙侠" and card.pacing == "慢热" and card.keywords == ["凡人流", "宗门", "炼气"]
    assert set(card.tropes) == set(TROPE_GLOSSARY)
    assert card.tropes["金手指"] == "yes" and card.tropes["后宫"] == "no" and card.tropes["圣母"] == "unclear"
    assert card.yes_tropes == ["金手指"]
    text = card.text()
    assert text.startswith("题材：仙侠") and "标签：金手指" in text and "一句话：普通少年一步步修仙。" in text

    odd = parse_card("n2", '{"genre": "奇怪", "pacing": "飞快", "tropes": "none"}')
    assert odd.genre == "其他" and odd.pacing == "" and all(v == "unclear" for v in odd.tropes.values())


def test_prompt_carries_the_glossary_and_caps_the_profile() -> None:
    prompt = build_card_prompt("档案" * 5000, max_chars=100)
    assert TROPE_GLOSSARY["后宫"] in prompt and prompt.endswith("档案" * 50)


class FakeTransport:
    def __init__(self, responses: list[str]) -> None:
        self.responses, self.calls = responses, 0

    def complete(self, prompt: str, max_tokens: int) -> str:
        self.calls += 1
        return self.responses.pop(0)


def test_builder_caches_resumes_and_keeps_parse_failures_out_of_the_cache(tmp_path: Path) -> None:
    transport = FakeTransport([RESPONSE, "不是 JSON", RESPONSE])
    builder = CardBuilder(transport, "m", cache_path=tmp_path / "c.jsonl")
    cards = builder.build_many([("a", "档案A"), ("b", "档案B")], workers=1)
    assert cards[0].genre == "仙侠" and cards[0].error == ""
    assert cards[1].error.startswith("parse") and transport.calls == 2

    again = CardBuilder(FakeTransport([RESPONSE]), "m", cache_path=tmp_path / "c.jsonl")
    cards = again.build_many([("a", "档案A"), ("b", "档案B")], workers=2)
    assert {c.novel_id: c.error for c in cards} == {"a": "", "b": ""}  # a from cache, b rebuilt


def test_frame_round_trip_and_card_section(tmp_path: Path) -> None:
    card = parse_card("a", RESPONSE, model="m")
    frame = cards_to_frame([card])
    frame.to_parquet(tmp_path / "cards.parquet", index=False)
    loaded = load_cards(tmp_path / "cards.parquet")
    assert loaded["a"].to_dict() == card.to_dict()

    profiles = pd.DataFrame([{"novel_id": "a", "title_guess": "《书》", "profile_text": "标题：《书》\n\n内容简介：\n简介"}, {"novel_id": "b", "title_guess": "《乙》", "profile_text": "简介乙"}])
    texts, records = build_section_table(profiles, {"a": card.text()})
    assert [(r.novel_id, r.kind) for r in records] == [("a", "blurb"), ("a", "card"), ("b", "blurb")]
    assert texts[1].startswith("标题：《书》\n题材：仙侠")


def test_validate_tropes_reports_precision_recall_and_undecided() -> None:
    def card(novel_id: str, value: str) -> BookCard:
        return BookCard(novel_id, "玄幻", [], {"爽文": value}, "", "", "", "", "", [])

    cards = {"tp": card("tp", "yes"), "fp": card("fp", "yes"), "fn": card("fn", "no"), "tn": card("tn", "no"), "u": card("u", "unclear")}
    labels = {"爽文": {"tp": True, "fp": False, "fn": True, "tn": False, "u": True, "missing": True}}
    report = validate_tropes(cards, labels)["爽文"]
    assert report == {"n": 6, "decided": 4, "unclear": 1, "missing": 1, "precision": 0.5, "recall": 0.5, "accuracy": 0.5, "positives": 4}
