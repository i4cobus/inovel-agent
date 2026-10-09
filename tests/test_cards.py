import json
from pathlib import Path

import pandas as pd

from src.retrieval.card_schema import ELEMENTS, GENRES, STYLE_OPTIONS, SUBGENRES, SUBGENRE_TO_GENRE, genre_of, vocabulary_text
from src.retrieval.cards import BookCard, CardBuilder, build_card_prompt, cards_to_frame, load_cards, normalise_subgenre, parse_card, quote_supported, vocabulary_report
from src.retrieval.multivector import build_section_table

RESPONSE = json.dumps(
    {
        "subgenre": "幻想修仙",
        "subgenre_evidence": "踏上修仙之路",
        "elements": {"凡人流": "资质平平的少年", "修真": "踏上修仙之路", "炼丹炼器": "炼制丹药", "不存在的标签": "x", "系统": "叮，系统绑定成功"},
        "style": {"爽度": "低", "基调": "沉重压抑", "感情线比重": "辅线", "主角起点": "普通", "节奏": "飞快"},
        "protagonist": "资质普通的少年",
        "setting": "修仙界",
        "tone": "沉稳",
        "one_liner": "普通少年一步步修仙。",
        "keywords": ["凡人流", "修仙", "宗门", "炼气", "a", "b", "c", "d", "e"],
    },
    ensure_ascii=False,
)


def test_schema_is_consistent() -> None:
    subs = [s for v in SUBGENRES.values() for s in v]
    assert len(subs) == len(set(subs)) and len(GENRES) == 12 and "现实" not in GENRES
    assert all(genre_of(s) in GENRES for s in subs) and genre_of("瞎编") == "其他"
    assert 40 <= len(ELEMENTS) <= 50 and set(STYLE_OPTIONS) == {"爽度", "基调", "感情线比重", "主角起点", "节奏"}
    text = vocabulary_text()
    assert "高武世界（玄幻）" in text and ELEMENTS["后宫"] in text and "开局无敌 / 普通 / 废柴逆袭" in text
    assert normalise_subgenre("修真文明") == ("修真文明", "仙侠") and normalise_subgenre("仙侠：修真文明") == ("修真文明", "仙侠")
    assert normalise_subgenre("修真文明（仙侠）") == ("修真文明", "仙侠") and normalise_subgenre("都市") == ("", "都市")
    assert normalise_subgenre("瞎编") == ("", "其他") and normalise_subgenre("") == ("", "其他")


SOURCE = "一个资质平平的少年，踏上修仙之路。他在山洞里炼制丹药，日复一日。"


def test_quote_supported_ignores_punctuation_but_not_paraphrase() -> None:
    assert quote_supported("资质平平的少年", SOURCE) and quote_supported("踏上修仙之路。", SOURCE)
    assert quote_supported("少年，踏上修仙之路，他在山洞", SOURCE)  # a 6-char window matches
    assert not quote_supported("系统绑定", SOURCE) and not quote_supported("少年", SOURCE) and not quote_supported("", SOURCE)


def test_parse_card_derives_genre_filters_vocabulary_and_records_drops() -> None:
    card = parse_card("n1", "<think>x</think>" + RESPONSE, model="m", source_text=SOURCE)
    assert card.genre == "仙侠" and card.subgenre == "幻想修仙"
    assert card.elements == ["凡人流", "修仙", "炼丹炼器"]  # 修真 -> 修仙 by alias; 系统's quote is not in the text
    assert card.evidence == {"凡人流": "资质平平的少年", "修仙": "踏上修仙之路", "炼丹炼器": "炼制丹药", "subgenre": "踏上修仙之路"}
    assert card.style == {"爽度": "低", "基调": "沉重压抑", "感情线比重": "辅线", "主角起点": "普通", "节奏": ""}
    assert card.keywords == ["宗门", "炼气", "a", "b", "c"]  # 凡人流 / 修仙 repeat elements
    assert card.dropped == ["element:不存在的标签", "element_unsupported:系统", "style:节奏=飞快"]
    text = card.text()
    assert text.startswith("题材：仙侠·幻想修仙") and "元素：凡人流、修仙、炼丹炼器" in text and "爽度 低" in text and "一句话：普通少年一步步修仙。" in text

    odd = parse_card("n2", '{"subgenre": "奇怪", "elements": "系统", "style": "none"}')
    assert odd.genre == "其他" and odd.subgenre == "" and odd.elements == ["系统"] and all(v == "" for v in odd.style.values())
    assert odd.dropped == ["subgenre:奇怪"]
    bare = parse_card("n3", '{"subgenre": "历史", "elements": ["空间", "星际文明", "刑侦"]}')
    assert bare.genre == "历史" and bare.subgenre == "" and bare.elements == ["随身空间", "机甲星际", "刑侦推理"] and bare.dropped == ["subgenre_missing:历史"]


def test_prompt_carries_the_vocabulary_and_caps_the_digest() -> None:
    prompt = build_card_prompt("档案" * 5000, max_chars=100)
    assert SUBGENRE_TO_GENRE and "幻想修仙" in prompt and ELEMENTS["系统"] in prompt and '"subgenre_evidence"' in prompt and "逐字抄自档案" in prompt
    assert prompt.endswith("档案" * 50)


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
    assert cards[1].error.startswith("parse") and cards[1].genre == "其他" and transport.calls == 2

    again = CardBuilder(FakeTransport([RESPONSE]), "m", cache_path=tmp_path / "c.jsonl")
    cards = again.build_many([("a", "档案A"), ("b", "档案B")], workers=2)
    assert {c.novel_id: c.error for c in cards} == {"a": "", "b": ""}  # a from cache, b rebuilt


def test_frame_round_trip_card_section_and_report(tmp_path: Path) -> None:
    card = parse_card("a", RESPONSE, model="m")
    frame = cards_to_frame([card])
    frame.to_parquet(tmp_path / "cards.parquet", index=False)
    loaded = load_cards(tmp_path / "cards.parquet")
    assert loaded["a"].to_dict() == card.to_dict()

    profiles = pd.DataFrame([{"novel_id": "a", "title_guess": "《书》", "profile_text": "标题：《书》\n\n内容简介：\n简介"}, {"novel_id": "b", "title_guess": "《乙》", "profile_text": "简介乙"}])
    texts, records = build_section_table(profiles, {"a": card.text()})
    assert [(r.novel_id, r.kind) for r in records] == [("a", "blurb"), ("a", "card"), ("b", "blurb")]
    assert texts[1].startswith("标题：《书》\n题材：仙侠·幻想修仙")

    report = vocabulary_report(loaded)
    assert report["genres"] == {"仙侠": 1} and report["elements"]["修仙"] == 1 and report["dropped"] == {"element:不存在的标签": 1, "style:节奏=飞快": 1}
    assert report["unsupported_elements"] == {}
    assert report["style"]["节奏"] == {"（空）": 1} and report["elements_per_card"] == 4.0  # no source_text here, so 系统 is kept
