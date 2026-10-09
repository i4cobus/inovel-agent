import json
from pathlib import Path

import pandas as pd

from src.retrieval.card_schema import ELEMENTS, GENRES, STYLE_OPTIONS, SUBGENRES, SUBGENRE_TO_GENRE, genre_of, vocabulary_text
from src.retrieval.cards import card_cache_key, BookCard, CardBuilder, build_card_prompt, cards_to_frame, load_cards, normalise_subgenre, parse_card, quote_supported, vocabulary_report
from src.retrieval.multivector import build_section_table

RESPONSE = json.dumps(
    {
        "subgenre": "幻想修仙",
        "subgenre_evidence": "踏上修仙之路",
        "elements": {"凡人流": "资质平平的少年", "修真": "踏上修仙之路", "炼丹炼器": "炼制丹药", "不存在的标签": "x", "系统": "叮，系统绑定成功"},
        "style": {"爽度": "低", "基调": "沉重压抑", "感情线比重": "辅线", "主角起点": "超强", "节奏": "快"},
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
    assert 40 <= len(ELEMENTS) <= 50 and set(STYLE_OPTIONS) == {"主角结构", "爽度", "基调", "感情线比重", "主角起点"}
    text = vocabulary_text()
    assert "玄幻：东方玄幻、异世大陆、高武世界" in text and ELEMENTS["后宫"] in text and "开局无敌 / 普通 / 废柴逆袭" in text
    assert normalise_subgenre("幻想修仙") == ("幻想修仙", "仙侠") and normalise_subgenre("仙侠：幻想修仙") == ("幻想修仙", "仙侠")
    assert normalise_subgenre("幻想修仙（仙侠）") == ("幻想修仙", "仙侠") and normalise_subgenre("都市") == ("都市生活", "都市") and normalise_subgenre("历史") == ("", "历史")
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
    assert card.style == {"主角结构": "", "爽度": "低", "基调": "沉重压抑", "感情线比重": "辅线", "主角起点": ""}  # 节奏 is no longer a scale and is ignored
    assert card.keywords == ["宗门", "炼气", "a", "b", "c"]  # 凡人流 / 修仙 repeat elements
    assert card.dropped == ["element:不存在的标签", "element_unsupported:系统=叮，系统绑定成功", "style:主角起点=超强"]
    text = card.text()
    assert text.startswith("题材：仙侠·幻想修仙") and "元素：凡人流、修仙、炼丹炼器" in text and "爽度 低" in text and "一句话：普通少年一步步修仙。" in text

    odd = parse_card("n2", '{"subgenre": "奇怪", "elements": "系统", "style": "none"}')
    assert odd.genre == "其他" and odd.subgenre == "" and odd.elements == ["系统"] and all(v == "" for v in odd.style.values())
    assert odd.dropped == ["subgenre:奇怪"]
    bare = parse_card("n3", '{"subgenre": "历史", "elements": ["空间", "星际文明", "刑侦"]}')
    assert bare.genre == "历史" and bare.subgenre == "" and bare.elements == ["随身空间", "机甲星际", "刑侦推理"] and bare.dropped == ["subgenre_missing:历史"]


def test_parse_card_flattens_elements_nested_by_group_and_routes_style_home() -> None:
    # What qwen3.5:9b did on 377 of 437 cards in the first v2.2 run: nested by group name, style scales inside.
    nested = json.dumps(
        {
            "subgenre": "幻想修仙",
            "elements": {
                "主角来路": {},
                "流派": {"凡人流": "资质平平的少年"},
                "世界设定": {"修真": "踏上修仙之路", "末世": "x"},
                "感情": ["多女主"],
                "爽度": "低",
            },
            "style": {"基调": "沉重压抑"},
        },
        ensure_ascii=False,
    )
    card = parse_card("n4", nested, source_text=SOURCE)
    assert card.elements == ["凡人流", "修仙"]
    assert card.dropped == ["element_unsupported:末世=x", "element_unsupported:后宫="]
    assert card.style["爽度"] == "低" and card.style["基调"] == "沉重压抑"

    aliased = parse_card("n6", json.dumps({"subgenre": "修真文明", "elements": {"修真": "踏上修仙之路"}}, ensure_ascii=False), source_text=SOURCE)
    assert aliased.subgenre == "幻想修仙" and aliased.elements == ["修仙"] and aliased.dropped == []
    with_gate = {"系统": ("面板",)}
    import src.retrieval.cards as cards_module
    cards_module.ELEMENT_QUOTE_SIGNATURES.update(with_gate)
    try:
        weak = parse_card("n7", json.dumps({"subgenre": "幻想修仙", "elements": {"系统": "这扇门的名字叫做次元之门"}}, ensure_ascii=False), source_text=SOURCE + "这扇门的名字叫做次元之门")
        assert weak.elements == [] and weak.dropped == ["element_weak_quote:系统=这扇门的名字叫做次元之门"]
    finally:
        cards_module.ELEMENT_QUOTE_SIGNATURES.clear()
    assert normalise_subgenre("军事：军旅生涯") == ("军旅特战", "军事") and normalise_subgenre("超级科技") == ("未来科技", "科幻")

    copied = parse_card("n5", json.dumps({"subgenre": "幻想修仙", "elements": {"权谋": "朝堂、家族或势力之间的谋略博弈是主要看点", "修真": "踏上修仙之路"}}, ensure_ascii=False), source_text=SOURCE)
    assert copied.elements == ["修仙"] and copied.dropped == ["element_definition:权谋=朝堂、家族或势力之间的谋略博弈是主要看点"]
    assert copied.elements_unverified == ["权谋"] and "疑似元素：权谋" in copied.text()

    harem_src = SOURCE + "苏雨妍笑了。森下丽香也来了。"
    harem = parse_card("n8", json.dumps({"subgenre": "现代言情", "elements": {"后宫": "苏雨妍、森下丽香、不存在的人"}}, ensure_ascii=False), source_text=harem_src)
    assert harem.elements == ["后宫"] and harem.evidence["后宫"] == "苏雨妍、森下丽香" and harem.dropped == []
    lone = parse_card("n9", json.dumps({"elements": {"后宫": "苏雨妍"}}, ensure_ascii=False), source_text=harem_src)
    assert lone.elements == [] and lone.dropped == ["element_unsupported:后宫=苏雨妍"]
    assert BookCard.from_dict(copied.to_dict()).elements_unverified == ["权谋"]


def test_prompt_carries_the_vocabulary_and_caps_the_digest() -> None:
    prompt = build_card_prompt("档案" * 5000, max_chars=100)
    assert SUBGENRE_TO_GENRE and "幻想修仙" in prompt and ELEMENTS["系统"] in prompt and '"subgenre_evidence"' in prompt and "逐字抄自档案" in prompt
    assert "【档案】\n" + "档案" * 50 + "\n" in prompt and prompt.index("【档案】") < prompt.index("【题材二级") < prompt.index("填写规则")  # digest first, rules last
    assert "档案" * 51 not in prompt and prompt.rstrip().endswith("}")


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
    assert [f["novel_id"] for f in builder.failures] == ["b"] and builder.failures[0]["response_tail"] == "不是 JSON"

    again = CardBuilder(FakeTransport([RESPONSE]), "m", cache_path=tmp_path / "c.jsonl")
    cards = again.build_many([("a", "档案A"), ("b", "档案B")], workers=2)
    assert {c.novel_id: c.error for c in cards} == {"a": "", "b": ""}  # a from cache, b rebuilt

    # The raw response is cached, so a parser change replays without the model.
    assert again.raw[card_cache_key("b", "m")] == RESPONSE
    assert again.reparse([("a", "档案A"), ("b", SOURCE), ("zzz", "没建过")]) == 2
    third = CardBuilder(FakeTransport([]), "m", cache_path=tmp_path / "c.jsonl")
    assert third.build_one("b", SOURCE).elements == ["凡人流", "修仙", "炼丹炼器"]  # quotes now checked against SOURCE
    assert third.build_one("a", "档案A").elements == []  # nothing in 档案A backs a quote


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
    assert report["genres"] == {"仙侠": 1} and report["elements"]["修仙"] == 1 and report["dropped"] == {"element:不存在的标签": 1, "style:主角起点=超强": 1}
    assert report["unsupported_elements"] == {}
    assert report["style"]["主角起点"] == {"（空）": 1} and report["style"]["主角结构"] == {"（空）": 1} and report["elements_per_card"] == 4.0  # no source_text here, so 系统 is kept
