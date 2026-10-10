import json

import pandas as pd

from src.retrieval.multivector import DEFAULT_CHUNK_CHARS, Section, build_section_table, chunk_section, chunk_text, split_long_paragraph


def chapter(paragraphs: int, para_chars: int = 100, title: str = "第三章 夜行") -> str:
    return title + "\n" + "\n".join(("第%d段。" % i).ljust(para_chars, "字") for i in range(paragraphs))


def test_short_text_is_one_piece_and_empty_text_none() -> None:
    assert chunk_text("短文本", 1500) == ["短文本"]
    assert chunk_text("   ", 1500) == []
    assert chunk_section(Section("blurb", ""), 1500) == []


def test_chunks_respect_the_budget_and_paragraph_boundaries() -> None:
    text = chapter(40)
    pieces = chunk_text(text, 1500)
    assert all(len(p) <= 1500 for p in pieces)
    assert "\n".join(pieces) == text  # nothing lost or reordered, cuts only at newlines
    # roughly balanced: no tiny tail piece
    assert min(len(p) for p in pieces) > 0.5 * max(len(p) for p in pieces)


def test_over_long_paragraph_is_cut_at_sentence_ends() -> None:
    paragraph = "".join("第%d句话说完了。" % i for i in range(300))
    pieces = split_long_paragraph(paragraph, 500)
    assert all(len(p) <= 500 for p in pieces)
    assert all(p.endswith("。") for p in pieces)
    assert "".join(pieces) == paragraph
    assert split_long_paragraph("字" * 1200, 500) == ["字" * 500, "字" * 500, "字" * 200]


def test_chapter_heading_repeats_on_every_piece_and_kind_is_kept() -> None:
    pieces = chunk_section(Section("middle", chapter(40)), 1500)
    assert len(pieces) >= 3
    assert all(p.kind == "middle" for p in pieces)
    assert all(p.text.startswith("第三章 夜行\n") for p in pieces)
    assert all(len(p.text) <= 1500 for p in pieces)


def test_title_list_splits_between_titles() -> None:
    titles = "章节目录（抽样）：\n" + " / ".join("第%d章 名字名字名字" % i for i in range(200))
    pieces = chunk_section(Section("titles", titles), 1500)
    assert len(pieces) == 2
    assert all(p.text.startswith("章节目录（抽样）：\n") for p in pieces)
    assert all("第%d章" % i in pieces[0].text + pieces[1].text for i in range(200))
    assert all("名字名字名字" in part for p in pieces for part in p.text.split("\n")[1].split(" / ") if part.strip())


def test_build_section_table_chunks_only_long_sections() -> None:
    sections = [
        {"kind": "blurb", "text": "标题：书\n内容简介：\n短简介"},
        {"kind": "opening", "text": chapter(40)},
        {"kind": "titles", "text": "章节目录（抽样）：\n第一章 / 第二章"},
    ]
    frame = pd.DataFrame([{"novel_id": "n1", "title_guess": "书", "profile_text": "x", "sections_json": json.dumps(sections, ensure_ascii=False)}])
    texts, records = build_section_table(frame, {"n1": "题材：玄幻"}, chunk_chars=1500)
    kinds = [r.kind for r in records]
    assert kinds[0] == "blurb" and kinds[-2] == "titles" and kinds[-1] == "card"
    assert kinds.count("opening") >= 3
    assert [r.ordinal for r in records] == list(range(len(records)))
    assert all(len(t) <= 1500 for t in texts)
    whole, _ = build_section_table(frame, {"n1": "题材：玄幻"}, chunk_chars=None)
    assert len(whole) == 4
    assert DEFAULT_CHUNK_CHARS == 1500
