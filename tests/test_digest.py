import json
from pathlib import Path

import pandas as pd

from src.digest import build_digest_worker, digest_chapter_indices, ending_status, load_sections, make_digest, sample_titles
from src.evidence import judge_chapter_indices
from src.retrieval.multivector import build_section_table
from src.split_chapters import split_chapters


def book(chapters: int = 40, extra_tail: str = "") -> str:
    text = "内容简介：\n一个普通少年的修仙故事。\n\n"
    for i in range(1, chapters + 1):
        text += f"第{i:03d}章 标题{i}\n" + f"第{i}章正文内容，句子一。句子二！" * 60 + "\n"
    return text + extra_tail


def test_digest_reads_opening_titles_middle_and_ending() -> None:
    text = book(40, "\n全书完\n")
    digest = make_digest("n0", "《书》", "某人", text)
    kinds = [s.kind for s in digest.sections]
    assert kinds == ["blurb"] + ["opening"] * 4 + ["titles"] + ["middle"] * 6 + ["ending"] * 2
    assert digest.ending_status == "完结标记" and "结局：完结标记" in digest.sections[0].text
    assert "内容简介：\n一个普通少年的修仙故事。" in digest.sections[0].text
    assert digest.sections[1].text.startswith("第001章") and digest.sections[-1].text.startswith("第040章")
    # digest_v2 reads chapters whole: each test chapter is about 1,000 characters and is not trimmed.
    assert all(950 <= len(s.text) <= 1100 for s in digest.sections if s.kind in ("opening", "middle", "ending"))
    assert digest.used_chapter_indices == sorted(set(digest.used_chapter_indices)) and len(digest.used_chapter_indices) == 12
    middle_titles = [s.text.split("\n", 1)[0] for s in digest.sections if s.kind == "middle"]
    assert middle_titles == ["第010章 标题10", "第015章 标题15", "第020章 标题20", "第025章 标题25", "第030章 标题30", "第035章 标题35"]  # 15%..90% of the 34 inner chapters
    assert 12000 < len(digest.text()) < 15000


def test_non_narrative_tail_chapters_are_skipped_for_the_ending() -> None:
    text = book(20) + "第021章 番外 甜甜的日常\n" + "番外正文。" * 100 + "\n第022章 完本感言\n" + "感谢大家。" * 100 + "\n"
    chapters = split_chapters(text)
    roles = digest_chapter_indices(chapters)
    assert [chapters[i].title for i in roles["ending"]] == ["第019章 标题19", "第020章 标题20"]
    assert ending_status("……大结局。") == "完结标记" and ending_status("未完待续") == "无标记"


def test_titles_are_sampled_evenly_and_capped() -> None:
    chapters = split_chapters(book(300))
    titles = sample_titles(chapters, list(range(len(chapters))), samples=10)
    assert len(titles) == 10 and titles[0] == "第001章 标题1" and titles[-1].startswith("第27")


def test_short_books_fall_back_to_character_windows() -> None:
    digest = make_digest("n1", "《短》", None, "没有章节结构的一段很长的文字。" * 400)
    assert [s.kind for s in digest.sections] == ["blurb", "opening"] + ["middle"] * 6 + ["ending"]
    assert digest.to_row()["digest_version"] == "digest_v2"
    assert digest.used_chapter_indices == []


def test_judge_evidence_avoids_digest_chapters() -> None:
    chapters = split_chapters(book(40))
    roles = digest_chapter_indices(chapters)
    used = set(roles["opening"]) | set(roles["middle"]) | set(roles["ending"])
    picked = judge_chapter_indices("n0", chapters, windows=6)
    assert picked and not (set(picked) & used)


def test_worker_and_section_table_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "b.txt"
    path.write_text(book(12), encoding="utf-8")
    row = {"novel_id": "n0", "absolute_path": str(path), "detected_encoding": "utf-8", "read_status": "ok", "title_guess": "《书》", "author_guess": float("nan")}
    result = build_digest_worker(row)
    assert result.reason == "ok" and result.row["author_guess"] is None
    frame = pd.DataFrame([result.row])
    sections = load_sections(frame)["n0"]
    assert sections[0].kind == "blurb"
    texts, records = build_section_table(frame)
    assert [r.kind for r in records] == [s.kind for s in sections] and texts[0] == sections[0].text
    assert build_digest_worker({**row, "read_status": "failed"}).reason == "failed"
    assert build_digest_worker({**row, "absolute_path": str(tmp_path / "nope")}).reason == "missing"
