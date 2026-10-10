"""Appending card sections to a built multi-vector index must equal building with the cards present."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from src.retrieval.multivector import MultiVectorIndex, SectionRecord, build_section_table, card_section_table, make_book_meta

LONG_CARD = "题材：玄幻·东方玄幻\n" + "\n".join(f"第{i}段。" + "少年崛起，" * 40 for i in range(12))  # > 1500 chars: chunks


def profiles() -> pd.DataFrame:
    rows = []
    for novel_id, title in (("n1", "甲"), ("n2", "乙"), ("n3", "丙")):
        sections = [{"kind": "blurb", "text": f"{title}的简介"}, {"kind": "titles", "text": "第一章 开端\n第二章 继续"}, {"kind": "opening", "text": "第一章 开端\n" + "正文。" * 700}]
        rows.append({"novel_id": novel_id, "title_guess": title, "profile_text": "x", "sections_json": json.dumps(sections, ensure_ascii=False)})
    return pd.DataFrame(rows)


CARDS = {"n1": "题材：玄幻\n元素：系统", "n3": LONG_CARD}  # n2 has no card


def fake_embed(texts: list[str], dim: int = 16) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        out[row] = rng.normal(size=dim)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_card_sections_continue_each_books_ordinals_like_a_joint_build() -> None:
    frame = profiles()
    joint_texts, joint_records = build_section_table(frame, CARDS, chunk_chars=1500)
    base_texts, base_records = build_section_table(frame, None, chunk_chars=1500)
    titles = {r.novel_id: r.title_guess for r in frame.itertuples(index=False)}
    card_texts, card_records = card_section_table(base_records, CARDS, titles, chunk_chars=1500)
    assert [r.kind for r in card_records] == ["card"] * len(card_records)
    assert {r.novel_id for r in card_records} == {"n1", "n3"}
    assert sum(1 for r in card_records if r.novel_id == "n3") > 1  # the long card was chunked
    assert sorted(zip(base_records + card_records, base_texts + card_texts), key=repr) == sorted(zip(joint_records, joint_texts), key=repr)


def test_appended_index_answers_like_the_joint_one_and_without_kind_undoes_it() -> None:
    frame = profiles()
    meta = make_book_meta(frame)
    joint_texts, joint_records = build_section_table(frame, CARDS, chunk_chars=1500)
    joint = MultiVectorIndex.build(fake_embed(joint_texts), joint_records, meta)
    base_texts, base_records = build_section_table(frame, None, chunk_chars=1500)
    appended = MultiVectorIndex.build(fake_embed(base_texts), list(base_records), meta)
    titles = {n: m["title_guess"] for n, m in meta.items()}
    card_texts, card_records = card_section_table(appended.records, CARDS, titles, chunk_chars=1500)
    appended.append(fake_embed(card_texts), card_records)
    assert appended.index.ntotal == joint.index.ntotal
    assert appended.kinds() == joint.kinds()
    for query in ("系统流玄幻", "少年崛起", "乙的简介"):
        q = fake_embed([query])[0]
        strip = lambda rows: [(r["novel_id"], r["section_kind"], r["section_ordinal"], round(r["score"], 5)) for r in rows]
        assert strip(appended.search_vector(q, 3)) == strip(joint.search_vector(q, 3))
    stripped = appended.without_kind("card")
    assert stripped.kinds() == {k: v for k, v in joint.kinds().items() if k != "card"}
    assert stripped.records == base_records
    with pytest.raises(ValueError):
        appended.append(fake_embed(["x"]), [])


def test_script_dry_run_reports_cards_and_refuses_single_index(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("append_cards", Path("scripts/30b_append_card_sections.py"))
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)  # type: ignore[union-attr]

    frame = profiles()
    texts, records = build_section_table(frame, None, chunk_chars=1500)
    index_dir = tmp_path / "multi"
    MultiVectorIndex.build(fake_embed(texts), records, make_book_meta(frame)).save(index_dir)
    metadata = {"model_name": "m", "dense": "multi", "dtype": "bf16", "batch_size": 8, "max_seq_length": 1536, "chunk_chars": 1500, "num_vectors": len(records)}
    (index_dir / "index_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    cards = pd.DataFrame(
        [
            {"novel_id": "n1", "genre": "玄幻", "subgenre": "东方玄幻", "elements": json.dumps(["系统"]), "elements_unverified": "[]", "keywords": "[]", "dropped": "[]", "style": "{}", "protagonist": "", "setting": "", "tone": "", "one_liner": "少年崛起", "model": "m", "prompt_version": "v", "error": ""}
        ]
    )
    cards.to_parquet(tmp_path / "cards.parquet")
    result = CliRunner().invoke(script.app, ["--index-dir", str(index_dir), "--cards", str(tmp_path / "cards.parquet"), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "'books_with_card': 1" in result.output and "'books_without_card': 2" in result.output

    metadata["dense"] = "single"
    (index_dir / "index_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    result = CliRunner().invoke(script.app, ["--index-dir", str(index_dir), "--cards", str(tmp_path / "cards.parquet"), "--dry-run"])
    assert result.exit_code != 0
