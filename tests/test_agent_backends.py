from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agent.backends import LazyRawText, ParquetProfiles, build_agent, load_density_table
from src.chat_transport import ChatResponse, TokenUsage
from src.retrieval.multivector import make_book_meta
from src.vector_index import build_faiss_index, make_id_map, save_faiss_index, save_id_map


def test_parquet_profiles_and_density_table(tmp_path: Path) -> None:
    pd.DataFrame([{"novel_id": "a", "title_guess": "书A", "profile_text": "正文"}]).to_parquet(tmp_path / "p.parquet")
    profiles = ParquetProfiles.load(tmp_path / "p.parquet")
    assert profiles.get("a") == {"title": "书A", "profile": "正文"} and profiles.get("zz") is None and len(profiles) == 1

    pd.DataFrame([{"novel_id": "a", "char_count": 1000, "系统": 12.5, "异能": 0.0}]).to_parquet(tmp_path / "d.parquet")
    assert load_density_table(tmp_path / "d.parquet") == {"a": {"系统": 12.5, "异能": 0.0}}


def test_lazy_raw_text_reads_from_the_inventory_and_caches(tmp_path: Path) -> None:
    book = tmp_path / "book.txt"
    book.write_text("第一章\n正文内容。", encoding="utf-8")
    pd.DataFrame(
        [
            {"novel_id": "a", "absolute_path": str(book), "detected_encoding": "utf-8", "read_status": "ok"},
            {"novel_id": "bad", "absolute_path": str(tmp_path / "missing.txt"), "detected_encoding": "utf-8", "read_status": "ok"},
            {"novel_id": "failed", "absolute_path": "", "detected_encoding": None, "read_status": "failed"},
        ]
    ).to_parquet(tmp_path / "inv.parquet")  # no decode_replacement_chars column, like the PC's May inventory
    lookup = LazyRawText(tmp_path / "inv.parquet", cache_size=1)
    assert lookup("a").startswith("第一章")
    assert "a" in lookup._cache
    assert lookup("bad") is None and lookup("failed") is None and lookup("nope") is None


class FakeEmbedder:
    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, len(t) % 4] = 1.0
        return out


def test_build_agent_assembles_tools_from_artifacts(tmp_path: Path, monkeypatch: Any) -> None:
    df = pd.DataFrame([{"novel_id": "a", "title_guess": "书A", "profile_text": "仙侠正文"}, {"novel_id": "b", "title_guess": "书B", "profile_text": "都市"}])
    df.to_parquet(tmp_path / "profiles.parquet")
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    save_faiss_index(build_faiss_index(FakeEmbedder().encode(df["profile_text"].tolist())), index_dir / "faiss.index")
    save_id_map(make_id_map(df), index_dir / "novel_id_map.json")
    (index_dir / "index_metadata.json").write_text('{"model_name": "fake", "dtype": "fp32"}', encoding="utf-8")
    pd.DataFrame([{"novel_id": "a", "char_count": 10, "系统": 0.0}]).to_parquet(tmp_path / "density.parquet")
    pd.DataFrame([{"novel_id": "a", "absolute_path": "", "detected_encoding": "utf-8", "read_status": "failed"}]).to_parquet(tmp_path / "inv.parquet")

    bundle = build_agent(
        model="m",
        base_url="http://127.0.0.1:1/v1",
        index_dir=index_dir,
        profiles_path=tmp_path / "profiles.parquet",
        density_path=tmp_path / "density.parquet",
        inventory_path=tmp_path / "inv.parquet",
        memory_path=tmp_path / "memory" / "user.json",
        embedder_factory=lambda name, device=None, dtype="fp32": FakeEmbedder(),
    )
    assert sorted(bundle.tools.specs) == ["check_term", "check_trope", "get_profile", "memory_read", "memory_write", "search_books"]
    assert bundle.info["searcher"] == "idx/dense_single" and bundle.info["profiles"] == 2
    rows = bundle.tools.call("search_books", {"query": "都市", "k": 2})
    assert {r["novel_id"] for r in rows} == {"a", "b"}
    assert bundle.tools.call("check_term", {"novel_id": "a", "terms": ["系统"]})["violates"] is False

    # One scripted turn through the real loop: the memory tool writes and the file is saved after the turn.
    def fake_chat(messages: list[dict], tools: list[dict] | None, max_tokens: int) -> ChatResponse:
        return ChatResponse(content="好的", tool_calls=(), usage=TokenUsage())

    monkeypatch.setattr(bundle.loop.model, "chat", fake_chat)
    bundle.memory.write("negative", "系统", True)
    run = bundle.chat("随便聊聊")
    assert run.final_answer == "好的" and bundle.turn == 1
    assert (tmp_path / "memory" / "user.json").exists()
