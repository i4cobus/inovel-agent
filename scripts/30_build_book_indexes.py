"""Build book-level indexes into one directory: dense (single or multi-vector) and/or BM25.

Each run names a configuration (``--out-dir data/index/<name>``) so several can
coexist and go through ``31_retrieval_bench.py`` side by side. Windows-safe:
``uv run python scripts/30_build_book_indexes.py ...``.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from src.config import INDEX_DIR
from src.embed import DEFAULT_BATCH_SIZE, DEFAULT_EMBEDDING_MODEL, encode_documents, load_embedding_model
from src.retrieval.bm25 import BM25Index
from src.retrieval.multivector import MultiVectorIndex, build_section_table, make_book_meta
from src.vector_index import (
    DEFAULT_PROFILES_PATH,
    build_faiss_index,
    load_profiles_for_index,
    make_id_map,
    make_index_metadata,
    save_faiss_index,
    save_id_map,
    save_index_metadata,
)

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    out_dir: Path = typer.Option(INDEX_DIR / "multi_0p6b", help="Directory for this configuration's artifacts."),
    profiles: Path = typer.Option(DEFAULT_PROFILES_PATH, help="Novel profiles parquet."),
    dense: str = typer.Option("multi", help="single | multi | none"),
    bm25: bool = typer.Option(True, "--bm25/--no-bm25", help="Also build a BM25 index over the full profile text."),
    model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer model for the dense index."),
    device: str | None = typer.Option(None, help="torch device, e.g. cuda:0 or cpu."),
    dtype: str = typer.Option("fp32", help="fp32 | bf16 | fp16; bf16 for the 4B model on a 16 GB card."),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE),
    limit: int | None = typer.Option(None, help="First N profiles only (smoke run)."),
    overwrite: bool = typer.Option(False),
) -> None:
    if dense not in ("single", "multi", "none"):
        raise typer.BadParameter("--dense must be single, multi or none")
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise typer.BadParameter(f"{out_dir} is not empty; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_profiles_for_index(profiles_path=profiles, limit=limit)
    frame = loaded.dataframe
    if frame.empty:
        raise typer.BadParameter("No valid profiles.")
    console.print(f"Profiles: {len(frame)} (skipped {loaded.skipped_rows})")
    summary: dict[str, object] = {"profiles": len(frame), "created_at": datetime.now(timezone.utc).isoformat()}

    if bm25:
        started = time.perf_counter()
        index = BM25Index.build(frame["profile_text"].tolist(), frame["novel_id"].tolist())
        index.save(out_dir / "bm25.json")
        (out_dir / "book_meta.json").write_text(json.dumps(make_book_meta(frame), ensure_ascii=False), encoding="utf-8")
        summary["bm25"] = {"docs": index.size, "vocab": len(index.postings), "seconds": round(time.perf_counter() - started, 1)}
        console.print(f"BM25: {index.size} docs, vocab {len(index.postings)}, {summary['bm25']['seconds']}s")

    if dense != "none":
        embedder = load_embedding_model(model, device=device, dtype=dtype)
        started = time.perf_counter()
        if dense == "single":
            embeddings = encode_documents(embedder, frame["profile_text"].tolist(), batch_size=batch_size)
            save_faiss_index(build_faiss_index(embeddings), out_dir / "faiss.index")
            save_id_map(make_id_map(frame), out_dir / "novel_id_map.json")
            vectors = int(embeddings.shape[0])
        else:
            texts, records = build_section_table(frame)
            embeddings = encode_documents(embedder, texts, batch_size=batch_size)
            MultiVectorIndex.build(embeddings, records, make_book_meta(frame)).save(out_dir)
            vectors = len(records)
        metadata = make_index_metadata(
            model_name=model, embedding_dim=int(embeddings.shape[1]), num_vectors=vectors, normalize_embeddings=True, source_profiles=profiles
        )
        metadata["dense"] = dense
        metadata["dtype"] = dtype
        save_index_metadata(metadata, out_dir / "index_metadata.json")
        summary["dense"] = {"mode": dense, "model": model, "vectors": vectors, "dim": int(embeddings.shape[1]), "seconds": round(time.perf_counter() - started, 1)}
        console.print(f"Dense ({dense}): {vectors} vectors × {embeddings.shape[1]}, {summary['dense']['seconds']}s")

    (out_dir / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"Wrote {out_dir}")


if __name__ == "__main__":
    app()
