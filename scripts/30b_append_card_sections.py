"""Append the book-card sections to a multi-vector index that was built without them.

The 4B chunk build takes ~8.5 h and the cards were still being generated, so the index was built from
the digest alone and the ~7,655 card vectors are added here (~10 min on the 4080). The embedding model,
dtype, max_seq_length and chunk_chars are read from the index's own metadata so the card vectors come
from the same model in the same configuration; card ordinals continue each book's section count exactly
as a joint build would have numbered them (see card_section_table).

    uv run python scripts/30b_append_card_sections.py --index-dir data/index/multi_4b --dry-run
    uv run python scripts/30b_append_card_sections.py --index-dir data/index/multi_4b
    uv run python scripts/30b_append_card_sections.py --index-dir data/index/multi_4b --replace   # cards regenerated

Afterwards derive the pooled single index again (scripts/33 --kind-mean): it reads sections.json.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from src.embed import encode_documents_with_backoff, load_embedding_model
from src.retrieval.cards import DEFAULT_CARDS_PATH, load_cards
from src.retrieval.multivector import INDEX_FILE, META_FILE, SECTIONS_FILE, MultiVectorIndex, card_section_table, section_stats

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    index_dir: Path = typer.Option(..., help="Multi-vector index directory (faiss.index + sections.json + index_metadata.json)."),
    cards: Path = typer.Option(DEFAULT_CARDS_PATH, help="book_cards parquet; books without a card are skipped."),
    batch_size: int | None = typer.Option(None, help="Default: the batch size recorded in index_metadata.json."),
    device: str | None = typer.Option(None, help="torch device, e.g. cuda:0 or cpu."),
    replace: bool = typer.Option(False, help="Drop the card sections already in the index first (cards were regenerated)."),
    dry_run: bool = typer.Option(False, help="Report what would be appended, load no model."),
) -> None:
    metadata_path = index_dir / META_FILE
    if not metadata_path.exists():
        raise typer.BadParameter(f"{metadata_path} missing: not a finished multi-vector index")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dense") != "multi":
        raise typer.BadParameter(f"{index_dir} is a {metadata.get('dense')!r} index; cards are appended to multi-vector indexes only")

    started = time.perf_counter()
    multi = MultiVectorIndex.load(index_dir)
    before = multi.kinds()
    console.print(f"Index: {multi.index.ntotal} vectors, kinds {before}")
    if before.get("card"):
        if not replace:
            raise typer.BadParameter(f"{index_dir} already holds {before['card']} card sections; pass --replace to rebuild them")
        multi = multi.without_kind("card")
        console.print(f"Dropped {before['card']} card sections -> {multi.index.ntotal} vectors")

    card_texts = {novel_id: card.text() for novel_id, card in load_cards(cards).items()}
    titles = {novel_id: meta.get("title_guess", "") for novel_id, meta in multi.meta.items()}
    chunk_chars = metadata.get("chunk_chars")
    texts, records = card_section_table(multi.records, card_texts, titles, chunk_chars=chunk_chars)
    books = {record.novel_id for record in records}
    missing = len(multi.meta) - len(books)
    chars = sorted(len(text) for text in texts)
    report = {
        "cards_path": cards.as_posix(),
        "books_with_card": len(books),
        "books_without_card": missing,
        "card_vectors": len(records),
        "card_chars_p50": chars[len(chars) // 2] if chars else 0,
        "card_chars_max": chars[-1] if chars else 0,
        "chunk_chars": chunk_chars,
    }
    console.print(f"Cards: {report}")
    if not records:
        raise typer.BadParameter("no book of this index has a card")
    if dry_run:
        return

    model_name = metadata["model_name"]
    dtype = metadata.get("dtype", "bf16")
    max_seq_length = metadata.get("max_seq_length")
    batch = batch_size or int(metadata.get("batch_size") or 8)
    console.print(f"Embedding {len(texts)} card sections with {model_name} ({dtype}, max_seq {max_seq_length}, batch {batch})")
    embedder = load_embedding_model(model_name, device=device, dtype=dtype, max_seq_length=max_seq_length)
    embeddings, used_batch = encode_documents_with_backoff(embedder, texts, batch_size=batch)
    multi.append(embeddings, records)

    # Write next to the live files and swap, so a crash mid-write cannot leave a half-written index behind.
    staging = index_dir / "_append_staging"
    multi.save(staging)
    for name in (INDEX_FILE, SECTIONS_FILE):
        os.replace(staging / name, index_dir / name)
    staging.rmdir()

    metadata["num_vectors"] = int(multi.index.ntotal)
    metadata["cards"] = {**report, "batch_size": used_batch, "appended_at": datetime.now(timezone.utc).isoformat()}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = index_dir / "build_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        sections = summary.get("sections", {})
        sections.update(section_stats(multi.records))
        summary["sections"] = sections
        summary["cards"] = metadata["cards"]
        if "dense" in summary:
            summary["dense"]["vectors"] = int(multi.index.ntotal)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"Appended {len(records)} card vectors -> {multi.index.ntotal} total, kinds {multi.kinds()}, {time.perf_counter() - started:.0f}s")
    console.print(f"Wrote {index_dir}")


if __name__ == "__main__":
    app()
