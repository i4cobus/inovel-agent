"""Derive a single-vector book index from a multi-vector one by pooling section vectors (CPU only).

    uv run python scripts/33_pool_single_from_multi.py --multi-dir data/index/multi_0p6b \
        --out-dir data/index/single_0p6b --weight blurb=2 --weight titles=2 --overwrite

Then benchmark it like any other directory: scripts/31_retrieval_bench.py data/index/single_0p6b
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from src.config import INDEX_DIR
from src.retrieval.pooling import derive_single_index, parse_weights

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    multi_dir: Path = typer.Option(INDEX_DIR / "multi_0p6b", help="Directory holding faiss.index + sections.json"),
    out_dir: Path = typer.Option(INDEX_DIR / "single_0p6b"),
    weight: list[str] = typer.Option([], help="Section-kind weight, repeatable: --weight blurb=2 --weight titles=2"),
    copy_bm25: bool = typer.Option(True, help="Copy bm25.json and book_meta.json so the hybrid can be benchmarked too"),
    overwrite: bool = typer.Option(False),
) -> None:
    try:
        weights = parse_weights(weight)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    summary = derive_single_index(multi_dir, out_dir, weights=weights, copy_bm25=copy_bm25, overwrite=overwrite)
    console.print(summary)
    console.print(f"Wrote {out_dir}")


if __name__ == "__main__":
    app()
