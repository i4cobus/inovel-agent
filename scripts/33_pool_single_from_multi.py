"""Derive a single-vector book index from a multi-vector one by pooling section vectors (CPU only).

    uv run python scripts/33_pool_single_from_multi.py --overwrite            # DEFAULT_SECTION_WEIGHTS
    uv run python scripts/33_pool_single_from_multi.py --out-dir data/index/pool_mean --plain-mean --no-copy-bm25
    uv run python scripts/33_pool_single_from_multi.py --out-dir data/index/pool_x --weight blurb=2 --weight titles=2
    uv run python scripts/33_pool_single_from_multi.py --multi-dir data/index/multi_4b --out-dir data/index/single_4b --kind-mean

Then benchmark it like any other directory: scripts/31_retrieval_bench.py data/index/single_0p6b
(on macOS run with KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1: faiss and torch each ship an OpenMP).
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from src.config import INDEX_DIR
from src.retrieval.pooling import DEFAULT_KIND_WEIGHTS, DEFAULT_SECTION_WEIGHTS, derive_single_index, parse_weights

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    multi_dir: Path = typer.Option(INDEX_DIR / "multi_0p6b", help="Directory holding faiss.index + sections.json"),
    out_dir: Path = typer.Option(INDEX_DIR / "single_0p6b"),
    weight: list[str] = typer.Option([], help=f"Section-kind weight, repeatable: --weight blurb=2 --weight titles=2. Default: {DEFAULT_SECTION_WEIGHTS}"),
    plain_mean: bool = typer.Option(False, help="Ignore DEFAULT_SECTION_WEIGHTS and average every section equally"),
    kind_mean: bool = typer.Option(False, help=f"Average each kind first, then weight the kinds (for chunked indexes). Default weights then: {DEFAULT_KIND_WEIGHTS}"),
    copy_bm25: bool = typer.Option(True, help="Copy bm25.json and book_meta.json so the hybrid can be benchmarked too"),
    overwrite: bool = typer.Option(False),
) -> None:
    try:
        default = DEFAULT_KIND_WEIGHTS if kind_mean else DEFAULT_SECTION_WEIGHTS
        weights = parse_weights(weight) if (weight or plain_mean) else dict(default)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    summary = derive_single_index(multi_dir, out_dir, weights=weights, copy_bm25=copy_bm25, overwrite=overwrite, kind_mean=kind_mean)
    console.print(summary)
    console.print(f"Wrote {out_dir}")


if __name__ == "__main__":
    app()
