"""Run the retrieval benchmark over one index directory: dense, BM25, and their RRF hybrid.

Results append to eval/results/retrieval_bench/results.jsonl, one line per
configuration, and the table is printed as Markdown ready for the README.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from src.config import PROJECT_ROOT
from src.embed import DEFAULT_EMBEDDING_MODEL, load_embedding_model
from src.retrieval.bench import evaluate, format_table, load_benchmark
from src.retrieval.hybrid import index_metadata, load_searchers

app = typer.Typer(add_completion=False)
console = Console()
DEFAULT_OUT = PROJECT_ROOT / "eval" / "results" / "retrieval_bench" / "results.jsonl"


@app.command()
def main(
    index_dir: Path = typer.Argument(..., help="Directory written by 30_build_book_indexes.py"),
    label: str | None = typer.Option(None, help="Config label prefix; defaults to the directory name."),
    model: str | None = typer.Option(None, help="Embedding model; defaults to the one recorded in index_metadata.json."),
    device: str | None = typer.Option(None),
    dtype: str | None = typer.Option(None, help="Defaults to the dtype recorded at build time."),
    depth: int = typer.Option(1000, help="Ranking depth per query; anchors beyond it count as unfound."),
    recall_k: int = typer.Option(20),
    hybrid_depth: int = typer.Option(100, help="Candidates each searcher contributes to RRF."),
    out: Path = typer.Option(DEFAULT_OUT),
) -> None:
    label = label or index_dir.name
    queries = load_benchmark()
    console.print(f"Benchmark: {len(queries)} queries, {sum(len(q.anchors) for q in queries)} anchors, {sum(len(q.strong) for q in queries)} strong pairs")

    metadata = index_metadata(index_dir)
    embedder = None
    if (index_dir / "faiss.index").exists():
        embedder = load_embedding_model(model or metadata["model_name"], device=device, dtype=dtype or metadata.get("dtype", "fp32"))
    try:
        searchers = load_searchers(index_dir, embedder, label=label, hybrid_depth=hybrid_depth)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc

    results = []
    for searcher in searchers:
        result = evaluate(searcher, queries, depth=depth, recall_k=recall_k)
        results.append(result)
        console.print(f"{result.name}: {result.metrics()}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for result in results:
            record = {
                "config": result.name,
                "index_dir": index_dir.as_posix(),
                "model": metadata.get("model_name"),
                "digest_version": metadata.get("digest_version"),
                "depth": depth,
                "recall_k": recall_k,
                "at": datetime.now(timezone.utc).isoformat(),
                **result.metrics(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    console.print(format_table(results))


if __name__ == "__main__":
    app()
