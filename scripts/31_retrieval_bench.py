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
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import BM25Searcher, HybridSearcher, MultiVectorSearcher, SingleVectorSearcher

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

    searchers = []
    metadata_path = index_dir / "index_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        embedder = load_embedding_model(model or metadata["model_name"], device=device, dtype=dtype or metadata.get("dtype", "fp32"))
        if (index_dir / "sections.json").exists():
            searchers.append(MultiVectorSearcher.load(embedder, index_dir, name=f"{label}/dense_multi"))
        else:
            searchers.append(SingleVectorSearcher.load(embedder, index_dir, name=f"{label}/dense_single"))
    if (index_dir / "bm25.json").exists():
        meta = json.loads((index_dir / "book_meta.json").read_text(encoding="utf-8"))
        searchers.append(BM25Searcher(BM25Index.load(index_dir / "bm25.json"), meta, name=f"{label}/bm25"))
    if not searchers:
        raise typer.BadParameter(f"Nothing to evaluate in {index_dir}")
    if len(searchers) == 2:
        searchers.append(HybridSearcher(list(searchers), depth=hybrid_depth, name=f"{label}/hybrid_rrf"))

    results = []
    for searcher in searchers:
        result = evaluate(searcher, queries, depth=depth, recall_k=recall_k)
        results.append(result)
        console.print(f"{result.name}: {result.metrics()}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for result in results:
            record = {"config": result.name, "index_dir": index_dir.as_posix(), "depth": depth, "recall_k": recall_k, "at": datetime.now(timezone.utc).isoformat(), **result.metrics()}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    console.print(format_table(results))


if __name__ == "__main__":
    app()
