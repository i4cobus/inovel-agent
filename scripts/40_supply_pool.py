"""Candidate-supply bench, step 1: build the pool and the judge-free metrics.

    uv run python scripts/40_supply_pool.py --index-dir data/index/single_4b --index-dir data/index/multi_4b \
        --cards data/processed/book_cards_flash.parquet --density data/processed/term_density.parquet

Every searcher an index directory supports becomes one configuration (dense, BM25, hybrid); with
--cards each dense searcher also gets a card-filtered twin. Queries: the agent tasks (always), the
v1 queries for their anchors (--legacy), rewrites (--rewrites) and trajectory queries
(--trajectories). Writes pool.jsonl + queries.jsonl + metrics_free.json into --out-dir.
On macOS run with KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from src.agent.backends import DEFAULT_DENSITY_PATH, load_density_table
from src.embed import load_embedding_model
from src.retrieval.cardfilter import CardFilteredSearcher
from src.retrieval.cards import DEFAULT_CARDS_PATH, load_cards
from src.retrieval.hybrid import index_metadata, load_searchers
from src.retrieval.supply import (
    DEFAULT_LEGACY_QUERIES_PATH,
    DEFAULT_REWRITES_PATH,
    DEFAULT_SUPPLY_RESULTS_DIR,
    DEFAULT_TASK_PATHS,
    SUPPLY_DEPTH,
    SUPPLY_K,
    build_pool,
    format_metrics,
    judge_free_metrics,
    legacy_queries,
    pool_pairs,
    rewrite_queries,
    task_queries,
    trajectory_queries,
    write_pool,
)

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    index_dir: list[Path] = typer.Option(..., help="Index directories; repeat for several configurations."),
    cards: Path | None = typer.Option(DEFAULT_CARDS_PATH, help="Book cards: adds a card-filtered twin of each dense searcher and the card-consistency metric. A missing file disables both."),
    density: Path = typer.Option(DEFAULT_DENSITY_PATH),
    tasks: list[Path] = typer.Option(list(DEFAULT_TASK_PATHS)),
    legacy: bool = typer.Option(True, help="Include the v1 queries (anchor titles)."),
    legacy_path: Path = typer.Option(DEFAULT_LEGACY_QUERIES_PATH),
    rewrites: Path = typer.Option(DEFAULT_REWRITES_PATH, help="Paraphrases JSONL (task_id, variant, query); skipped when missing."),
    trajectories: list[Path] = typer.Option([], help="Trajectory JSONL files; their search_books queries join the set."),
    depth: int = typer.Option(SUPPLY_DEPTH, help="Pool depth per configuration (the judge labels this many per query)."),
    k: int = typer.Option(SUPPLY_K, help="What the agent sees."),
    device: str | None = typer.Option(None),
    out_dir: Path = typer.Option(DEFAULT_SUPPLY_RESULTS_DIR),
) -> None:
    queries = task_queries(tasks)
    by_task = {q.task_id: q for q in queries}
    if legacy:
        queries += legacy_queries(legacy_path)
    queries += rewrite_queries(rewrites, by_task)
    if trajectories:
        queries += trajectory_queries(trajectories, by_task)
    console.print(f"queries: {len(queries)} " + str({s: sum(1 for q in queries if q.source == s) for s in ("task", "legacy", "rewrite", "trajectory")}))

    card_map = load_cards(cards) if cards is not None and cards.exists() else {}
    searchers = []
    for directory in index_dir:
        metadata = index_metadata(directory)
        embedder = load_embedding_model(metadata["model_name"], device=device, dtype=metadata.get("dtype", "fp32")) if (directory / "faiss.index").exists() else None
        for searcher in load_searchers(directory, embedder):
            searchers.append(searcher)
            if card_map and "dense" in searcher.name:
                searchers.append(CardFilteredSearcher(searcher, card_map))
    console.print("configs: " + ", ".join(s.name for s in searchers))

    rows, latencies = build_pool(searchers, queries, depth=depth)
    write_pool(rows, queries, out_dir)
    densities = load_density_table(density) if density.exists() else None
    report = judge_free_metrics(rows, queries, k=k, cards=card_map or None, densities=densities, latencies=latencies)
    (out_dir / "metrics_free.json").write_text(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "k": k, "depth": depth, "configs": report}, ensure_ascii=False, indent=1), encoding="utf-8")
    console.print(format_metrics(report))
    console.print(f"pool: {len(rows)} rows, {len(pool_pairs(rows, depth))} distinct (query, book) pairs to judge -> {out_dir}")


if __name__ == "__main__":
    app()
