"""The retrieval benchmark: anchor titles and the v1 judge's strong-relevance labels.

Free to run and judge-free, so every retrieval configuration goes through it
before anything downstream is discussed. Two caveats travel with the numbers:
anchors are 55 hand-picked famous titles, and the labelled pairs all come from
the v1 retrieval's top-200 pool, so a configuration that finds relevant books
the old one never surfaced gets no credit. It ranks configurations; it does
not measure absolute quality.
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.config import PROJECT_ROOT
from src.evaluation import title_matches_anchor
from src.retrieval.query import retrieval_query

DEFAULT_QUERIES_PATH = PROJECT_ROOT / "eval" / "eval_queries.jsonl"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
STRONG_LABEL = "2"


@dataclass
class BenchQuery:
    query_id: str
    query: str
    anchors: list[str] = field(default_factory=list)
    strong: set[str] = field(default_factory=set)


def load_benchmark(queries_path: Path = DEFAULT_QUERIES_PATH, results_dir: Path = DEFAULT_RESULTS_DIR) -> list[BenchQuery]:
    queries: dict[str, BenchQuery] = {}
    with queries_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                queries[record["query_id"]] = BenchQuery(
                    query_id=record["query_id"], query=record["query"], anchors=list(record.get("anchor_titles") or [])
                )
    for csv_path in sorted(results_dir.rglob("eval_results_judged*.csv")):
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("judge_relevance_label") == STRONG_LABEL and row["query_id"] in queries:
                    queries[row["query_id"]].strong.add(row["novel_id"])
    return list(queries.values())


def anchor_rank(rows: Iterable[dict[str, Any]], anchor: str) -> int | None:
    for position, row in enumerate(rows, start=1):
        if title_matches_anchor(str(row.get("title_guess", "")), anchor):
            return int(row.get("rank") or position)
    return None


@dataclass
class BenchResult:
    name: str
    depth: int
    anchor_ranks: dict[str, int | None]
    recall_at_k: dict[str, float]
    recall_k: int
    strip_negatives: bool = True

    def metrics(self) -> dict[str, Any]:
        ranks = list(self.anchor_ranks.values())
        found = [rank for rank in ranks if rank is not None]
        censored = [rank if rank is not None else self.depth + 1 for rank in ranks]
        recalls = list(self.recall_at_k.values())
        return {
            "anchors": len(ranks),
            "anchor_hit@10": _share(ranks, 10),
            "anchor_hit@50": _share(ranks, 50),
            "anchor_hit@200": _share(ranks, 200),
            "anchor_median_rank": statistics.median(censored) if censored else None,
            "anchor_unfound": len(ranks) - len(found),
            f"recall@{self.recall_k}_macro": round(sum(recalls) / len(recalls), 4) if recalls else None,
            "strong_queries": len(recalls),
            "negatives_stripped": self.strip_negatives,
        }


def _share(ranks: list[int | None], k: int) -> float | None:
    if not ranks:
        return None
    return round(sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks), 4)


def evaluate(searcher: Any, queries: list[BenchQuery], depth: int = 1000, recall_k: int = 20, strip_negatives: bool = True) -> BenchResult:
    """Run every query once at ``depth`` and read both metric families off the same ranking.

    ``strip_negatives`` mirrors the search_books tool: the dense retriever sees
    positive terms only. The first sweep (2026-10-08, r3/r4) ran with the raw
    query and so measured 「不系统」 pulling system-novels in.
    """

    anchor_ranks: dict[str, int | None] = {}
    recall: dict[str, float] = {}
    for query in queries:
        if not query.anchors and not query.strong:
            continue
        text = retrieval_query(query.query) if strip_negatives else query.query
        rows = searcher.search(text, depth)
        for anchor in query.anchors:
            anchor_ranks[f"{query.query_id}|{anchor}"] = anchor_rank(rows, anchor)
        if query.strong:
            top = {str(row["novel_id"]) for row in rows[:recall_k]}
            recall[query.query_id] = len(top & query.strong) / len(query.strong)
    return BenchResult(
        name=getattr(searcher, "name", "searcher"),
        depth=depth,
        anchor_ranks=anchor_ranks,
        recall_at_k=recall,
        recall_k=recall_k,
        strip_negatives=strip_negatives,
    )


def format_table(results: list[BenchResult]) -> str:
    if not results:
        return ""
    columns = list(results[0].metrics().keys())
    lines = ["| config | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for result in results:
        metrics = result.metrics()
        lines.append(f"| {result.name} | " + " | ".join(str(metrics[column]) for column in columns) + " |")
    return "\n".join(lines)
