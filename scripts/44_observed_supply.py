"""Candidate-supply evaluation on the agent's own trajectories: the pool is what the agent saw.

    uv run python scripts/44_observed_supply.py --run-id test-flash-02
    uv run python scripts/41_supply_judge.py --pool-dir eval/results/retrieval_supply/test-flash-02 --dry-run
    uv run python scripts/42_supply_metrics.py --pool-dir eval/results/retrieval_supply/test-flash-02 --verdicts .../verdicts_T1_qwen3.8-max.jsonl

No index is run (so no model, so it runs on the Mac): every search_books call in a request session
becomes a query whose returned books are the ``agent_seen`` rows, filters and k as the agent sent
them, and the books finish() recommended become ``agent_recommended`` rows. Queries of one session
share a request id, so the judge labels each (request, book) once and the same verdicts give the
index-side precision / feasibility and the agent-side recommendation satisfaction. Reads the full
trajectories under data/eval_runs/<run-id>/ when present (titles are intact in the redacted copy too).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from src.agent.backends import DEFAULT_DENSITY_PATH, load_density_table
from src.agent_eval.runner import RunPaths
from src.config import PROJECT_ROOT
from src.retrieval.cards import DEFAULT_CARDS_PATH, load_cards
from src.retrieval.supply import (
    DEFAULT_SUPPLY_RESULTS_DIR,
    RECOMMENDED_CONFIG,
    SEEN_CONFIG,
    SUPPLY_DEPTH,
    SUPPLY_K,
    format_metrics,
    judge_free_metrics,
    load_task_records,
    observed_supply,
    pool_pairs,
    write_pool,
)

app = typer.Typer(add_completion=False)
console = Console()
TASK_DIR = PROJECT_ROOT / "eval" / "agent" / "tasks"
DEFAULT_TASKS = [TASK_DIR / name for name in ("constrained_rec.jsonl", "memory.jsonl", "session_state.jsonl")]


@app.command()
def main(
    run_id: str = typer.Option(..., help="An agent evaluation run (scripts/35); its trajectories are the pool."),
    tasks: list[Path] = typer.Option(DEFAULT_TASKS, help="Task files; only kinds with a preference request contribute."),
    cards: Path = typer.Option(DEFAULT_CARDS_PATH),
    density: Path = typer.Option(DEFAULT_DENSITY_PATH),
    k: int = typer.Option(SUPPLY_K, help="Depth of the seen results that count (the tool's default k)."),
    out_dir: Path | None = typer.Option(None, help="Defaults to eval/results/retrieval_supply/<run-id>."),
) -> None:
    paths = RunPaths(run_dir=PROJECT_ROOT / "eval" / "agent" / "runs" / run_id, local_dir=PROJECT_ROOT / "data" / "eval_runs" / run_id)
    source = paths.full_path if paths.full_path.exists() else paths.redacted_path
    if not source.exists():
        raise typer.BadParameter(f"no trajectories for run {run_id!r} under {paths.run_dir} or {paths.local_dir}")
    queries, rows = observed_supply([source], load_task_records(tasks))
    out = out_dir or DEFAULT_SUPPLY_RESULTS_DIR / run_id
    write_pool(rows, queries, out)

    searches = [q for q in queries if q.source == "trajectory"]
    requests = {q.request_id for q in searches}
    seen_pairs = pool_pairs([r for r in rows if r.config == SEEN_CONFIG], SUPPLY_DEPTH, queries)  # everything the agent saw (k <= 20)
    rec_pairs = pool_pairs([r for r in rows if r.config == RECOMMENDED_CONFIG], SUPPLY_DEPTH, queries)
    all_pairs = pool_pairs(rows, SUPPLY_DEPTH, queries)
    console.print(f"run {run_id} ({source.name}): {len(searches)} search_books queries over {len(requests)} request sessions; kinds {dict(Counter(q.kind for q in searches))}")
    console.print(f"filters used: {dict(Counter(key for q in searches for key, _ in q.filters))}")
    console.print(f"(request, book) pairs: seen {len(seen_pairs)}, recommended {len(rec_pairs)} of which {len(set(rec_pairs) - set(seen_pairs))} not among the seen, to judge {len(all_pairs)}")

    card_map = load_cards(cards) if cards.exists() else {}
    densities = load_density_table(density) if density.exists() else None
    report = judge_free_metrics([r for r in rows if r.config == SEEN_CONFIG], queries, k=k, cards=card_map or None, densities=densities)
    summary = {
        "at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source": str(source.relative_to(PROJECT_ROOT)),
        "k": k,
        "queries": len(searches),
        "requests": len(requests),
        "pairs_seen": len(seen_pairs),
        "pairs_recommended": len(rec_pairs),
        "pairs_to_judge": len(all_pairs),
        "configs": report,
    }
    (out / "metrics_free.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    console.print(format_metrics(report))
    console.print(f"-> {out}")


if __name__ == "__main__":
    app()
