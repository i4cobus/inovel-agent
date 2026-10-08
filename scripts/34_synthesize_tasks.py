"""Synthesise the evaluation tasks and assign test / dev splits.

    uv run python scripts/34_synthesize_tasks.py --n-test 50 --n-dev 14 --mem-test 25 --mem-dev 6

Writes eval/agent/tasks/constrained_rec.jsonl and memory.jsonl. Optional
``--precheck`` (needs the index and density table, i.e. the PC) keeps only
constrained tasks whose top-20 retrieval pool holds at least one rule-violating
and one clean candidate for an in-text negative, so the task can discriminate.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from src.agent_eval.tasks import DEFAULT_TASKS_DIR, assign_splits, save_tasks, synthesize_constrained, synthesize_memory

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    out_dir: Path = typer.Option(DEFAULT_TASKS_DIR),
    n_test: int = typer.Option(50),
    n_dev: int = typer.Option(14),
    mem_test: int = typer.Option(25),
    mem_dev: int = typer.Option(6),
    seed: int = typer.Option(20261008),
    oversample: float = typer.Option(2.0, help="Generate this many times the needed constrained tasks before precheck / split."),
    precheck: bool = typer.Option(False, help="Filter constrained tasks by retrieval pool composition (needs index + density table)."),
    device: str | None = typer.Option(None),
) -> None:
    wanted = int((n_test + n_dev) * oversample)
    rec = synthesize_constrained(wanted, seed=seed)
    if precheck:
        rec = _precheck(rec, device)
        console.print(f"precheck kept {len(rec)} / {wanted}")
    rec = assign_splits(rec, n_test, n_dev, seed=seed)
    mem = assign_splits(synthesize_memory(mem_test + mem_dev, seed=seed), mem_test, mem_dev, seed=seed)
    save_tasks(out_dir / "constrained_rec.jsonl", rec)
    save_tasks(out_dir / "memory.jsonl", mem)
    console.print(f"constrained_rec: {len(rec)} ({sum(t.split == 'test' for t in rec)} test)  memory: {len(mem)} ({sum(t.split == 'test' for t in mem)} test)  -> {out_dir}")


def _precheck(tasks: list, device: str | None) -> list:
    from src.agent.backends import load_density_table
    from src.config import DEFAULT_INDEX_DIR
    from src.embed import load_embedding_model
    from src.preferences import constraint_violation_from_densities
    from src.retrieval.hybrid import index_metadata, load_searchers
    from src.retrieval.query import retrieval_query

    metadata = index_metadata(DEFAULT_INDEX_DIR)
    embedder = load_embedding_model(metadata["model_name"], device=device, dtype=metadata.get("dtype", "fp32"))
    searcher = load_searchers(DEFAULT_INDEX_DIR, embedder)[0]
    densities = load_density_table()
    kept = []
    for task in tasks:
        if not task.negatives_in_text:
            kept.append(task)  # meta-only tasks cannot be prechecked by rule
            continue
        rows = searcher.search(retrieval_query(task.sessions[0].user_message), 20)
        verdicts = [constraint_violation_from_densities(densities.get(str(r["novel_id"]), {}), task.negatives_in_text) for r in rows]
        if any(v is True for v in verdicts) and any(v is False for v in verdicts):
            task.meta["pool_violators"] = sum(1 for v in verdicts if v is True)
            task.meta["pool_clean"] = sum(1 for v in verdicts if v is False)
            kept.append(task)
    return kept


if __name__ == "__main__":
    app()
