"""Synthesise the evaluation tasks and assign test / dev splits.

    uv run python scripts/34_synthesize_tasks.py

Writes eval/agent/tasks/{constrained_rec,memory,session_state,clarify,book_qa}.jsonl. The first batch of
constrained_rec (ids < 128) and memory (ids < 31) is generated exactly as on 2026-10-08, so the original
75 test / 20 dev tasks keep their ids and splits; the extension batches (2026-10-27) use seed+1 and
continue the numbering. Optional
``--precheck`` (needs the index and density table, i.e. the PC) keeps only
constrained tasks whose top-20 retrieval pool holds at least one rule-violating
and one clean candidate for an in-text negative, so the task can discriminate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer
from rich.console import Console

from src.agent_eval.tasks import (
    DEFAULT_TASKS_DIR,
    Task,
    assign_splits,
    load_digest_books,
    save_tasks,
    synthesize_book_qa,
    synthesize_clarify,
    synthesize_constrained,
    synthesize_memory,
    synthesize_session_state,
)
from src.digest import DEFAULT_DIGEST_PATH
from src.retrieval.cards import DEFAULT_CARDS_PATH

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    out_dir: Path = typer.Option(DEFAULT_TASKS_DIR),
    n_test: int = typer.Option(50),
    n_dev: int = typer.Option(14),
    mem_test: int = typer.Option(25),
    mem_dev: int = typer.Option(6),
    rec_extra_test: int = typer.Option(90, help="Extension batch of constrained_rec (ids from 128)."),
    rec_extra_dev: int = typer.Option(22),
    mem_extra_test: int = typer.Option(35, help="Extension batch of memory (ids from 31)."),
    mem_extra_dev: int = typer.Option(10),
    state_test: int = typer.Option(50, help="session_state tasks (换几本 / 第二本 / 看过了)."),
    state_dev: int = typer.Option(14),
    clarify_test: int = typer.Option(12, help="clarify tasks (vague requests)."),
    clarify_dev: int = typer.Option(4),
    qa_test: int = typer.Option(45, help="book_qa tasks in the test split; 0 skips the kind."),
    qa_dev: int = typer.Option(15),
    digests: Path = typer.Option(DEFAULT_DIGEST_PATH, help="Digest table the book_qa tasks are generated from."),
    cards: Path = typer.Option(DEFAULT_CARDS_PATH, help="Only books with a card are used for book_qa."),
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
    if rec_extra_test + rec_extra_dev > 0:
        rec += _extension(lambda k: synthesize_constrained(k, seed=seed + 1, start=wanted), rec, rec_extra_test, rec_extra_dev, seed + 1)
    if mem_extra_test + mem_extra_dev > 0:
        mem += _extension(lambda k: synthesize_memory(k, seed=seed + 1, start=mem_test + mem_dev), mem, mem_extra_test, mem_extra_dev, seed + 1)
    save_tasks(out_dir / "constrained_rec.jsonl", sorted(rec, key=lambda t: t.task_id))
    save_tasks(out_dir / "memory.jsonl", sorted(mem, key=lambda t: t.task_id))
    if state_test + state_dev > 0:
        save_tasks(out_dir / "session_state.jsonl", assign_splits(synthesize_session_state(state_test + state_dev, seed=seed), state_test, state_dev, seed=seed))
    if clarify_test + clarify_dev > 0:
        save_tasks(out_dir / "clarify.jsonl", assign_splits(synthesize_clarify(clarify_test + clarify_dev, seed=seed), clarify_test, clarify_dev, seed=seed))
    if qa_test + qa_dev > 0:
        books = load_digest_books(digests, cards_path=cards)
        qa = assign_splits(synthesize_book_qa(books, qa_test + qa_dev, seed=seed), qa_test, qa_dev, seed=seed)
        save_tasks(out_dir / "book_qa.jsonl", qa)
        typer.echo(f"book_qa: {len(qa)} tasks from {len(books)} books")
    console.print(f"constrained_rec: {len(rec)} ({sum(t.split == 'test' for t in rec)} test)  memory: {len(mem)} ({sum(t.split == 'test' for t in mem)} test)  session_state: {state_test + state_dev}  clarify: {clarify_test + clarify_dev}  -> {out_dir}")


def _extension(make: Callable[[int], list[Task]], existing: list[Task], n_test: int, n_dev: int, seed: int) -> list[Task]:
    """A later batch: generate with headroom, drop any task whose conversation duplicates an existing one, then split."""

    def key(task: Task) -> tuple[str, ...]:
        return tuple(s.user_message for s in task.sessions)

    seen = {key(t) for t in existing}
    fresh: list[Task] = []
    for task in make((n_test + n_dev) * 3):
        if key(task) in seen:
            continue
        seen.add(key(task))
        fresh.append(task)
    if len(fresh) < n_test + n_dev:
        raise RuntimeError(f"only {len(fresh)} distinct tasks for an extension of {n_test + n_dev}")
    kept = assign_splits(fresh, n_test, n_dev, seed=seed)
    prefix, first = kept[0].task_id.rsplit("-", 1)[0], min(int(t.task_id.rsplit("-", 1)[1]) for t in fresh)
    for offset, task in enumerate(kept):  # contiguous ids after the original batch
        task.task_id = f"{prefix}-{first + offset:03d}"
    return kept


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
