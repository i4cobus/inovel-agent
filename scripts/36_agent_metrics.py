"""Score a run with the hard metrics and write rows, a summary and a Markdown table.

    uv run python scripts/36_agent_metrics.py --run-id dev-01 --tasks eval/agent/tasks/constrained_rec.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from src.agent.backends import DEFAULT_DENSITY_PATH, load_density_table
from src.agent_eval.metrics import aggregate, format_summary, score_task
from src.agent_eval.runner import RunPaths, load_run
from src.agent_eval.tasks import load_digest_books
from src.digest import DEFAULT_DIGEST_PATH
from src.agent_eval.tasks import load_tasks
from src.config import DATA_DIR, PROJECT_ROOT

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    run_id: str = typer.Option(...),
    tasks: list[Path] = typer.Option(...),
    density_path: Path = typer.Option(DEFAULT_DENSITY_PATH),
    digests: Path = typer.Option(DEFAULT_DIGEST_PATH, help="Digest table; read only when the task files contain book_qa tasks."),
) -> None:
    paths = RunPaths(run_dir=PROJECT_ROOT / "eval" / "agent" / "runs" / run_id, local_dir=DATA_DIR / "eval_runs" / run_id)
    trajectories, memory_states = load_run(paths)
    by_task: dict[str, list[dict]] = {}
    for trajectory in trajectories:
        by_task.setdefault(str(trajectory["metadata"].get("task_id")), []).append(trajectory)
    task_index = {t.task_id: t.to_dict() for path in tasks for t in load_tasks(path)}
    densities = load_density_table(density_path)
    qa_ids = {str(s.get("expect", {}).get("novel_id", "")) for t in task_index.values() if t.get("kind") == "book_qa" for s in t.get("sessions", [])}
    books = load_digest_books(digests, ids=qa_ids) if qa_ids else {}
    rows = []
    for task_id, trajs in by_task.items():
        task = task_index.get(task_id)
        if task is None:
            console.print(f"[yellow]skip {task_id}: not in the task files[/yellow]")
            continue
        trajs.sort(key=lambda t: int(t["metadata"].get("session", 0)))
        rows.append(score_task(task, trajs, memory_states.get(task_id, [{}] * len(trajs)), densities, digests=books))
    summary = aggregate(rows)
    (paths.run_dir / "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (paths.run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    table = format_summary(summary)
    (paths.run_dir / "summary.md").write_text(table + "\n", encoding="utf-8")
    console.print(table)


if __name__ == "__main__":
    app()
