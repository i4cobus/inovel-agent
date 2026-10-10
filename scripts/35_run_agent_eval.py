"""Run the agent over a task file and record trajectories.

    uv run python scripts/35_run_agent_eval.py --tasks eval/agent/tasks/constrained_rec.jsonl --split dev --run-id dev-01

Redacted trajectories, memory states and the config go to eval/agent/runs/<run_id>/;
full trajectories to data/eval_runs/<run_id>/. Score with 36_agent_metrics.py.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from src.retrieval.cards import DEFAULT_CARDS_PATH
from src.agent.backends import DEFAULT_AGENT_MODEL, DEFAULT_CHAT_BASE_URL, build_agent
from src.agent.loop import AgentConfig
from src.agent_eval.runner import RunPaths, run_tasks
from src.agent_eval.tasks import load_tasks
from src.config import DATA_DIR, DEFAULT_INDEX_DIR, PROJECT_ROOT

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    run_id: str = typer.Option(...),
    tasks: list[Path] = typer.Option(..., help="Task JSONL files; repeat the option for several."),
    split: str = typer.Option("dev", help="test | dev | all"),
    limit: int | None = typer.Option(None),
    model: str = typer.Option(DEFAULT_AGENT_MODEL),
    base_url: str = typer.Option(DEFAULT_CHAT_BASE_URL),
    reasoning_effort: str | None = typer.Option("none"),
    max_steps: int = typer.Option(10),
    index_dir: Path = typer.Option(DEFAULT_INDEX_DIR),
    cards: Path | None = typer.Option(DEFAULT_CARDS_PATH, help="Book cards parquet for get_profile; a missing file means no cards."),
    device: str | None = typer.Option("cpu", help="Device for the query embedder; the GPU belongs to the model server."),
) -> None:
    selected = []
    for path in tasks:
        for task in load_tasks(path):
            if split == "all" or task.split == split:
                selected.append(task)
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise typer.BadParameter("no tasks selected")
    paths = RunPaths(run_dir=PROJECT_ROOT / "eval" / "agent" / "runs" / run_id, local_dir=DATA_DIR / "eval_runs" / run_id)
    if paths.redacted_path.exists():
        raise typer.BadParameter(f"{paths.run_dir} already has trajectories; pick another run id")
    agent = build_agent(model=model, base_url=base_url, index_dir=index_dir, cards_path=cards, device=device, reasoning_effort=reasoning_effort, config=AgentConfig(max_steps=max_steps))
    console.print(f"ready {agent.info}; {len(selected)} tasks ({split})")
    config = {"run_id": run_id, "split": split, "model": model, "reasoning_effort": reasoning_effort, "max_steps": max_steps, "index_dir": index_dir.as_posix(), "task_files": [p.as_posix() for p in tasks], **agent.info}

    def report(task, trajectories) -> None:
        ends = ",".join(t["termination"] for t in trajectories)
        steps = sum(len(t["steps"]) for t in trajectories)
        console.print(f"  {task.task_id} [{task.variant}] {ends} steps={steps}")

    records = run_tasks(agent, selected, paths, config, on_task=report)
    console.print(f"done: {len(records)} tasks -> {paths.run_dir}")


if __name__ == "__main__":
    app()
