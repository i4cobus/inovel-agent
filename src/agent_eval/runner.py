"""Run tasks through an agent and write trajectories, memory snapshots and a config record.

Each task gets its own memory file, so sessions within a task share memory and
tasks never leak into each other. Sessions run with no chat history: that is
what "cross-session" means here. The full trajectories stay under ``data/``;
the redacted copies go next to the summary for committing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from src.agent.trajectory import append_jsonl
from src.agent_eval.tasks import Task


class TaskAgent(Protocol):
    """What the runner needs from an agent: AgentBundle satisfies it."""

    memory: Any
    tools: Any

    def reset_memory(self, memory_path: Path) -> None: ...

    def chat(self, user_message: str, history: list[dict[str, Any]] | None = None, task_id: str = "") -> Any: ...


@dataclass
class RunPaths:
    run_dir: Path  # committed: redacted trajectories, memory snapshots, config, metrics
    local_dir: Path  # not committed: full trajectories

    @property
    def redacted_path(self) -> Path:
        return self.run_dir / "trajectories.redacted.jsonl"

    @property
    def full_path(self) -> Path:
        return self.local_dir / "trajectories.jsonl"

    @property
    def memory_dir(self) -> Path:
        return self.local_dir / "memory"


def run_tasks(
    agent: TaskAgent,
    tasks: list[Task],
    paths: RunPaths,
    config: dict[str, Any],
    on_task: Callable[[Task, list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run every task; return one record per task with its trajectories and memory states."""

    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.local_dir.mkdir(parents=True, exist_ok=True)
    paths.memory_dir.mkdir(parents=True, exist_ok=True)
    (paths.run_dir / "config.json").write_text(
        json.dumps({**config, "started_at": datetime.now(timezone.utc).isoformat(), "tasks": len(tasks)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    redactors = agent.tools.redactors()
    records: list[dict[str, Any]] = []
    for task in tasks:
        memory_path = paths.memory_dir / f"{task.task_id}.json"
        if memory_path.exists():
            memory_path.unlink()
        agent.reset_memory(memory_path)
        trajectories: list[dict[str, Any]] = []
        memory_states: list[dict[str, Any]] = []
        started = time.perf_counter()
        for index, session in enumerate(task.sessions):
            run = agent.chat(session.user_message, history=None, task_id=f"{task.task_id}/s{index}")
            trajectory = run.trajectory
            trajectory.metadata.update({"task_id": task.task_id, "session": index, "kind": task.kind, "variant": task.variant})
            trajectories.append(trajectory.to_dict())
            memory_states.append(agent.memory.to_dict())
            append_jsonl(paths.full_path, trajectory.to_dict())
            append_jsonl(paths.redacted_path, trajectory.redacted(redactors))
        record = {
            "task_id": task.task_id,
            "kind": task.kind,
            "variant": task.variant,
            "split": task.split,
            "seconds": round(time.perf_counter() - started, 1),
            "trajectories": trajectories,
            "memory_states": memory_states,
        }
        append_jsonl(paths.run_dir / "memory_states.jsonl", {"task_id": task.task_id, "memory_states": memory_states})
        records.append(record)
        if on_task is not None:
            on_task(task, trajectories)
    return records


def load_run(paths: RunPaths) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Redacted trajectories (one per session) and memory states keyed by task, as written by run_tasks."""

    trajectories = [json.loads(line) for line in paths.redacted_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    memory_states: dict[str, list[dict[str, Any]]] = {}
    states_path = paths.run_dir / "memory_states.jsonl"
    if states_path.exists():
        for line in states_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                memory_states[record["task_id"]] = record["memory_states"]
    return trajectories, memory_states
