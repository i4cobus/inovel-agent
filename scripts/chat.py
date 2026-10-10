"""Talk to the agent from a terminal, against the real index and a local model server.

    uv run python scripts/chat.py --model qwen3.5:9b --base-url http://127.0.0.1:11434/v1

Every turn's trajectory is appended, unredacted, to data/trajectories/chat.jsonl
(local only); memory is saved after each turn.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from src.retrieval.cards import DEFAULT_CARDS_PATH
from src.agent.backends import DEFAULT_AGENT_MODEL, DEFAULT_CHAT_BASE_URL, build_agent
from src.agent.loop import AgentConfig
from src.agent.trajectory import append_jsonl
from src.config import DATA_DIR, DEFAULT_INDEX_DIR

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    model: str = typer.Option(DEFAULT_AGENT_MODEL),
    base_url: str = typer.Option(DEFAULT_CHAT_BASE_URL),
    index_dir: Path = typer.Option(DEFAULT_INDEX_DIR),
    cards: Path | None = typer.Option(DEFAULT_CARDS_PATH, help="Book cards parquet for get_profile; a missing file means no cards."),
    device: str | None = typer.Option(None, help="Device for the query embedder; cpu is fine."),
    max_steps: int = typer.Option(10),
    reasoning_effort: str | None = typer.Option(None, help="Ollama thinking control, e.g. none / low / high; None leaves the model default."),
    show_steps: bool = typer.Option(True, "--show-steps/--quiet"),
    trajectories: Path = typer.Option(DATA_DIR / "trajectories" / "chat.jsonl"),
    once: str | None = typer.Option(None, help="Ask one question and exit (for smoke tests)."),
) -> None:
    agent = build_agent(model=model, base_url=base_url, index_dir=index_dir, cards_path=cards, device=device, reasoning_effort=reasoning_effort, config=AgentConfig(max_steps=max_steps))
    console.print(f"[bold]ready[/bold] {agent.info}")
    history: list[dict] = []

    def ask(text: str) -> None:
        run = agent.chat(text, history=history, task_id=datetime.now(timezone.utc).strftime("chat-%Y%m%dT%H%M%S"))
        traj = run.trajectory
        if show_steps:
            for step in traj.steps:
                for call, obs in zip(step.tool_calls, step.observations):
                    status = f"error: {obs.error}" if obs.error else f"{obs.latency_s}s"
                    console.print(f"  [dim]step {step.index}[/dim] {call['name']}({json.dumps(call['arguments'], ensure_ascii=False)[:120]}) → {status}")
        console.print(f"[bold cyan]agent[/bold cyan] ({traj.termination}, {traj.step_count} steps, {traj.prompt_tokens}+{traj.completion_tokens} tok)\n{traj.final_answer}\n")
        append_jsonl(trajectories, traj.to_dict())
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": traj.final_answer})

    if once:
        ask(once)
        return
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text in ("exit", "quit", "q"):
            break
        ask(text)


if __name__ == "__main__":
    app()
