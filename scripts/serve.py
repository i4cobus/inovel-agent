"""Serve the web UI: sessions over the real index and a chat model.

    uv run python scripts/serve.py --base-url <OpenAI 兼容地址> --api-key-file <key 文件> --no-thinking --port 8765

Binds to 127.0.0.1 by default: on the PC reach it from the Mac through an ssh tunnel
(``ssh -N -L 8765:127.0.0.1:8765 pc``), then open http://127.0.0.1:8765/. Sessions live under
data/sessions/<id>/ (memory, turns, full trajectories); they survive a restart.
"""

from __future__ import annotations

from pathlib import Path

import typer

from src.agent.backends import DEFAULT_AGENT_MODEL, DEFAULT_CHAT_BASE_URL, build_agent
from src.agent.loop import AgentConfig
from src.agent.sessions import DEFAULT_SESSIONS_DIR, SessionStore
from src.agent.web import create_app
from src.config import DEFAULT_INDEX_DIR
from src.retrieval.cards import DEFAULT_CARDS_PATH

app = typer.Typer(add_completion=False)


@app.command()
def main(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8765),
    model: str = typer.Option(DEFAULT_AGENT_MODEL),
    base_url: str = typer.Option(DEFAULT_CHAT_BASE_URL),
    index_dir: Path = typer.Option(DEFAULT_INDEX_DIR),
    cards: Path | None = typer.Option(DEFAULT_CARDS_PATH),
    sessions_dir: Path = typer.Option(DEFAULT_SESSIONS_DIR),
    device: str | None = typer.Option(None, help="Device for the query embedder."),
    max_steps: int = typer.Option(10),
    reasoning_effort: str | None = typer.Option(None),
    api_key_file: Path | None = typer.Option(None, help="File holding the endpoint's API key; else INOVELREC_LLM_API_KEY."),
    no_thinking: bool = typer.Option(False, help="Send enable_thinking=false (百炼 Qwen3.x) and reasoning_effort=none (Ollama)."),
    multi_dir: Path | None = typer.Option(None, help="Multi-vector index for ask_book; default: the sibling multi_* of --index-dir."),
) -> None:
    import uvicorn

    api_key = api_key_file.read_text(encoding="utf-8").strip() if api_key_file else None
    extra_body = {"enable_thinking": False} if no_thinking else None
    if no_thinking and reasoning_effort is None:
        reasoning_effort = "none"
    bundle = build_agent(model=model, base_url=base_url, index_dir=index_dir, cards_path=cards, device=device, reasoning_effort=reasoning_effort, api_key=api_key, extra_body=extra_body, multi_dir=multi_dir, config=AgentConfig(max_steps=max_steps))
    print(f"ready {bundle.info}", flush=True)
    store = SessionStore(bundle, root=sessions_dir)
    print(f"sessions: {len(store.sessions)} restored from {sessions_dir}", flush=True)
    uvicorn.run(create_app(store), host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
