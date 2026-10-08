"""Assemble the agent from real artifacts on disk.

Everything the tools need is built here, once, from the paths in ``src.config``:
the profile table, the book index chosen by ``DEFAULT_INDEX_DIR``, the term
density table, the inventory (for raw text behind ``check_trope``), the user
memory file, and an OpenAI-compatible chat endpoint (Ollama or vLLM). Tests
never call ``build_agent``; they assemble the same pieces from stubs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.agent.context import ContextBudget
from src.agent.loop import AgentConfig, AgentLoop
from src.agent.memory import DEFAULT_MEMORY_PATH, UserMemory
from src.agent.tools import (
    ToolRegistry,
    build_check_term,
    build_check_trope,
    build_get_profile,
    build_memory_tools,
    build_search_books,
)
from src.agent.trope import CachedTropeJudge
from src.chat_transport import HTTPChatTransport
from src.config import DEFAULT_INDEX_DIR, DEFAULT_OUTPUT_PATH, PROCESSED_DATA_DIR
from src.vector_index import DEFAULT_PROFILES_PATH

DEFAULT_DENSITY_PATH = PROCESSED_DATA_DIR / "term_density.parquet"
DEFAULT_CHAT_BASE_URL = "http://127.0.0.1:11434/v1"  # Ollama's OpenAI-compatible endpoint
DEFAULT_AGENT_MODEL = "qwen3.5:9b"


class ParquetProfiles:
    """``ProfileLookup`` over novel_profiles.parquet, loaded once into memory."""

    def __init__(self, rows: dict[str, dict[str, str]]) -> None:
        self.rows = rows

    @classmethod
    def load(cls, path: Path = DEFAULT_PROFILES_PATH) -> "ParquetProfiles":
        import pandas as pd

        frame = pd.read_parquet(path, columns=["novel_id", "title_guess", "profile_text"])
        rows = {
            str(row.novel_id): {"title": str(row.title_guess or ""), "profile": str(row.profile_text or "")}
            for row in frame.itertuples(index=False)
        }
        return cls(rows)

    def get(self, novel_id: str) -> dict[str, str] | None:
        return self.rows.get(novel_id)

    def __len__(self) -> int:
        return len(self.rows)


def load_density_table(path: Path = DEFAULT_DENSITY_PATH) -> dict[str, dict[str, float]]:
    """novel_id -> {surface form -> occurrences per 100k chars}, as script 16 writes it."""

    import pandas as pd

    frame = pd.read_parquet(path)
    terms = [column for column in frame.columns if column not in ("novel_id", "char_count")]
    table: dict[str, dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        record = row._asdict()
        table[str(record["novel_id"])] = {term: float(record[term] or 0.0) for term in terms}
    return table


class LazyRawText:
    """Reads a novel's raw text on first request, from the inventory's path and encoding."""

    def __init__(self, inventory_path: Path = DEFAULT_OUTPUT_PATH, cache_size: int = 8) -> None:
        import pandas as pd

        wanted = ["novel_id", "absolute_path", "detected_encoding", "read_status", "decode_replacement_chars"]
        import pyarrow.parquet as pq

        present = set(pq.read_schema(inventory_path).names)
        frame = pd.read_parquet(inventory_path, columns=[c for c in wanted if c in present])
        self.rows = {str(r["novel_id"]): r for r in frame.to_dict(orient="records") if r.get("read_status") == "ok"}
        self.cache_size = cache_size
        self._cache: dict[str, str] = {}

    def __call__(self, novel_id: str) -> str | None:
        if novel_id in self._cache:
            return self._cache[novel_id]
        row = self.rows.get(novel_id)
        if row is None:
            return None
        from src.profile import read_text_with_encoding

        try:
            text = read_text_with_encoding(
                Path(str(row["absolute_path"])),
                row.get("detected_encoding"),
                allow_lossy=int(row.get("decode_replacement_chars", 0) or 0) > 0,
            )
        except (OSError, UnicodeError, LookupError, ValueError):
            return None
        if len(self._cache) >= self.cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[novel_id] = text
        return text


@dataclass
class AgentBundle:
    loop: AgentLoop
    tools: ToolRegistry
    memory: UserMemory
    memory_path: Path
    turn: int = 0
    info: dict[str, Any] = field(default_factory=dict)

    def chat(self, user_message: str, history: list[dict[str, Any]] | None = None, task_id: str = "") -> Any:
        self.turn += 1
        run = self.loop.run(user_message, history=history, task_id=task_id)
        self.memory.save(self.memory_path)
        return run


def build_agent(
    model: str = DEFAULT_AGENT_MODEL,
    base_url: str = DEFAULT_CHAT_BASE_URL,
    index_dir: Path = DEFAULT_INDEX_DIR,
    profiles_path: Path = DEFAULT_PROFILES_PATH,
    density_path: Path = DEFAULT_DENSITY_PATH,
    inventory_path: Path = DEFAULT_OUTPUT_PATH,
    memory_path: Path = DEFAULT_MEMORY_PATH,
    device: str | None = None,
    embedding_dtype: str | None = None,
    trope_model: str | None = None,
    config: AgentConfig = AgentConfig(),
    budget: ContextBudget = ContextBudget(),
    embedder_factory: Callable[..., Any] | None = None,
) -> AgentBundle:
    """Load every artifact and return a ready agent. Slow: loads the embedder."""

    from src.embed import load_embedding_model
    from src.retrieval.hybrid import index_metadata, load_searchers

    metadata = index_metadata(index_dir)
    factory = embedder_factory or load_embedding_model
    embedder = factory(metadata["model_name"], device=device, dtype=embedding_dtype or metadata.get("dtype", "fp32"))
    searcher = load_searchers(index_dir, embedder)[0]  # the dense searcher; BM25/hybrid are not search_books

    profiles = ParquetProfiles.load(profiles_path)
    densities = load_density_table(density_path)
    memory = UserMemory.load(memory_path)
    transport = HTTPChatTransport(model=model, base_url=base_url)
    trope_transport = HTTPChatTransport(model=trope_model or model, base_url=base_url)
    judge = CachedTropeJudge(trope_transport, LazyRawText(inventory_path), model_name=trope_model or model)

    bundle = AgentBundle(loop=None, tools=ToolRegistry(), memory=memory, memory_path=memory_path)  # type: ignore[arg-type]
    bundle.tools.register(build_search_books(searcher, budget))
    bundle.tools.register(build_get_profile(profiles, budget))
    bundle.tools.register(build_check_term(densities))
    bundle.tools.register(build_check_trope(judge))
    for spec in build_memory_tools(memory, turn_counter=lambda: bundle.turn):
        bundle.tools.register(spec)
    bundle.loop = AgentLoop(transport, bundle.tools, memory, config=config, model_name=model)
    bundle.info = {
        "model": model,
        "base_url": base_url,
        "index_dir": index_dir.as_posix(),
        "searcher": searcher.name,
        "profiles": len(profiles),
        "density_rows": len(densities),
    }
    return bundle
