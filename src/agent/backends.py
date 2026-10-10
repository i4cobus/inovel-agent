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
from src.retrieval.cards import DEFAULT_CARDS_PATH
from src.vector_index import DEFAULT_PROFILES_PATH

DEFAULT_DENSITY_PATH = PROCESSED_DATA_DIR / "term_density.parquet"
DEFAULT_CHAT_BASE_URL = "http://127.0.0.1:11434/v1"  # Ollama's OpenAI-compatible endpoint
DEFAULT_AGENT_MODEL = "qwen3.5:9b"


class ParquetProfiles:
    """``ProfileLookup`` over the digest table (``novel_digests.parquet``), loaded once into memory.

    Each row keeps ``title``, ``blurb`` (the digest's header + synopsis section), ``profile`` (the
    opening chapters, cut to ``keep_chars``: get_profile returns at most 1,200 of them and a whole
    digest_v2 table is 330M characters) and ``card`` (the book card's text, when a cards parquet is
    given). An old profile table without ``sections_json`` keeps its text head as ``profile``.
    """

    def __init__(self, rows: dict[str, dict[str, str]], digest_version: str | None = None, cards: int = 0) -> None:
        self.rows = rows
        self.digest_version = digest_version
        self.cards = cards

    @classmethod
    def load(cls, path: Path = DEFAULT_PROFILES_PATH, cards_path: Path | None = None, keep_chars: int = 3000) -> "ParquetProfiles":
        import json

        import pyarrow.parquet as pq

        names = set(pq.read_schema(path).names)
        # A digest table carries its sections explicitly; the joined profile_text is then redundant and
        # is not read (reading both peaked at 5 GB of RSS on the 1.27 GB digest_v2 table). Row batches
        # keep the peak at one batch of texts rather than the whole column.
        wanted = ("novel_id", "title_guess", "sections_json", "digest_version") if "sections_json" in names else ("novel_id", "title_guess", "profile_text")
        columns = [c for c in wanted if c in names]
        card_texts: dict[str, str] = {}
        if cards_path is not None and cards_path.exists():
            from src.retrieval.cards import load_cards

            card_texts = {novel_id: card.text() for novel_id, card in load_cards(cards_path).items()}
        rows: dict[str, dict[str, str]] = {}
        versions: set[str] = set()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
            for record in batch.to_pylist():
                novel_id = str(record["novel_id"])
                blurb, opening = "", ""
                if record.get("sections_json"):
                    sections = json.loads(str(record["sections_json"]))
                    blurb = next((str(s["text"]) for s in sections if s.get("kind") == "blurb"), "")
                    opening = "\n\n".join(str(s["text"]) for s in sections if s.get("kind") == "opening")
                profile = (opening or str(record.get("profile_text") or ""))[:keep_chars]
                rows[novel_id] = {"title": str(record.get("title_guess") or ""), "profile": profile, "blurb": blurb, "card": card_texts.get(novel_id, "")}
                if record.get("digest_version"):
                    versions.add(str(record["digest_version"]))
        return cls(rows, digest_version=next(iter(versions)) if len(versions) == 1 else None, cards=sum(1 for r in rows.values() if r["card"]))

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

    def reset_memory(self, memory_path: Path) -> None:
        """Point the agent at a fresh memory file (one per evaluation task) without reloading anything else."""

        from src.agent.tools import build_memory_tools

        self.memory = UserMemory.load(memory_path)
        self.memory_path = memory_path
        self.turn = 0
        for spec in build_memory_tools(self.memory, turn_counter=lambda: self.turn):
            self.tools.specs[spec.name] = spec
        self.loop.memory = self.memory

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
    cards_path: Path | None = DEFAULT_CARDS_PATH,
    density_path: Path = DEFAULT_DENSITY_PATH,
    inventory_path: Path = DEFAULT_OUTPUT_PATH,
    memory_path: Path = DEFAULT_MEMORY_PATH,
    device: str | None = None,
    embedding_dtype: str | None = None,
    trope_model: str | None = None,
    reasoning_effort: str | None = None,
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

    profiles = ParquetProfiles.load(profiles_path, cards_path=cards_path)
    densities = load_density_table(density_path)
    memory = UserMemory.load(memory_path)
    transport = HTTPChatTransport(model=model, base_url=base_url, reasoning_effort=reasoning_effort)
    trope_transport = HTTPChatTransport(model=trope_model or model, base_url=base_url, reasoning_effort=reasoning_effort)
    judge = CachedTropeJudge(trope_transport, LazyRawText(inventory_path), model_name=trope_model or model)

    bundle = AgentBundle(loop=None, tools=ToolRegistry(), memory=memory, memory_path=memory_path)  # type: ignore[arg-type]
    bundle.tools.register(build_search_books(searcher, budget, profiles))
    bundle.tools.register(build_get_profile(profiles, budget))
    bundle.tools.register(build_check_term(densities))
    bundle.tools.register(build_check_trope(judge))
    for spec in build_memory_tools(memory, turn_counter=lambda: bundle.turn):
        bundle.tools.register(spec)
    bundle.loop = AgentLoop(transport, bundle.tools, memory, config=config, model_name=model)
    bundle.info = {
        "model": model,
        "base_url": base_url,
        "reasoning_effort": reasoning_effort,
        "index_dir": index_dir.as_posix(),
        "searcher": searcher.name,
        "profiles": len(profiles),
        "cards": profiles.cards,
        "digest_version": profiles.digest_version,
        "index_digest_version": metadata.get("digest_version"),
        "density_rows": len(densities),
    }
    if metadata.get("digest_version") and profiles.digest_version and metadata["digest_version"] != profiles.digest_version:
        bundle.info["warning"] = f"index built from {metadata['digest_version']} but profiles are {profiles.digest_version}"
    return bundle
