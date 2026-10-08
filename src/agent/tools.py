"""Tool specs and the registry the loop dispatches through.

A tool is a JSON-schema description plus a handler. Handlers take already
validated keyword arguments and return JSON-serialisable data; they raise
``ToolError`` for a user-facing failure (unknown novel, bad argument), which the
loop hands back to the model as an observation rather than crashing the run.

Backends (the index, the profile table, the density table, the trope judge)
are injected so tests run on stubs. ``redact`` hooks mark which results carry
corpus text and how to strip it for the committed trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from src.agent.context import ContextBudget, truncate
from src.agent.memory import MEMORY_KINDS, UserMemory
from src.agent.trajectory import text_fingerprint
from src.preferences import constraint_violation_from_densities, is_rule_checkable, merged_density_from_table
from src.retrieval.query import retrieval_query


class ToolError(Exception):
    """A failure the model should see and recover from."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    redact: Callable[[Any], Any] | None = None

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


@dataclass
class ToolRegistry:
    specs: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self.specs:
            raise ValueError(f"Tool already registered: {spec.name}")
        self.specs[spec.name] = spec

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.openai_schema() for spec in self.specs.values()]

    def redactors(self) -> dict[str, Callable[[Any], Any]]:
        return {name: spec.redact for name, spec in self.specs.items() if spec.redact is not None}

    def call(self, name: str, arguments: Mapping[str, Any]) -> Any:
        spec = self.specs.get(name)
        if spec is None:
            raise ToolError(f"未知工具：{name}。可用工具：{', '.join(self.specs)}")
        required = spec.parameters.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing:
            raise ToolError(f"{name} 缺少参数：{', '.join(missing)}")
        allowed = set(spec.parameters.get("properties", {}))
        unknown = [key for key in arguments if key not in allowed]
        if unknown:
            raise ToolError(f"{name} 不认识的参数：{', '.join(unknown)}")
        return spec.handler(**arguments)


# ---- backends the tools are built on --------------------------------------------------------


class BookSearcher(Protocol):
    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Return ranked rows with novel_id, title_guess, profile_text_preview, score."""


class ProfileLookup(Protocol):
    def get(self, novel_id: str) -> dict[str, str] | None:
        """Return {"title": ..., "profile": ...} or None when unknown."""


class TropeJudge(Protocol):
    def judge(self, novel_id: str, trope: str) -> dict[str, Any]:
        """Return {"verdict": yes|no|unclear, "quotes": [...], "confidence": ...} or raise ToolError."""


# ---- tool builders --------------------------------------------------------------------------


def build_search_books(searcher: BookSearcher, budget: ContextBudget) -> ToolSpec:
    def handler(query: str, k: int = 10) -> list[dict[str, Any]]:
        query = str(query).strip()
        if not query:
            raise ToolError("query 不能为空")
        k = int(k)
        if k < 1 or k > budget.max_search_k:
            raise ToolError(f"k 必须在 1 到 {budget.max_search_k} 之间")
        # Negatives never reach the embedder; the agent enforces them with check_term / check_trope.
        rows = searcher.search(retrieval_query(query), k)
        return [
            {
                "novel_id": str(row.get("novel_id", "")),
                "title": str(row.get("title_guess", "")),
                "preview": truncate(str(row.get("profile_text_preview", "")), budget.preview_chars),
                "score": round(float(row.get("score", 0.0)), 4),
            }
            for row in rows
        ]

    def redact(result: Any) -> Any:
        return [{**row, "preview": text_fingerprint(row.get("preview", ""))} for row in result]

    return ToolSpec(
        name="search_books",
        description="按描述检索书库，返回候选书的 novel_id、书名、简介片段和相似度。只写想要的特征，不要写「不要……」：负向约束检索不认，要用 check_term / check_trope 核查。换不同措辞可以多搜几次。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "想找的书的描述，例如：凡人流 仙侠 慢热 宗门"},
                "k": {"type": "integer", "description": f"返回条数，1 到 {budget.max_search_k}", "default": 10},
            },
            "required": ["query"],
        },
        handler=handler,
        redact=redact,
    )


def build_get_profile(profiles: ProfileLookup, budget: ContextBudget) -> ToolSpec:
    def handler(novel_id: str) -> dict[str, Any]:
        row = profiles.get(str(novel_id))
        if row is None:
            raise ToolError(f"没有这本书：{novel_id}")
        return {
            "novel_id": str(novel_id),
            "title": row.get("title", ""),
            "profile": truncate(row.get("profile", ""), budget.profile_chars),
        }

    def redact(result: Any) -> Any:
        return {**result, "profile": text_fingerprint(result.get("profile", ""))}

    return ToolSpec(
        name="get_profile",
        description="读一本书的档案：作者简介加若干章节摘录，用来判断题材、主角、节奏和是否含有某种元素。",
        parameters={
            "type": "object",
            "properties": {"novel_id": {"type": "string", "description": "search_books 返回的 novel_id"}},
            "required": ["novel_id"],
        },
        handler=handler,
        redact=redact,
    )


def build_check_term(densities_by_novel: Mapping[str, Mapping[str, float]]) -> ToolSpec:
    """The deterministic rule: term density per 100k characters, from the precomputed table."""

    def handler(novel_id: str, terms: list[str]) -> dict[str, Any]:
        novel_id = str(novel_id)
        if not isinstance(terms, list) or not terms:
            raise ToolError("terms 必须是非空列表")
        terms = [str(term).strip() for term in terms if str(term).strip()]
        densities = densities_by_novel.get(novel_id)
        if densities is None:
            raise ToolError(f"词频表里没有这本书：{novel_id}")
        checkable = [term for term in terms if is_rule_checkable(term)]
        not_checkable = [term for term in terms if not is_rule_checkable(term)]
        verdict = constraint_violation_from_densities(densities, checkable) if checkable else None
        return {
            "novel_id": novel_id,
            "violates": verdict,
            "evidence": {term: round(merged_density_from_table(densities, term), 3) for term in checkable},
            "not_checkable": not_checkable,
            "note": (
                "null 表示无法用规则判定：要么这些词不是文中词，要么密度落在灰区。请改用 check_trope 或读档案判断。"
                if verdict is None
                else ""
            ),
        }

    return ToolSpec(
        name="check_term",
        description="用词频规则核查一本书是否含有文中会出现的元素（系统、异能、僵尸、兽人等）。对题材或风格标签无效，会返回 null。",
        parameters={
            "type": "object",
            "properties": {
                "novel_id": {"type": "string"},
                "terms": {"type": "array", "items": {"type": "string"}, "description": "要排除的词，例如 [\"系统\"]"},
            },
            "required": ["novel_id", "terms"],
        },
        handler=handler,
    )


def build_check_trope(judge: TropeJudge) -> ToolSpec:
    def handler(novel_id: str, trope: str) -> dict[str, Any]:
        return judge.judge(str(novel_id), str(trope).strip())

    def redact(result: Any) -> Any:
        if isinstance(result, dict) and "quotes" in result:
            return {**result, "quotes": [text_fingerprint(str(quote)) for quote in result["quotes"]]}
        return result

    return ToolSpec(
        name="check_trope",
        description="判断一本书是否属于某个题材或风格标签（后宫、种马、爽文、圣母、无脑等）。从全书采样证据后由模型判断，结论是 yes、no 或 unclear，附引文。",
        parameters={
            "type": "object",
            "properties": {
                "novel_id": {"type": "string"},
                "trope": {"type": "string", "description": "一个标签，例如 后宫"},
            },
            "required": ["novel_id", "trope"],
        },
        handler=handler,
        redact=redact,
    )


def build_memory_tools(memory: UserMemory, turn_counter: Callable[[], int] = lambda: 0) -> list[ToolSpec]:
    def read() -> dict[str, Any]:
        return memory.to_dict()

    def write(kind: str, value: str, persistent: bool = True) -> dict[str, Any]:
        if kind not in MEMORY_KINDS:
            raise ToolError(f"kind 必须是 {', '.join(MEMORY_KINDS)} 之一")
        try:
            entry = memory.write(kind, str(value), bool(persistent), turn=turn_counter())
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return {"written": entry.to_dict()}

    read_spec = ToolSpec(
        name="memory_read",
        description="读取关于这位用户的长期记忆：偏好、不要的元素、已读的书、备注。",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=read,
    )
    write_spec = ToolSpec(
        name="memory_write",
        description="记住用户的一条信息。persistent=true 表示长期偏好，以后每次都要遵守；false 表示只对本次对话有效。",
        parameters={
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(MEMORY_KINDS), "description": "positive 喜欢 / negative 不要 / read 已读 / note 备注"},
                "value": {"type": "string"},
                "persistent": {"type": "boolean", "default": True},
            },
            "required": ["kind", "value"],
        },
        handler=write,
    )
    return [read_spec, write_spec]
