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
from src.agent.session import SessionState
from src.agent.trajectory import text_fingerprint
from src.preferences import constraint_violation_from_densities, is_rule_checkable, merged_density_from_table
from src.retrieval.catalog import CardCatalog, CardFilter, FilterError, filter_options_text


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
        """Return ranked rows with novel_id, title_guess, profile_text_preview, score.

        Real searchers also accept ``allowed_ids`` (a set of novel_ids to search within) and the single-vector
        one has ``similar(novel_id, k, allowed_ids)``; the tools use those only when a filter is given."""


class PassageBackend(Protocol):
    def ask(self, novel_id: str, question: str, k: int = 3) -> dict[str, Any] | None:
        """{"coverage": str, "passages": [Passage], "source": str} or None when the book is unknown."""


class ProfileLookup(Protocol):
    def get(self, novel_id: str) -> dict[str, str] | None:
        """Return {"title": ..., "profile": ...} or None when unknown."""


class TropeJudge(Protocol):
    def judge(self, novel_id: str, trope: str) -> dict[str, Any]:
        """Return {"verdict": yes|no|unclear, "quotes": [...], "confidence": ...} or raise ToolError."""


# ---- tool builders --------------------------------------------------------------------------


def build_search_books(
    searcher: BookSearcher,
    budget: ContextBudget,
    profiles: ProfileLookup | None = None,
    catalog: CardCatalog | None = None,
    session: SessionState | None = None,
) -> ToolSpec:
    """``profiles`` lets the preview be the synopsis even when the index's stored preview is the raw profile head.
    ``catalog`` (the book cards) enables the genre / elements / style filters and the per-row 题材 label;
    ``session`` enables ``exclude_shown``."""

    from src.retrieval.multivector import synopsis_preview

    def preview_for(row: dict[str, Any]) -> str:
        if profiles is not None:
            full = profiles.get(str(row.get("novel_id", "")))
            if full and (full.get("blurb") or full.get("profile")):
                return synopsis_preview(full.get("blurb") or full["profile"], budget.preview_chars)
        return truncate(str(row.get("profile_text_preview", "")), budget.preview_chars)

    def handler(query: str, k: int = 10, genre: str | None = None, elements: list[str] | None = None, style: list[str] | None = None, exclude_shown: bool = False) -> dict[str, Any]:
        query = str(query).strip()
        if not query:
            raise ToolError("query 不能为空")
        k = int(k)
        if k < 1 or k > budget.max_search_k:
            raise ToolError(f"k 必须在 1 到 {budget.max_search_k} 之间")
        try:
            spec = CardFilter.parse(genre, elements, style)
        except FilterError as exc:
            raise ToolError(str(exc)) from exc
        allowed: set[str] | None = None
        if not spec.empty:
            if catalog is None:
                raise ToolError("这个书库没有书卡，不能按 genre / elements / style 过滤；去掉过滤条件，把特征写进 query。")
            allowed = catalog.select(spec)
            if not allowed:
                raise ToolError(f"没有书卡同时满足 {spec.describe()}；放宽条件再搜。")
        avoid = session.avoid() if (exclude_shown and session is not None) else set()
        if allowed is not None and avoid:
            allowed = allowed - avoid
            if not allowed:
                raise ToolError("排除已推荐的书后没有候选了；放宽过滤条件。")
        # The query reaches the embedder as the agent wrote it: no parsing, no term stripping (dropped 2026-10-11).
        if allowed is not None:
            rows = searcher.search(query, k, allowed_ids=allowed)
        else:
            rows = searcher.search(query, k + len(avoid))
            if avoid:
                rows = [row for row in rows if str(row.get("novel_id", "")) not in avoid][:k]
        out = []
        for row in rows:
            novel_id = str(row.get("novel_id", ""))
            item = {"novel_id": novel_id, "title": str(row.get("title_guess", "")), "preview": preview_for(row), "score": round(float(row.get("score", 0.0)), 4)}
            if catalog is not None:
                item["genre"] = catalog.label(novel_id)
            out.append(item)
        result: dict[str, Any] = {"results": out}
        if not spec.empty:
            result["filter"] = spec.describe()
            result["filter_matches"] = len(allowed or ())
        if avoid:
            result["excluded"] = len(avoid)
        return result

    def redact(result: Any) -> Any:
        if isinstance(result, dict) and isinstance(result.get("results"), list):
            return {**result, "results": [{**row, "preview": text_fingerprint(row.get("preview", ""))} for row in result["results"]]}
        return result

    filters = filter_options_text() if catalog is not None else "这个书库没有书卡，genre / elements / style 不可用。"
    return ToolSpec(
        name="search_books",
        description=(
            "按描述在书库里找书，返回候选的 novel_id、书名、题材、简介片段和相似度。query 是对想要的书的正面描述，用自然语言写清题材、"
            "设定、主角、节奏、基调即可，不要写否定（「不要系统」检索不认，负向条件要在拿到候选后用 check_term / check_trope 核查）。"
            "genre / elements / style 是按书卡做的硬过滤，用户点名了题材、元素或风格时优先用它们而不是靠措辞；没有书卡的书会被过滤掉。"
            "exclude_shown=true 跳过本次对话已推荐和用户已排除的书（用户说「换几本」时用）。换不同描述可以多搜几次。" + filters
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "想找的书的描述，自然语言，例如：主角是凡人出身的修仙故事，节奏慢，重宗门生活和炼丹"},
                "k": {"type": "integer", "description": f"返回条数，1 到 {budget.max_search_k}", "default": 10},
                "genre": {"type": "string", "description": "题材一级或二级，例如 仙侠 / 幻想修仙"},
                "elements": {"type": "array", "items": {"type": "string"}, "description": "必须具备的元素，例如 [\"系统\", \"穿越\"]"},
                "style": {"type": "array", "items": {"type": "string"}, "description": "风格档位，写「维度=档」，例如 [\"爽度=低\", \"感情线=无\"]"},
                "exclude_shown": {"type": "boolean", "default": False},
            },
            "required": ["query"],
        },
        handler=handler,
        redact=redact,
    )


def build_similar_books(searcher: Any, budget: ContextBudget, profiles: ProfileLookup | None = None, catalog: CardCatalog | None = None, session: SessionState | None = None) -> ToolSpec | None:
    """「类似某本书的」: the anchor book's own vector against the index. None when the searcher cannot do it."""

    if not hasattr(searcher, "similar"):
        return None
    from src.retrieval.multivector import synopsis_preview

    def handler(novel_id: str, k: int = 10, exclude_shown: bool = False) -> dict[str, Any]:
        novel_id = str(novel_id).strip()
        k = int(k)
        if k < 1 or k > budget.max_search_k:
            raise ToolError(f"k 必须在 1 到 {budget.max_search_k} 之间")
        avoid = session.avoid() if (exclude_shown and session is not None) else set()
        rows = searcher.similar(novel_id, k + len(avoid))
        if avoid:
            rows = [row for row in rows if str(row.get("novel_id", "")) not in avoid][:k]
        if not rows and (profiles is None or profiles.get(novel_id) is None):
            raise ToolError(f"没有这本书：{novel_id}。先用 search_books 按书名找到它的 novel_id。")
        out = []
        for row in rows:
            nid = str(row.get("novel_id", ""))
            full = profiles.get(nid) if profiles is not None else None
            preview = synopsis_preview(full.get("blurb") or full.get("profile", ""), budget.preview_chars) if full else truncate(str(row.get("profile_text_preview", "")), budget.preview_chars)
            item = {"novel_id": nid, "title": str(row.get("title_guess", "")), "preview": preview, "score": round(float(row.get("score", 0.0)), 4)}
            if catalog is not None:
                item["genre"] = catalog.label(nid)
            out.append(item)
        return {"anchor": novel_id, "results": out}

    def redact(result: Any) -> Any:
        if isinstance(result, dict) and isinstance(result.get("results"), list):
            return {**result, "results": [{**row, "preview": text_fingerprint(row.get("preview", ""))} for row in result["results"]]}
        return result

    return ToolSpec(
        name="similar_books",
        description="找和某一本书整体最相近的书（题材、设定、写法），用于「类似某某的」或「和刚才那本差不多的」。要先有那本书的 novel_id：用户报书名时先 search_books 书名。相近只是整体相似，用户的具体约束仍要核查。",
        parameters={
            "type": "object",
            "properties": {
                "novel_id": {"type": "string"},
                "k": {"type": "integer", "default": 10},
                "exclude_shown": {"type": "boolean", "default": False},
            },
            "required": ["novel_id"],
        },
        handler=handler,
        redact=redact,
    )


def build_ask_book(passages: PassageBackend, budget: ContextBudget, profiles: ProfileLookup | None = None) -> ToolSpec:
    def handler(novel_id: str, question: str, k: int = 3) -> dict[str, Any]:
        novel_id = str(novel_id).strip()
        question = str(question).strip()
        if not question:
            raise ToolError("question 不能为空")
        k = int(k)
        if k < 1 or k > budget.max_passages:
            raise ToolError(f"k 必须在 1 到 {budget.max_passages} 之间")
        found = passages.ask(novel_id, question, k)
        if found is None:
            raise ToolError(f"没有这本书：{novel_id}")
        title = ""
        if profiles is not None:
            row = profiles.get(novel_id)
            title = row.get("title", "") if row else ""
        return {
            "novel_id": novel_id,
            "title": title,
            "coverage": found["coverage"],
            "passages": [
                {"chapter": p.heading, "kind": p.kind, "text": truncate(p.text, budget.passage_chars), "score": round(p.score, 4)} for p in found["passages"]
            ],
            "note": "只检索了 coverage 里列出的章节；如果这些段落答不了问题，就告诉用户摘要未收录相关章节，不要推测。",
        }

    def redact(result: Any) -> Any:
        if isinstance(result, dict) and isinstance(result.get("passages"), list):
            return {**result, "passages": [{**p, "text": text_fingerprint(str(p.get("text", "")))} for p in result["passages"]]}
        return result

    return ToolSpec(
        name="ask_book",
        description=(
            "在一本书的摘要章节里找和问题相关的段落（返回章节标题和原文片段），用来回答关于这本书情节、设定、主角、结局的问题。"
            "能查到的只有这本书的开头几章、中段几章、结尾几章和章节目录（结果里的 coverage 会列出），不是全书：开头类问题（设定、金手指、主角来路）和结局类问题通常答得了，"
            "中段具体情节多半答不了，答不了就如实说摘要未收录。回答时在 finish 的 citations 里填用到的 novel_id 和 chapter。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "novel_id": {"type": "string"},
                "question": {"type": "string", "description": "要在书里找什么，自然语言"},
                "k": {"type": "integer", "description": f"返回段落数，1 到 {budget.max_passages}", "default": 3},
            },
            "required": ["novel_id", "question"],
        },
        handler=handler,
        redact=redact,
    )


def build_set_aside(session: SessionState, profiles: ProfileLookup | None = None) -> ToolSpec:
    def handler(novel_ids: list[str], reason: str = "") -> dict[str, Any]:
        if not isinstance(novel_ids, list) or not novel_ids:
            raise ToolError("novel_ids 必须是非空列表")
        added = session.exclude(novel_ids)
        return {"excluded": added, "total_excluded": len(session.excluded), "note": "之后 search_books / similar_books 用 exclude_shown=true 就不会再返回这些书。长期不想看的类型请另用 memory_write。"}

    return ToolSpec(
        name="set_aside",
        description="把用户本次对话里不要的书记下来（看过了、没兴趣、刚推荐的不满意），之后带 exclude_shown=true 搜索会跳过它们。只对本次对话有效。",
        parameters={
            "type": "object",
            "properties": {
                "novel_ids": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string", "description": "用户的原话或原因，可省略"},
            },
            "required": ["novel_ids"],
        },
        handler=handler,
    )


def build_get_profile(profiles: ProfileLookup, budget: ContextBudget) -> ToolSpec:
    """Since 2026-10-10 the answer is the book card first (题材 / 元素 / 风格 / 主角 / 一句话, built offline
    from the whole digest), then the author's synopsis, then an opening excerpt. The digest's first
    1,200 characters alone were the header and a sliver of chapter one once digest_v2 made the opening
    sections whole chapters."""

    from src.retrieval.multivector import synopsis_preview

    def handler(novel_id: str) -> dict[str, Any]:
        row = profiles.get(str(novel_id))
        if row is None:
            raise ToolError(f"没有这本书：{novel_id}")
        result: dict[str, Any] = {"novel_id": str(novel_id), "title": row.get("title", "")}
        if row.get("card"):
            result["card"] = row["card"]
        if row.get("blurb"):
            result["blurb"] = synopsis_preview(row["blurb"], budget.blurb_chars)
        result["opening"] = truncate(row.get("profile", ""), budget.profile_chars)
        return result

    def redact(result: Any) -> Any:
        # The card is derived text and may stay; the synopsis and the excerpt are corpus text.
        redacted = {**result, "opening": text_fingerprint(result.get("opening", ""))}
        if "blurb" in result:
            redacted["blurb"] = text_fingerprint(result["blurb"])
        return redacted

    return ToolSpec(
        name="get_profile",
        description="读一本书的档案：card 是离线建好的书卡（题材、元素、风格五维、主角、背景、一句话），blurb 是作者简介，opening 是开头摘录。用来判断题材、主角、感情线、爽度和是否含有某种元素；没有 card 的书只能看 blurb 和 opening。",
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
        description="判断一本书是否属于某个题材或风格标签（后宫、种马、爽文、圣母、无脑等）。先查离线书卡（source=card，题材 / 元素 / 风格五维能直接回答的），书卡没说的再从全书采样证据由模型判断（source=text，附引文）。结论是 yes、no 或 unclear。",
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
