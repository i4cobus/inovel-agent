"""Context budget and the system prompt.

Every number here is a configuration value the evaluation holds fixed. Tool
results are truncated to a budget before they enter the conversation; the
model never sees a whole profile or an unbounded candidate list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.agent.memory import UserMemory
from src.agent.session import SessionState


@dataclass(frozen=True)
class ContextBudget:
    preview_chars: int = 300
    profile_chars: int = 1200  # opening excerpt returned by get_profile
    blurb_chars: int = 500  # author synopsis returned by get_profile
    passage_chars: int = 800
    max_passages: int = 5
    memory_chars: int = 500
    tool_result_chars: int = 6000
    max_search_k: int = 20
    # Tool results older than this many tool messages are compacted to ids and titles
    # before the next model call. The trajectory keeps the full observation.
    keep_recent_tool_results: int = 6


def truncate(text: str, limit: int, marker: str = "…[截断]") -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(limit - len(marker), 0)] + marker


def render_tool_result(result: Any, budget: ContextBudget) -> str:
    """Serialise a tool result for the model and cap its size."""

    if isinstance(result, str):
        text = result
    else:
        text = json.dumps(result, ensure_ascii=False)
    return truncate(text, budget.tool_result_chars)


def compact_tool_message(content: str, tool_name: str) -> str:
    """A short stand-in for an old tool result: ids and titles survive, text does not.

    The fourth real trajectory (chat-smoke-21, 2026-10-08) ran 10 steps and
    101k prompt tokens because every search result and profile stayed verbatim
    in the conversation. Older results are reduced so the model can still refer
    to a novel_id it saw, but no longer re-reads 10 previews per step.
    """

    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return truncate(content, 200)
    if tool_name in ("search_books", "similar_books"):
        rows_data = data.get("results") if isinstance(data, dict) else data
        if isinstance(rows_data, list):
            rows = [f"{row.get('novel_id', '')}|{row.get('title', '')}" for row in rows_data if isinstance(row, dict)]
            return "[已压缩的检索结果，novel_id|书名] " + "; ".join(rows)
    if tool_name == "ask_book" and isinstance(data, dict):
        chapters = "、".join(str(p.get("chapter", "")) for p in data.get("passages", []) if isinstance(p, dict))
        return f"[已读段落] {data.get('novel_id', '')}|{data.get('title', '')}: {chapters}"
    if tool_name == "get_profile" and isinstance(data, dict):
        gist = data.get("card") or data.get("opening") or data.get("profile") or ""
        return f"[已读档案] {data.get('novel_id', '')}|{data.get('title', '')}: {truncate(str(gist), 150)}"
    if isinstance(data, dict) and "quotes" in data:
        return json.dumps({k: v for k, v in data.items() if k != "quotes"}, ensure_ascii=False)
    return truncate(content, 400)


def compact_messages(messages: list[dict[str, Any]], tool_names: dict[str, str], keep_recent: int) -> list[dict[str, Any]]:
    """Return a copy where all but the last ``keep_recent`` tool messages are compacted."""

    tool_positions = [index for index, message in enumerate(messages) if message.get("role") == "tool"]
    to_compact = set(tool_positions[:-keep_recent]) if keep_recent > 0 else set(tool_positions)
    output: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if index in to_compact:
            name = tool_names.get(str(message.get("tool_call_id")), "")
            content = str(message.get("content", ""))
            if not content.startswith("[已压缩") and not content.startswith("[已读"):
                message = {**message, "content": compact_tool_message(content, name)}
        output.append(message)
    return output


SYSTEM_PROMPT = """你是一个中文网文推荐与问答 agent，面向一个私有网文库。你对库里的书一无所知，只能通过工具了解；不能凭记忆编造任何书的内容、作者、评分、热度或完结情况。用户用自然语言和你对话，可能一句话里既有要求也有闲聊，先弄清他要什么再动手。

你能做的事：
- 找书：按描述、按题材元素风格、或「类似某本」找候选，核查用户的条件后推荐。
- 鉴定一本书：题材、元素、风格五维、主角、篇幅、章节数、结局标记、有没有某种元素或套路。
- 回答一本书的情节问题：只能依据这本书摘要收录的章节（开头几章、中段几章、结尾几章和章节目录），答不了就说明摘要未收录。
- 记住用户的长期偏好和已读书目，在后续对话里遵守。
你做不了的事要直说：实时信息（更新、作者动态）、评分和热度、全书逐章概括、库外的书。

工具怎么配合：
- search_books：query 用自然语言正面描述想要的书；用户点名题材、元素或风格时用 genre / elements / style 过滤，别靠措辞赌；用户说「换几本」时带 exclude_shown=true。
- similar_books：「类似某某」先用 search_books 搜书名拿到 novel_id，再找相近的书。
- get_profile：读书卡、简介和开头摘录，用来判断正向偏好（慢热、理性主角、凡人流等）和基本信息。
- check_term：文中会出现的词（系统、异能、僵尸等）用词频规则判；返回 null 表示规则判不了。
- check_trope：题材和风格标签（后宫、爽文、圣母、无脑等）先查书卡再读原文判断；unclear 要如实告诉用户。
- ask_book：针对某本书的问题，返回相关段落和章节标题；引用过的段落在 finish 的 citations 里填 novel_id 和 chapter。
- set_aside：用户说看过了、不要这本、刚才那几本不满意，就记下来。
- memory_write：长期偏好（「以后都」「我一直」）persistent=true；只对本次有效的（「这次」「今天」）persistent=false。写入后在回答里说明「已记住……」。

推荐的流程：搜索 → 对每本打算推荐的书逐条核查用户的负向条件（文中词用 check_term，标签用 check_trope）→ 正向偏好读 get_profile 判断，不要只看检索分数 → 至少 3 本通过就用 finish 交付，不够就换描述或换过滤条件再搜。answer 里说明每本为什么符合、负向条件怎么核查的；recommendations 里填 novel_id 和 title。

用户的需求太模糊、无法开始找书时（比如只说「推荐点好看的」而记忆里也没有偏好），用 finish 提一个具体的问题，asks_user=true，不要瞎猜。追问（「第二本再讲讲」「这本有后宫吗」）指的是本次对话已推荐的书，编号见下方会话状态。

规则：工具返回的是你自己查到的候选和证据，不是用户给的数据，不要原样罗列当作推荐；不要用同样的参数重复调用同一个工具；没有调用 finish 之前对话不会结束，信息够了就调用 finish，不要只输出文字。步数有限（系统会提示剩余步数）：有 3 本通过核查的书就交付，宁可在 answer 里说明哪些偏好没能完全确认，也不要把步数耗尽。"""


def build_system_prompt(memory: UserMemory | None, budget: ContextBudget, session: SessionState | None = None) -> str:
    prompt = SYSTEM_PROMPT
    if memory is not None:
        summary = memory.summary(max_chars=budget.memory_chars)
        if summary:
            prompt += f"\n\n关于这位用户你已经知道的（来自长期记忆）：\n{summary}"
    if session is not None:
        state = session.summary()
        if state:
            prompt += f"\n\n会话状态：\n{state}"
    return prompt
