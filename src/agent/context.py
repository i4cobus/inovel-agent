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


@dataclass(frozen=True)
class ContextBudget:
    preview_chars: int = 300
    profile_chars: int = 1200
    passage_chars: int = 800
    max_passages: int = 5
    memory_chars: int = 500
    tool_result_chars: int = 6000
    max_search_k: int = 20


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


SYSTEM_PROMPT = """你是一个中文网文推荐与问答助手。语料是一个私有的网文库，你只能通过工具了解其中的书，不能凭记忆编造任何书的内容、作者、评分或完结情况。

工作方式：
1. 推荐书时先用 search_books 检索，必要时换不同措辞多搜几次。
2. 对每一本准备推荐的书，用户的每条负向约束都要核查：文中会出现的词（系统、异能、僵尸等）用 check_term；题材或风格类标签（后宫、爽文、圣母、无脑等）用 check_trope。check_term 返回 null 表示这个词不能用规则判，要改用 check_trope 或读 get_profile 自己判断。check_trope 返回 unclear 时要在回答里说明不确定。
3. 要了解一本书的细节用 get_profile；回答关于某本书情节的问题用 ask_book，并引用它返回的章节。
4. 用户表达长期偏好（"以后都""我一直"）时用 memory_write 写入 persistent=true；只对本次有效的要求（"这次""今天"）写 persistent=false。写入后在回答里明确说"已记住……"。
5. 不要重复调用同一个工具同样的参数。信息够了就直接回答。

最终回答用自然语言，并在末尾附一个 JSON 对象（单独一行），格式：
{"recommendations": [{"novel_id": "...", "title": "...", "reason": "..."}], "citations": [{"novel_id": "...", "chapter_idx": 0}]}
没有推荐就给空列表，没有引用就给空列表。"""


def build_system_prompt(memory: UserMemory | None, budget: ContextBudget) -> str:
    prompt = SYSTEM_PROMPT
    if memory is not None:
        summary = memory.summary(max_chars=budget.memory_chars)
        if summary:
            prompt += f"\n\n关于这位用户你已经知道的（来自长期记忆）：\n{summary}"
    return prompt
