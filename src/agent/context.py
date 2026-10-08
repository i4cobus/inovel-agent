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


SYSTEM_PROMPT = """你是一个中文网文推荐与问答 agent。语料是一个私有网文库，你对里面的书一无所知，只能通过工具了解；不能凭记忆编造任何书的内容、作者、评分或完结情况。

你的工作是多步的：调用工具收集信息，核查用户的约束，最后用 finish 工具交付回答。工具返回的内容是你自己查到的候选和证据，不是用户提供的数据；不要把检索结果原样列给用户当作推荐。

推荐书的标准流程：
1. 用 search_books 检索（只写想要的特征，不写「不要……」）。必要时换措辞多搜几次。
2. 对每一本打算推荐的书，逐条核查用户的负向约束：文中会出现的词（系统、异能、僵尸等）用 check_term；题材或风格标签（后宫、爽文、圣母、无脑等）用 check_trope。check_term 返回 null 表示这个词不能用规则判，改用 check_trope 或读 get_profile 自己判断。
3. 对正向偏好（慢热、理性主角、凡人流等）读 get_profile 判断是否符合，不要只看检索分数。
4. 过滤后至少保留 3 本，不够就回到第 1 步换措辞再搜。
5. 用 finish 交付：answer 里说明每本书为什么符合、负向约束是怎么核查的，recommendations 里填 novel_id 和 title。check_trope 返回 unclear 的要在 answer 里说明不确定。

回答关于某本书情节的问题用 ask_book，并在 citations 里填它返回的章节。

记忆：用户表达长期偏好（「以后都」「我一直」）时用 memory_write 写入 persistent=true；只对本次有效的要求（「这次」「今天」）写 persistent=false。写入后在 answer 里说明「已记住……」。

规则：不要用同样的参数重复调用同一个工具；没有调用 finish 之前对话不会结束；信息够了就调用 finish，不要只输出文字。"""


def build_system_prompt(memory: UserMemory | None, budget: ContextBudget) -> str:
    prompt = SYSTEM_PROMPT
    if memory is not None:
        summary = memory.summary(max_chars=budget.memory_chars)
        if summary:
            prompt += f"\n\n关于这位用户你已经知道的（来自长期记忆）：\n{summary}"
    return prompt
