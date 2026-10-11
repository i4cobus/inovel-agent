"""What one conversation has already put in front of the user.

Long-term memory (``memory.py``) is about the person; this is about the session: which books were
recommended, in what order, and which the user has set aside. It lets 「换几本」 exclude what was
shown, 「第二本」 resolve to a novel_id, and a follow-up question know which book it is about. The
loop updates it from tool results and from ``finish``; the system prompt renders it each turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionState:
    recommended: list[dict[str, str]] = field(default_factory=list)  # ordered, deduplicated: novel_id, title
    shown: dict[str, str] = field(default_factory=dict)  # every novel_id a tool result carried -> title
    excluded: set[str] = field(default_factory=set)  # user set aside (read already, not interested)
    turn: int = 0

    def note_rows(self, rows: Any) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if isinstance(row, dict) and row.get("novel_id"):
                self.shown.setdefault(str(row["novel_id"]), str(row.get("title") or row.get("title_guess") or ""))

    def note_recommendations(self, recommendations: Any) -> None:
        for rec in recommendations or []:
            if not isinstance(rec, dict) or not rec.get("novel_id"):
                continue
            novel_id = str(rec["novel_id"])
            if all(r["novel_id"] != novel_id for r in self.recommended):
                self.recommended.append({"novel_id": novel_id, "title": str(rec.get("title") or self.shown.get(novel_id, ""))})

    def exclude(self, novel_ids: Any) -> list[str]:
        added = []
        for novel_id in novel_ids or []:
            novel_id = str(novel_id).strip()
            if novel_id and novel_id not in self.excluded:
                self.excluded.add(novel_id)
                added.append(novel_id)
        return added

    def avoid(self) -> set[str]:
        """What a fresh search should skip when the user asks for different books."""

        return {r["novel_id"] for r in self.recommended} | self.excluded

    def summary(self, max_items: int = 12) -> str:
        if not self.recommended and not self.excluded:
            return ""
        lines = []
        if self.recommended:
            items = self.recommended[-max_items:]
            lines.append("本次对话已推荐（按顺序编号，用户说「第二本」指的就是这里的第 2 本）：" + "；".join(f"{i}. {r['title']}（{r['novel_id']}）" for i, r in enumerate(items, start=1)))
        if self.excluded:
            lines.append("用户已排除（不要再推荐）：" + "、".join(self.shown.get(n, n) for n in sorted(self.excluded)))
        return "\n".join(lines)
