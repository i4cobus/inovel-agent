"""Single-user long-term memory, persisted as one JSON file.

Two kinds of entry share one store. Persistent entries survive across sessions;
session entries ("这次不要系统") are kept for the current run only and dropped on
save. The distinction is the agent's call from the user's wording, and the
evaluation has a task class that checks it is made correctly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.config import DATA_DIR

DEFAULT_MEMORY_PATH = DATA_DIR / "memory" / "user.json"
MEMORY_KINDS = ("positive", "negative", "read", "note")
MemoryKind = Literal["positive", "negative", "read", "note"]

KIND_LABELS = {"positive": "偏好", "negative": "不要", "read": "已读", "note": "备注"}


@dataclass(frozen=True)
class MemoryEntry:
    kind: str
    value: str
    persistent: bool
    turn: int
    written_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "persistent": self.persistent,
            "turn": self.turn,
            "written_at": self.written_at,
        }


@dataclass
class UserMemory:
    entries: list[MemoryEntry] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = DEFAULT_MEMORY_PATH) -> "UserMemory":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = [
            MemoryEntry(
                kind=str(item["kind"]),
                value=str(item["value"]),
                persistent=True,
                turn=int(item.get("turn", 0)),
                written_at=str(item.get("written_at", "")),
            )
            for item in data.get("entries", [])
            if str(item.get("kind")) in MEMORY_KINDS
        ]
        return cls(entries=entries)

    def save(self, path: Path = DEFAULT_MEMORY_PATH) -> None:
        """Persist the persistent entries only; session entries are dropped by design."""

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": [entry.to_dict() for entry in self.entries if entry.persistent]}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def write(self, kind: str, value: str, persistent: bool, turn: int = 0) -> MemoryEntry:
        if kind not in MEMORY_KINDS:
            raise ValueError(f"kind must be one of {MEMORY_KINDS}, got {kind!r}")
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        # The same value written again moves to the end so "latest wins" is visible
        # in the ordering, and a later persistent write upgrades a session entry.
        self.entries = [entry for entry in self.entries if not (entry.kind == kind and entry.value == value)]
        entry = MemoryEntry(
            kind=kind,
            value=value,
            persistent=persistent,
            turn=turn,
            written_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.entries.append(entry)
        return entry

    def forget(self, kind: str, value: str) -> int:
        before = len(self.entries)
        self.entries = [entry for entry in self.entries if not (entry.kind == kind and entry.value == value.strip())]
        return before - len(self.entries)

    def values(self, kind: str) -> list[str]:
        return [entry.value for entry in self.entries if entry.kind == kind]

    def to_dict(self) -> dict[str, Any]:
        return {kind: [entry.to_dict() for entry in self.entries if entry.kind == kind] for kind in MEMORY_KINDS}

    def summary(self, max_chars: int = 500) -> str:
        """Compact rendering for the system prompt; latest entries win the budget."""

        lines: list[str] = []
        for kind in MEMORY_KINDS:
            values = self.values(kind)
            if values:
                lines.append(f"{KIND_LABELS[kind]}：{'、'.join(values)}")
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]
