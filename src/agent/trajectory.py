"""Structured trajectory records: one JSON object per run, one step per model turn.

Tool observations can contain corpus text. The full trajectory stays local; what
goes into git is the redacted form, where every tool spec's ``redact`` hook
replaces text with a hash and a length. Every metric must be computable from
the redacted form — that is a constraint on what the hooks keep.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


def text_fingerprint(text: str) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "length": len(text)}


@dataclass
class Observation:
    tool: str
    call_id: str
    arguments: dict[str, Any] | None
    result: Any = None
    error: str | None = None
    latency_s: float = 0.0


@dataclass
class Step:
    index: int
    assistant_content: str
    tool_calls: list[dict[str, Any]]
    observations: list[Observation]
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


@dataclass
class Trajectory:
    task_id: str
    model: str
    user_message: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    structured: dict[str, Any] | None = None
    termination: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def tool_call_count(self) -> int:
        return sum(len(step.tool_calls) for step in self.steps)

    @property
    def prompt_tokens(self) -> int:
        return sum(step.prompt_tokens for step in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(step.completion_tokens for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def redacted(self, redactors: dict[str, Callable[[Any], Any]]) -> dict[str, Any]:
        """Copy with every observation passed through its tool's redactor.

        A tool without a redactor is assumed to return no corpus text. The final
        answer is fingerprinted rather than kept: it may quote passages verbatim.
        """

        data = self.to_dict()
        for step in data["steps"]:
            for observation in step["observations"]:
                redactor = redactors.get(observation["tool"])
                if redactor is not None and observation["result"] is not None:
                    observation["result"] = redactor(observation["result"])
        data["final_answer"] = text_fingerprint(self.final_answer)
        return data


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
