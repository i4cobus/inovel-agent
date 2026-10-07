"""Extract JSON objects from LLM output that may carry a reasoning trace."""

from __future__ import annotations

import json
import re
from typing import Any


def split_first_json_object(text: str) -> tuple[str, str]:
    """Return the first balanced JSON object and whatever follows it.

    A greedy ``\{.*\}`` spans from the first brace anywhere in the output to the
    last one, so a single brace inside a reasoning model's ``<think>`` block
    swallows the real answer. Reasoning traces are stripped first, then braces are
    matched by depth so the first complete object wins.

    The trailing remainder is returned because a rollout that answers correctly and
    then keeps generating is a distinct, separately-scored failure: the verdict is
    fine, the termination is not. One depth matcher serves both readings so they
    cannot disagree about where the object ends.
    """

    body = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    body = re.sub(r"^.*?</think>", "", body, flags=re.S)  # truncated trace, no opener
    depth = 0
    start = -1
    for index, char in enumerate(body):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                return body[start : index + 1], body[index + 1 :]
    raise ValueError("No JSON object found in LLM output")


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from generated text."""

    return json.loads(split_first_json_object(text)[0])
