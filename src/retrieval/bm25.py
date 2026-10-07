"""A small BM25 over jieba tokens.

Implemented here rather than imported because the index must round-trip to a
plain JSON file on Windows, stay inspectable, and later serve per-book passage
indexes where a dependency's global state would get in the way. Okapi BM25
with the standard k1 / b; nothing clever.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

KEEP_RE = re.compile(r"[\w一-鿿]")


def tokenize(text: str) -> list[str]:
    """jieba tokens, lower-cased, with pure punctuation and whitespace dropped."""

    import jieba

    jieba.setLogLevel(40)
    return [token.lower() for token in jieba.lcut(text) if KEEP_RE.search(token)]


Tokenizer = Callable[[str], list[str]]


@dataclass
class BM25Index:
    doc_ids: list[str]
    doc_lengths: list[int]
    postings: dict[str, list[tuple[int, int]]]
    k1: float = 1.5
    b: float = 0.75
    avg_length: float = field(init=False)

    def __post_init__(self) -> None:
        self.avg_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0

    @classmethod
    def build(cls, docs: Iterable[str], ids: Iterable[str], tokenizer: Tokenizer = tokenize, k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        doc_ids = list(ids)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths: list[int] = []
        for doc_index, text in enumerate(docs):
            tokens = tokenizer(text)
            lengths.append(len(tokens))
            counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            for token, count in counts.items():
                postings[token].append((doc_index, count))
        if len(lengths) != len(doc_ids):
            raise ValueError("docs and ids must have the same length")
        return cls(doc_ids=doc_ids, doc_lengths=lengths, postings=dict(postings), k1=k1, b=b)

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def idf(self, token: str) -> float:
        df = len(self.postings.get(token, ()))
        return math.log(1 + (self.size - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int, tokenizer: Tokenizer = tokenize) -> list[tuple[str, float]]:
        """Top-k (doc_id, score); documents sharing no token with the query are absent."""

        if k <= 0 or not self.doc_ids:
            return []
        scores: dict[int, float] = defaultdict(float)
        for token in set(tokenizer(query)):
            plist = self.postings.get(token)
            if not plist:
                continue
            idf = self.idf(token)
            for doc_index, tf in plist:
                norm = self.k1 * (1 - self.b + self.b * self.doc_lengths[doc_index] / (self.avg_length or 1.0))
                scores[doc_index] += idf * tf * (self.k1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [(self.doc_ids[doc_index], score) for doc_index, score in ranked]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "doc_lengths": self.doc_lengths,
            "postings": {token: [[d, c] for d, c in plist] for token, plist in self.postings.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            doc_ids=[str(item) for item in data["doc_ids"]],
            doc_lengths=[int(item) for item in data["doc_lengths"]],
            postings={token: [(int(d), int(c)) for d, c in plist] for token, plist in data["postings"].items()},
            k1=float(data.get("k1", 1.5)),
            b=float(data.get("b", 0.75)),
        )
