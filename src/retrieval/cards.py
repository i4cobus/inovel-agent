"""Offline book cards: one structured record per novel, extracted by a local model from its profile.

The corpus is raw text with no metadata, and users ask for abstract things
(凡人流、慢热、理性主角) that a vector over narrative excerpts does not match.
A card gives every book a short structured description to embed and a trope
field to filter on at retrieval time, which is what turns 「不要后宫」 from a
post-hoc check into a pre-filter (design doc D9).

Reliability is measured before use: ``validate_tropes`` compares the trope
field with the v1 judge's constraint labels (753 (trope, book) pairs).
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from src.agent.trope import TROPE_GLOSSARY
from src.chat_transport import ChatTransport
from src.config import DATA_DIR
from src.llm_json import extract_json_object

CARD_PROMPT_VERSION = "card_v1"
DEFAULT_CARDS_PATH = DATA_DIR / "processed" / "book_cards.parquet"
DEFAULT_CARD_CACHE_PATH = DATA_DIR / "cache" / "book_cards.jsonl"
GENRES = ("仙侠", "玄幻", "都市", "历史", "武侠", "科幻", "悬疑", "网游", "种田", "西幻", "言情", "灵异", "军事", "同人", "其他")
PACING = ("慢热", "中等", "快")
TROPE_VALUES = ("yes", "no", "unclear")


@dataclass
class BookCard:
    novel_id: str
    genre: str
    subgenres: list[str]
    tropes: dict[str, str]  # label -> yes | no | unclear
    protagonist: str
    setting: str
    pacing: str
    tone: str
    one_liner: str
    keywords: list[str]
    model: str = ""
    prompt_version: str = CARD_PROMPT_VERSION
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "novel_id": self.novel_id,
            "genre": self.genre,
            "subgenres": list(self.subgenres),
            "tropes": dict(self.tropes),
            "protagonist": self.protagonist,
            "setting": self.setting,
            "pacing": self.pacing,
            "tone": self.tone,
            "one_liner": self.one_liner,
            "keywords": list(self.keywords),
            "model": self.model,
            "prompt_version": self.prompt_version,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BookCard":
        return cls(**{k: data.get(k, "" if k not in ("subgenres", "keywords", "tropes") else ({} if k == "tropes" else [])) for k in cls.__dataclass_fields__})

    @property
    def yes_tropes(self) -> list[str]:
        return [t for t, v in self.tropes.items() if v == "yes"]

    def text(self) -> str:
        """What gets embedded as the card section."""

        parts = [f"题材：{self.genre}"]
        if self.subgenres:
            parts.append("子类：" + "、".join(self.subgenres))
        if self.protagonist:
            parts.append(f"主角：{self.protagonist}")
        if self.setting:
            parts.append(f"背景：{self.setting}")
        if self.pacing:
            parts.append(f"节奏：{self.pacing}")
        if self.tone:
            parts.append(f"基调：{self.tone}")
        if self.yes_tropes:
            parts.append("标签：" + "、".join(self.yes_tropes))
        if self.keywords:
            parts.append("关键词：" + "、".join(self.keywords))
        if self.one_liner:
            parts.append(f"一句话：{self.one_liner}")
        return "\n".join(parts)


# A digest runs to 12k characters (16k for the few books without chapter structure); the default
# sends it whole, about 10k tokens with the glossary. The old 4,000-character cut kept only the
# blurb and two opening chapters, dropping the chapter titles and the ending the digest exists for.
DEFAULT_CARD_MAX_CHARS = 16000


def build_card_prompt(profile_text: str, max_chars: int = DEFAULT_CARD_MAX_CHARS) -> str:
    glossary = "\n".join(f"- {label}：{definition}" for label, definition in TROPE_GLOSSARY.items())
    return (
        "下面是一本中文网文的档案：标题、简介、开头几章、采样的章节名、中段片段和结尾。请提炼一张结构化的书卡。只根据给出的文字判断，"
        "不要用你对这本书的任何先验知识。\n\n"
        f"题材只能从这些里选一个：{'、'.join(GENRES)}。\n"
        f"节奏只能是：{'、'.join(PACING)}。\n"
        "标签逐条判断，值只能是 yes / no / unclear：看到符合依据的内容判 yes，完全没有相关迹象判 no，"
        "只有间接迹象或节选不足以判断时判 unclear。判 yes 的依据：\n"
        f"{glossary}\n\n"
        "只输出一个 JSON 对象，不要其他文字，格式：\n"
        '{"genre": "...", "subgenres": ["..."], "protagonist": "主角类型，不超过 30 字", "setting": "世界或时代背景，不超过 30 字", '
        '"pacing": "慢热|中等|快", "tone": "基调，不超过 20 字", "one_liner": "一句话简介，不超过 60 字", '
        '"keywords": ["5 到 10 个检索关键词"], "tropes": {"后宫": "yes|no|unclear", "...": "..."}}\n\n'
        f"【档案】\n{profile_text[:max_chars]}"
    )


def parse_card(novel_id: str, text: str, model: str = "") -> BookCard:
    data = extract_json_object(text)
    genre = str(data.get("genre", "")).strip()
    if genre not in GENRES:
        genre = "其他"
    pacing = str(data.get("pacing", "")).strip()
    if pacing not in PACING:
        pacing = ""
    raw_tropes = data.get("tropes") or {}
    tropes: dict[str, str] = {}
    for label in TROPE_GLOSSARY:
        value = str(raw_tropes.get(label, "unclear")).strip().lower() if isinstance(raw_tropes, Mapping) else "unclear"
        tropes[label] = value if value in TROPE_VALUES else "unclear"

    def strs(key: str, limit: int) -> list[str]:
        value = data.get(key) or []
        if not isinstance(value, list):
            value = [value]
        return [str(v).strip()[:40] for v in value if str(v).strip()][:limit]

    return BookCard(
        novel_id=novel_id,
        genre=genre,
        subgenres=strs("subgenres", 5),
        tropes=tropes,
        protagonist=str(data.get("protagonist", ""))[:40],
        setting=str(data.get("setting", ""))[:40],
        pacing=pacing,
        tone=str(data.get("tone", ""))[:30],
        one_liner=str(data.get("one_liner", ""))[:80],
        keywords=strs("keywords", 10),
        model=model,
    )


def card_cache_key(novel_id: str, model: str) -> str:
    return f"{CARD_PROMPT_VERSION}|{model}|{novel_id}"


class CardBuilder:
    """Extract cards through a chat transport with a JSONL cache, resumable and thread-safe."""

    def __init__(self, transport: ChatTransport, model_name: str, cache_path: Path = DEFAULT_CARD_CACHE_PATH, max_tokens: int = 700, max_chars: int = DEFAULT_CARD_MAX_CHARS) -> None:
        self.transport = transport
        self.model_name = model_name
        self.cache_path = cache_path
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self._lock = threading.Lock()
        self.cache: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        out: dict[str, dict[str, Any]] = {}
        with self.cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    out[record["key"]] = record["card"]
        return out

    def _append(self, key: str, card: dict[str, Any]) -> None:
        with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "card": card}, ensure_ascii=False) + "\n")
            self.cache[key] = card

    def build_one(self, novel_id: str, profile_text: str) -> BookCard:
        key = card_cache_key(novel_id, self.model_name)
        cached = self.cache.get(key)
        if cached is not None:
            return BookCard.from_dict(cached)
        response = self.transport.complete(build_card_prompt(profile_text, self.max_chars), max_tokens=self.max_tokens)
        try:
            card = parse_card(novel_id, response, model=self.model_name)
        except (ValueError, json.JSONDecodeError, TypeError) as exc:
            # Not cached: a parse failure is this call's failure, not a fact about the book.
            return BookCard(novel_id, "其他", [], {label: "unclear" for label in TROPE_GLOSSARY}, "", "", "", "", "", [], self.model_name, error=f"parse: {type(exc).__name__}")
        self._append(key, card.to_dict())
        return card

    def build_many(self, items: Iterable[tuple[str, str]], workers: int = 2, on_result: Callable[[BookCard], None] | None = None) -> list[BookCard]:
        items = list(items)
        results: list[BookCard | None] = [None] * len(items)

        def run(index: int) -> None:
            novel_id, text = items[index]
            try:
                results[index] = self.build_one(novel_id, text)
            except Exception as exc:  # noqa: BLE001 - one dead request must not kill the batch
                results[index] = BookCard(novel_id, "其他", [], {label: "unclear" for label in TROPE_GLOSSARY}, "", "", "", "", "", [], self.model_name, error=f"{type(exc).__name__}: {exc}"[:200])
            if on_result is not None:
                on_result(results[index])  # type: ignore[arg-type]

        if workers <= 1:
            for index in range(len(items)):
                run(index)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(run, range(len(items))))
        return [card for card in results if card is not None]


def cards_to_frame(cards: Iterable[BookCard]) -> Any:
    import pandas as pd

    rows = []
    for card in cards:
        row = card.to_dict()
        row["tropes"] = json.dumps(row["tropes"], ensure_ascii=False)
        row["subgenres"] = json.dumps(row["subgenres"], ensure_ascii=False)
        row["keywords"] = json.dumps(row["keywords"], ensure_ascii=False)
        row["card_text"] = card.text()
        rows.append(row)
    return pd.DataFrame(rows)


def load_cards(path: Path = DEFAULT_CARDS_PATH) -> dict[str, BookCard]:
    import pandas as pd

    frame = pd.read_parquet(path)
    out: dict[str, BookCard] = {}
    for row in frame.to_dict(orient="records"):
        row["tropes"] = json.loads(row["tropes"]) if isinstance(row["tropes"], str) else row["tropes"]
        row["subgenres"] = json.loads(row["subgenres"]) if isinstance(row["subgenres"], str) else row["subgenres"]
        row["keywords"] = json.loads(row["keywords"]) if isinstance(row["keywords"], str) else row["keywords"]
        row.pop("card_text", None)
        out[str(row["novel_id"])] = BookCard.from_dict(row)
    return out


def validate_tropes(cards: Mapping[str, BookCard], labels: Mapping[str, Mapping[str, bool]]) -> dict[str, dict[str, Any]]:
    """Per-trope precision / recall of ``tropes[label] == "yes"`` against judge labels.

    ``labels`` is trope -> {novel_id -> judge said violated}. Books without a
    card or with an unclear value are counted separately, not as negatives.
    """

    report: dict[str, dict[str, Any]] = {}
    for trope, by_book in labels.items():
        tp = fp = fn = tn = unclear = missing = 0
        for novel_id, violated in by_book.items():
            card = cards.get(novel_id)
            if card is None:
                missing += 1
                continue
            value = card.tropes.get(trope, "unclear")
            if value == "unclear":
                unclear += 1
                continue
            predicted = value == "yes"
            tp += predicted and violated
            fp += predicted and not violated
            fn += (not predicted) and violated
            tn += (not predicted) and not violated
        decided = tp + fp + fn + tn
        report[trope] = {
            "n": len(by_book),
            "decided": decided,
            "unclear": unclear,
            "missing": missing,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "accuracy": round((tp + tn) / decided, 3) if decided else None,
            "positives": sum(1 for v in by_book.values() if v),
        }
    return report
