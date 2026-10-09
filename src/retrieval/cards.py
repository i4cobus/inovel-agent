"""Offline book cards: one structured record per novel, extracted by a local model from its digest.

The corpus is raw text with no metadata, and users ask for abstract things
(凡人流、慢热、不要系统) that a vector over narrative excerpts does not match.
A card gives every book a short structured description to embed and closed
vocabularies to filter on at retrieval time, which is what turns 「不要后宫」
from a post-hoc check into a pre-filter (design doc D9).

Schema v2 (docs/card-schema-v2.md): the model picks a sub-genre (the genre is
derived), lists the elements that are present, rates five style scales, and
writes short free fields. Unknown labels are dropped and recorded, so vocabulary
drift is measurable. The v1 trope field and its validation against the v1
judge labels are gone: those labels were produced under a different design.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from src.chat_transport import ChatTransport
from src.config import DATA_DIR
from src.llm_json import extract_json_object
from src.retrieval.card_schema import ELEMENT_ALIASES, ELEMENT_QUOTE_SIGNATURES, ELEMENTS, GENRES, MAX_KEYWORDS, STYLE_OPTIONS, SUBGENRE_ALIASES, SUBGENRE_TO_GENRE, UNKNOWN_GENRE, genre_of, vocabulary_text

CARD_PROMPT_VERSION = "card_v3"
DEFAULT_CARDS_PATH = DATA_DIR / "processed" / "book_cards.parquet"
DEFAULT_CARD_CACHE_PATH = DATA_DIR / "cache" / "book_cards.jsonl"
LIST_FIELDS = ("elements", "keywords", "dropped")
DICT_FIELDS = ("style", "evidence")
MIN_EVIDENCE_CHARS = 4
EVIDENCE_WINDOW = 6


@dataclass
class BookCard:
    novel_id: str
    genre: str = UNKNOWN_GENRE  # derived from subgenre
    subgenre: str = ""
    elements: list[str] = field(default_factory=list)
    style: dict[str, str] = field(default_factory=dict)  # dimension -> option ("" when the model gave none)
    protagonist: str = ""
    setting: str = ""
    tone: str = ""  # free one-phrase description, distinct from the 基调 scale
    one_liner: str = ""
    keywords: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # labels the model used that are not in the vocabulary, or not backed by the text
    evidence: dict[str, str] = field(default_factory=dict)  # element (and "subgenre") -> the quote the model gave
    model: str = ""
    prompt_version: str = CARD_PROMPT_VERSION
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "novel_id": self.novel_id,
            "genre": self.genre,
            "subgenre": self.subgenre,
            "elements": list(self.elements),
            "style": dict(self.style),
            "protagonist": self.protagonist,
            "setting": self.setting,
            "tone": self.tone,
            "one_liner": self.one_liner,
            "keywords": list(self.keywords),
            "dropped": list(self.dropped),
            "evidence": dict(self.evidence),
            "model": self.model,
            "prompt_version": self.prompt_version,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BookCard":
        kwargs: dict[str, Any] = {}
        for name in cls.__dataclass_fields__:
            if name in LIST_FIELDS:
                kwargs[name] = list(data.get(name) or [])
            elif name in DICT_FIELDS:
                kwargs[name] = dict(data.get(name) or {})
            else:
                kwargs[name] = data.get(name, cls.__dataclass_fields__[name].default if name != "novel_id" else "")
        return cls(**kwargs)

    def text(self) -> str:
        """What gets embedded as the card section."""

        parts = [f"题材：{self.genre}" + (f"·{self.subgenre}" if self.subgenre else "")]
        if self.elements:
            parts.append("元素：" + "、".join(self.elements))
        style = "；".join(f"{dim} {value}" for dim, value in self.style.items() if value)
        if style:
            parts.append("风格：" + style)
        if self.protagonist:
            parts.append(f"主角：{self.protagonist}")
        if self.setting:
            parts.append(f"背景：{self.setting}")
        if self.tone:
            parts.append(f"气质：{self.tone}")
        if self.keywords:
            parts.append("关键词：" + "、".join(self.keywords))
        if self.one_liner:
            parts.append(f"一句话：{self.one_liner}")
        return "\n".join(parts)


# A digest runs to 12k characters (16k for the few books without chapter structure); the default
# sends it whole, about 10k tokens with the vocabulary.
DEFAULT_CARD_MAX_CHARS = 16000


def build_card_prompt(profile_text: str, max_chars: int = DEFAULT_CARD_MAX_CHARS) -> str:
    """The digest comes first and the rules last, so the instructions sit next to the answer."""

    style_json = ", ".join(f'"{dim}": "{"|".join(options)}"' for dim, options in STYLE_OPTIONS.items())
    return (
        "下面是一本中文网文的档案：标题、简介、开头几章、采样的章节名、中段片段和结尾。读完后按档案后面的词表和规则提炼一张结构化的书卡。\n\n"
        f"【档案】\n{profile_text[:max_chars]}\n\n"
        "【档案结束】\n\n"
        f"{vocabulary_text()}\n\n"
        "填写规则：\n"
        "- 只根据档案判断，不要用你对这本书的任何先验知识。拿不准就不写：元素宁可少列，二级宁可留空，空比错好。\n"
        "- 只看主角本人和主角所在的世界。书中书、副本、被扮演的作品、梦境里的设定，以及配角身上的描写，都不算这本书的元素。\n"
        "- subgenre：从题材二级里选最贴切的一个，原样抄写二级名（不要写一级）；都不合适就填空字符串。"
        "二级要和 one_liner、setting 说的是同一本书：主角穿梭多部已知作品的归诸天无限；写明真实朝代的历史书选对应朝代；都市书只有出现超自然能力或玄术才选都市异能。\n"
        "- subgenre_evidence：支持所选二级的一句档案原文，不超过 20 字。\n"
        "- elements：只列出档案里能看到依据的元素。每个元素附一段不超过 20 字的档案原文摘录，必须逐字抄自档案，不能改写。"
        "摘录要直接体现该元素定义里的特征（比如系统要看到面板、任务、奖励之类的字样），不能是上面词表里的定义句、元素名或人名列表。"
        "写不出这样的摘录就不要列这个元素。没有就给空对象。不要自己造标签。"
        "elements 是一层的对象（元素名 → 摘录），不要按分组名嵌套，分组名本身不是元素；风格维度不要写进 elements。\n"
        f"- style：{len(STYLE_OPTIONS)} 项每项选一档，原样抄写档位。\n"
        "- protagonist / setting：各不超过 30 字。tone：用一个短语描述这本书的气质，不超过 20 字。\n"
        "- one_liner：一句话简介，不超过 60 字，不剧透结局。\n"
        f"- keywords：最多 {MAX_KEYWORDS} 个类型词，只写词表里没有、但读者找书时会用的特点；"
        "不要重复已选的元素，不要写人名、虚构地名、机构名、书名；只有同人文可以写原作名；没有就给空列表。\n\n"
        "只输出一个 JSON 对象，不要其他文字，格式：\n"
        '{"subgenre": "二级名", "subgenre_evidence": "原文摘录", "elements": {"元素名": "原文摘录", "元素名": "原文摘录"}, '
        f"\"style\": {{{style_json}}}, "
        '"protagonist": "...", "setting": "...", "tone": "...", "one_liner": "...", "keywords": ["..."]}'
    )


def _strs(value: Any, limit: int, max_len: int = 40) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    out: list[str] = []
    for item in value:
        text = str(item).strip()[:max_len]
        if text and text not in out:
            out.append(text)
    return out[:limit]


_NORMALISE_RE = re.compile(r"[\s，。！？、：；“”‘’\"'（）()《》【】…—\-,.!?:;]+")


def _normalise(text: str) -> str:
    return _NORMALISE_RE.sub("", text)


def quote_supported(quote: str, source_text: str) -> bool:
    """Does the model's quote actually occur in the text it read?

    Exact match after stripping whitespace and punctuation; a longer quote also passes when
    any 6-character window of it occurs, which tolerates a trimmed or slightly misremembered
    edge but not a paraphrase. Short quotes (under 4 characters) never count as evidence.
    """

    q, src = _normalise(quote), _normalise(source_text)
    if len(q) < MIN_EVIDENCE_CHARS:
        return False
    if q in src:
        return True
    if len(q) >= EVIDENCE_WINDOW + 2:
        return any(q[i : i + EVIDENCE_WINDOW] in src for i in range(0, len(q) - EVIDENCE_WINDOW + 1))
    return False


_VOCAB_NORM: str | None = None


def _vocabulary_normalised() -> str:
    global _VOCAB_NORM
    if _VOCAB_NORM is None:
        _VOCAB_NORM = _normalise(vocabulary_text())
    return _VOCAB_NORM


def _element_pairs(raw: Any) -> list[tuple[str, str]]:
    """(label, quote) pairs from whatever shape the model gave `elements`.

    The prompt asks for a flat {label: quote} object, but the vocabulary is shown in groups and
    the 9B model nested its answer by group name on 377 of 437 cards in the first v2.2 run, so a
    mapping value is walked recursively; a plain list is labels without quotes.
    """

    if raw is None:
        return []
    if isinstance(raw, Mapping):
        pairs: list[tuple[str, str]] = []
        for key, value in raw.items():
            label = str(key).strip()
            if isinstance(value, Mapping):
                pairs.extend(_element_pairs(value))
            elif isinstance(value, list):
                pairs.extend(_element_pairs(value))
            else:
                pairs.append((label, str(value or "").strip()))
        return pairs
    if isinstance(raw, list):
        pairs = []
        for item in raw:
            if isinstance(item, (Mapping, list)):
                pairs.extend(_element_pairs(item))
            else:
                pairs.append((str(item).strip(), ""))
        return pairs
    return [(str(raw).strip(), "")]


def normalise_subgenre(raw: str) -> tuple[str, str]:
    """(subgenre, genre) from what the model wrote: a sub-genre, "一级：二级", "二级（一级）", or just a genre.

    A bare genre keeps the genre and leaves the sub-genre empty rather than losing both,
    which is what the first v2 run did on 94 of 118 cards.
    """

    text = SUBGENRE_ALIASES.get(raw.strip().strip("（）()"), raw.strip().strip("（）()"))
    for sep in ("：", ":", "·", "-", "/", "（", "("):
        if sep in text:
            left, right = (SUBGENRE_ALIASES.get(part.strip(" （）()"), part.strip(" （）()")) for part in text.split(sep, 1))
            if right in SUBGENRE_TO_GENRE:
                return right, SUBGENRE_TO_GENRE[right]
            if left in SUBGENRE_TO_GENRE:
                return left, SUBGENRE_TO_GENRE[left]
            text = left if left in GENRES else right
            break
    if text in SUBGENRE_TO_GENRE:
        return text, SUBGENRE_TO_GENRE[text]
    if text in GENRES:
        return "", text
    return "", UNKNOWN_GENRE


def parse_card(novel_id: str, text: str, model: str = "", source_text: str | None = None) -> BookCard:
    data = extract_json_object(text)
    dropped: list[str] = []

    raw_subgenre = str(data.get("subgenre", "") or "").strip()
    subgenre, genre = normalise_subgenre(raw_subgenre)
    if raw_subgenre and not subgenre:
        dropped.append(f"subgenre_missing:{raw_subgenre}" if genre != UNKNOWN_GENRE else f"subgenre:{raw_subgenre}")

    elements: list[str] = []
    evidence: dict[str, str] = {}
    style_in_elements: dict[str, str] = {}
    for label, quote in _element_pairs(data.get("elements")):
        if label in STYLE_OPTIONS:
            # The 9B model sometimes files the style scales under elements; route them home.
            style_in_elements[label] = quote
            continue
        canonical = ELEMENT_ALIASES.get(label, label)
        if canonical not in ELEMENTS:
            dropped.append(f"element:{label}")
            continue
        if source_text is not None:
            signature = ELEMENT_QUOTE_SIGNATURES.get(canonical)
            if signature and quote and not any(word in quote for word in signature):
                # A real quote, but one that does not show the element: "次元之门" offered for 系统.
                dropped.append(f"element_weak_quote:{canonical}={quote[:40]}")
                continue
            if not quote_supported(quote, source_text):
                # The API models' favourite dodge is to paste the vocabulary's own definition as the quote.
                kind = "element_definition" if _normalise(quote) and _normalise(quote) in _vocabulary_normalised() else "element_unsupported"
                dropped.append(f"{kind}:{canonical}={quote[:40]}")
                continue
        if canonical not in elements:
            elements.append(canonical)
            if quote:
                evidence[canonical] = quote[:40]
    sub_quote = str(data.get("subgenre_evidence", "") or "").strip()
    if sub_quote:
        evidence["subgenre"] = sub_quote[:40]
        if source_text is not None and not quote_supported(sub_quote, source_text):
            dropped.append("subgenre_evidence_unsupported")

    style: dict[str, str] = {}
    raw_style = data.get("style") if isinstance(data.get("style"), Mapping) else {}
    for dim, options in STYLE_OPTIONS.items():
        value = str(raw_style.get(dim, "") or style_in_elements.get(dim, "") or "").strip()
        if value and value not in options:
            dropped.append(f"style:{dim}={value}")
            value = ""
        style[dim] = value

    taken = set(elements) | {subgenre, genre}
    keywords = [k for k in _strs(data.get("keywords"), 20, 20) if k not in taken and ELEMENT_ALIASES.get(k, k) not in taken][:MAX_KEYWORDS]

    return BookCard(
        novel_id=novel_id,
        genre=genre,
        subgenre=subgenre,
        elements=elements,
        style=style,
        protagonist=str(data.get("protagonist", "") or "")[:40],
        setting=str(data.get("setting", "") or "")[:40],
        tone=str(data.get("tone", "") or "")[:30],
        one_liner=str(data.get("one_liner", "") or "")[:80],
        keywords=keywords,
        dropped=dropped,
        evidence=evidence,
        model=model,
    )


def card_cache_key(novel_id: str, model: str) -> str:
    return f"{CARD_PROMPT_VERSION}|{model}|{novel_id}"


class CardBuilder:
    """Extract cards through a chat transport with a JSONL cache, resumable and thread-safe."""

    def __init__(self, transport: ChatTransport, model_name: str, cache_path: Path = DEFAULT_CARD_CACHE_PATH, max_tokens: int = 1200, max_chars: int = DEFAULT_CARD_MAX_CHARS) -> None:
        # Failures are not cached (a parse failure is this call's failure, not a fact about the
        # book), so they are kept here for the caller to write out and look at.
        self.failures: list[dict[str, Any]] = []
        self.transport = transport
        self.model_name = model_name
        self.cache_path = cache_path
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self._lock = threading.Lock()
        self.cache: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        # The raw model response is kept beside the parsed card (since card_v2.3) so that a
        # parser change can be replayed over the cache instead of over the GPU: see reparse().
        self.raw: dict[str, str] = {}
        if not self.cache_path.exists():
            return {}
        out: dict[str, dict[str, Any]] = {}
        with self.cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    out[record["key"]] = record["card"]
                    if record.get("raw"):
                        self.raw[record["key"]] = record["raw"]
        return out

    def _append(self, key: str, card: dict[str, Any], raw: str = "") -> None:
        with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "card": card, "raw": raw}, ensure_ascii=False) + "\n")
            self.cache[key] = card
            if raw:
                self.raw[key] = raw

    def reparse(self, items: Iterable[tuple[str, str]]) -> int:
        """Re-run parse_card over the cached raw responses of these books with the current parser.

        Rewrites the cache file. Returns how many cards were reparsed; books without a cached raw
        response (older cache lines, or never built) are left alone.
        """

        count = 0
        with self._lock:
            for novel_id, profile_text in items:
                key = card_cache_key(novel_id, self.model_name)
                raw = self.raw.get(key)
                if raw is None:
                    continue
                try:
                    card = parse_card(novel_id, raw, model=self.model_name, source_text=profile_text[: self.max_chars])
                except (ValueError, json.JSONDecodeError, TypeError):
                    continue
                self.cache[key] = card.to_dict()
                count += 1
            if count:
                with self.cache_path.open("w", encoding="utf-8") as handle:
                    for key, card_dict in self.cache.items():
                        handle.write(json.dumps({"key": key, "card": card_dict, "raw": self.raw.get(key, "")}, ensure_ascii=False) + "\n")
        return count

    def build_one(self, novel_id: str, profile_text: str) -> BookCard:
        key = card_cache_key(novel_id, self.model_name)
        cached = self.cache.get(key)
        if cached is not None:
            return BookCard.from_dict(cached)
        response = self.transport.complete(build_card_prompt(profile_text, self.max_chars), max_tokens=self.max_tokens)
        try:
            card = parse_card(novel_id, response, model=self.model_name, source_text=profile_text[: self.max_chars])
        except (ValueError, json.JSONDecodeError, TypeError) as exc:
            error = f"parse: {type(exc).__name__}: {exc}"[:200]
            self._record_failure(novel_id, error, response)
            return BookCard(novel_id, model=self.model_name, error=error)
        self._append(key, card.to_dict(), raw=response)
        return card

    def _record_failure(self, novel_id: str, error: str, response: str = "") -> None:
        with self._lock:
            self.failures.append({"novel_id": novel_id, "error": error, "response_chars": len(response), "response_tail": response[-300:]})

    def build_many(self, items: Iterable[tuple[str, str]], workers: int = 2, on_result: Callable[[BookCard], None] | None = None) -> list[BookCard]:
        items = list(items)
        results: list[BookCard | None] = [None] * len(items)

        def run(index: int) -> None:
            novel_id, text = items[index]
            try:
                results[index] = self.build_one(novel_id, text)
            except Exception as exc:  # noqa: BLE001 - one dead request must not kill the batch
                error = f"{type(exc).__name__}: {exc}"[:200]
                self._record_failure(novel_id, error)
                results[index] = BookCard(novel_id, model=self.model_name, error=error)
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
        for name in LIST_FIELDS + DICT_FIELDS:
            row[name] = json.dumps(row[name], ensure_ascii=False)
        row["card_text"] = card.text()
        rows.append(row)
    return pd.DataFrame(rows)


def load_cards(path: Path = DEFAULT_CARDS_PATH) -> dict[str, BookCard]:
    import pandas as pd

    frame = pd.read_parquet(path)
    out: dict[str, BookCard] = {}
    for row in frame.to_dict(orient="records"):
        for name in LIST_FIELDS + DICT_FIELDS:
            if isinstance(row.get(name), str):
                row[name] = json.loads(row[name])
        row.pop("card_text", None)
        out[str(row["novel_id"])] = BookCard.from_dict(row)
    return out


def vocabulary_report(cards: Mapping[str, BookCard]) -> dict[str, Any]:
    """How the model used the vocabulary: per-label counts and what it tried to say outside it."""

    from collections import Counter

    genres: Counter[str] = Counter()
    subgenres: Counter[str] = Counter()
    elements: Counter[str] = Counter()
    style: dict[str, Counter[str]] = {dim: Counter() for dim in STYLE_OPTIONS}
    dropped: Counter[str] = Counter()
    keywords: Counter[str] = Counter()
    for card in cards.values():
        genres[card.genre] += 1
        subgenres[card.subgenre or "（空）"] += 1
        elements.update(card.elements)
        for dim, value in card.style.items():
            style.setdefault(dim, Counter())[value or "（空）"] += 1  # older cards may carry a scale since dropped
        dropped.update(card.dropped)
        keywords.update(card.keywords)
    return {
        "cards": len(cards),
        "genres": dict(genres.most_common()),
        "subgenres": dict(subgenres.most_common()),
        "elements": dict(elements.most_common()),
        "style": {dim: dict(c.most_common()) for dim, c in style.items()},
        "dropped": dict(dropped.most_common(50)),
        "unsupported_elements": dict(Counter(d.split(":", 1)[1].split("=", 1)[0] for d in dropped.elements() if d.startswith("element_unsupported:")).most_common()),
        "definition_copied": sum(1 for d in dropped.elements() if d.startswith("element_definition:")),
        "weak_quotes": sum(1 for d in dropped.elements() if d.startswith("element_weak_quote:")),
        "keywords_top": dict(keywords.most_common(60)),
        "keywords_distinct": len(keywords),
        "elements_per_card": round(sum(len(c.elements) for c in cards.values()) / max(len(cards), 1), 2),
    }
