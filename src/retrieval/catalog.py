"""The book cards as a filter: which books match a genre, a set of elements, a style option.

``search_books`` lets the agent ask for 「仙侠里带宗门和凡人流的」 as a structured filter instead of
hoping the wording lands on the right books. Filters are resolved against the card vocabulary
(``card_schema``) with its aliases, so 「修仙」 means the sub-genre 幻想修仙 and 「金手指」 the element
系统; anything the vocabulary does not know is an error that names the valid options, which the model
can repair in its next call. Books without a card never match a filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.retrieval.card_schema import (
    ELEMENT_ALIASES,
    ELEMENTS,
    GENRES,
    STYLE_OPTIONS,
    SUBGENRE_ALIASES,
    SUBGENRE_TO_GENRE,
    SUBGENRES,
)


class FilterError(ValueError):
    """A filter value outside the vocabulary; the message lists what is valid."""


def resolve_genre(value: str) -> tuple[str | None, str | None]:
    """A genre name -> (genre, None); a sub-genre name or alias -> (its genre, sub-genre)."""

    name = str(value).strip()
    if not name:
        raise FilterError("genre 不能为空")
    if name in SUBGENRES:
        return name, None
    name = SUBGENRE_ALIASES.get(name, name)
    if name in SUBGENRE_TO_GENRE:
        return SUBGENRE_TO_GENRE[name], name
    raise FilterError(f"不认识的题材「{value}」。一级：{'、'.join(GENRES)}；二级：{'、'.join(SUBGENRE_TO_GENRE)}")


def resolve_element(value: str) -> str:
    name = str(value).strip()
    name = ELEMENT_ALIASES.get(name, name)
    if name in ELEMENTS:
        return name
    raise FilterError(f"不认识的元素「{value}」。可用：{'、'.join(ELEMENTS)}")


def resolve_style(value: str) -> tuple[str, str]:
    """「爽度=低」 or 「爽度 低」 -> (dimension, option)."""

    text = str(value).strip().replace("：", "=").replace(":", "=")
    if "=" in text:
        dim, _, option = text.partition("=")
    else:
        dim, _, option = text.partition(" ")
    dim, option = dim.strip(), option.strip()
    if dim not in STYLE_OPTIONS:
        raise FilterError(f"不认识的风格维度「{dim}」。可用：{style_options_text()}")
    if option not in STYLE_OPTIONS[dim]:
        raise FilterError(f"{dim} 只能是 {'/'.join(STYLE_OPTIONS[dim])}，不是「{option}」")
    return dim, option


@dataclass(frozen=True)
class CardFilter:
    genre: str | None = None
    subgenre: str | None = None
    elements: tuple[str, ...] = ()
    style: tuple[tuple[str, str], ...] = ()

    @classmethod
    def parse(cls, genre: str | None = None, elements: Iterable[str] | None = None, style: Iterable[str] | None = None) -> "CardFilter":
        top, sub = resolve_genre(genre) if genre else (None, None)
        els = tuple(dict.fromkeys(resolve_element(e) for e in (elements or []) if str(e).strip()))
        sty = tuple(dict.fromkeys(resolve_style(s) for s in (style or []) if str(s).strip()))
        return cls(top, sub, els, sty)

    @property
    def empty(self) -> bool:
        return not (self.genre or self.elements or self.style)

    def describe(self) -> str:
        parts = []
        if self.genre:
            parts.append("题材=" + (f"{self.genre}·{self.subgenre}" if self.subgenre else self.genre))
        if self.elements:
            parts.append("元素=" + "、".join(self.elements))
        if self.style:
            parts.append("风格=" + "、".join(f"{d}:{o}" for d, o in self.style))
        return "；".join(parts)


class CardCatalog:
    """Cards indexed for filtering. ``cards`` maps novel_id -> BookCard (or anything with the same attributes)."""

    def __init__(self, cards: Mapping[str, Any]) -> None:
        self.cards = dict(cards)

    def __len__(self) -> int:
        return len(self.cards)

    def label(self, novel_id: str) -> str:
        """「玄幻·东方玄幻」 for search rows; empty when the book has no card."""

        card = self.cards.get(str(novel_id))
        if card is None:
            return ""
        return f"{card.genre}·{card.subgenre}" if card.subgenre else str(card.genre)

    def matches(self, card: Any, spec: CardFilter) -> bool:
        if spec.subgenre and card.subgenre != spec.subgenre:
            return False
        if spec.genre and not spec.subgenre and card.genre != spec.genre:
            return False
        if spec.elements:
            have = set(card.elements) | set(getattr(card, "elements_unverified", []) or [])
            if not set(spec.elements) <= have:
                return False
        for dim, option in spec.style:
            if (card.style or {}).get(dim) != option:
                return False
        return True

    def select(self, spec: CardFilter) -> set[str]:
        if spec.empty:
            return set(self.cards)
        return {novel_id for novel_id, card in self.cards.items() if self.matches(card, spec)}


def filter_options_text() -> str:
    """What the tool description tells the model it may filter on."""

    return (
        "genre 可填一级（" + "、".join(GENRES) + "）或二级（如 东方玄幻、幻想修仙、都市异能、架空历史、虚拟网游、无限）；"
        "elements 从书卡元素词表里选（如 系统、穿越、重生、随身空间、种田经营、权谋、争霸建国、无限流、修仙、机甲星际、末世、克苏鲁诡异、鬼怪灵异、刑侦推理、同人、御兽、学院、商战、美食、医术）；"
        "style 写「维度=档」：" + style_options_text()
    )


def style_options_text() -> str:
    return "；".join(dim + "=" + "/".join(options) for dim, options in STYLE_OPTIONS.items())
