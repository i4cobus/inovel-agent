"""check_trope: evidence-sampled, model-judged, cached trope verdicts.

The glossary is the judging standard. Each entry says what, if seen in a
~5,000-character sample, counts as *yes*. The model is asked to quote the
evidence it relied on so a verdict can be audited and a wrong one traced to
either the sample or the reading.

Reliability is measured, not assumed: before the agent uses this tool its
per-trope precision and recall are computed against the v1 judge's and human
violation labels (docs/agent-plan.md §3.2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from src.chat_transport import ChatTransport
from src.config import DATA_DIR
from src.evidence import sample_judge_evidence
from src.llm_json import extract_json_object

TROPE_PROMPT_VERSION = "trope_v1"
TROPE_EVIDENCE_SALT = "trope:"
DEFAULT_TROPE_CACHE_PATH = DATA_DIR / "cache" / "trope_cache.jsonl"
VERDICTS = ("yes", "no", "unclear")

# 判 yes 的依据。改动任何一条都要升 TROPE_PROMPT_VERSION，否则缓存里的旧结论会混进来。
TROPE_GLOSSARY: dict[str, str] = {
    # 题材类
    "玄幻": "架空世界，有修炼体系或超自然力量体系，且不是仙侠语境（没有修仙、飞升、道门词汇）",
    "灵异": "鬼怪、灵体、诅咒、凶宅等超自然恐怖元素是主线",
    "超能力": "现代或近未来背景，人物拥有非修炼来源的特殊能力",
    "克苏鲁": "不可名状的旧日支配者、理智值、疯狂、邪神信仰等要素",
    "言情": "男女感情线是主线，情节围绕两人关系推进",
    "争霸": "主角以势力扩张、攻城略地、建国称帝为主线",
    "宫斗": "后宫或朝堂内部的权谋争斗是主线",
    # 人物关系与设定类
    "后宫": "主角与两名以上异性存在并存的、被叙事认可的感情或伴侣关系",
    "种马": "主角与多名异性发生关系且叙事不作感情铺垫，关系对象持续增加",
    "独狼": "主角长期无固定伙伴，拒绝组织与同伴，独自行动是性格设定",
    "金手指": "主角拥有他人没有的外挂式优势（系统、随身空间、重生先知、特殊血脉）且情节依赖它",
    "开挂": "主角的优势远超同阶且几乎没有代价",
    "开局无敌": "故事开始时主角已处于实力顶点或立即获得顶级实力",
    "速通": "升级或目标达成极快，几章内跨越常规需要大量篇幅的阶段",
    "魔改": "以已有作品或历史为底本，改动人物命运或核心设定",
    # 叙事风格与读者口味类（主观，按代理特征判）
    "爽文": "冲突被迅速解决，主角连续占上风，受挫后快速反转，几乎没有长期失败",
    "打脸": "反复出现「被轻视后当众证明实力、对方难堪」的情节模式",
    "宠文": "男女主之间一方对另一方无条件偏爱、纵容，几乎没有感情危机",
    "圣母": "主角反复为陌生人或敌人牺牲自身利益，不计后果地原谅或救助对手",
    "玛丽苏": "主角被几乎所有人无理由喜爱或倾慕，缺点不被叙事承认",
    "恋爱脑": "人物的重大决定主要由感情驱动，为感情放弃明显更重要的目标",
    "虐主": "主角持续遭受羞辱、重大损失或肉体精神折磨，且篇幅占比高",
    "压抑": "整体基调阴郁，反复出现绝望、无力、牺牲无回报的情节，少有轻松段落",
    "狗血": "密集的巧合、身世反转、误会、三角关系等戏剧化桥段",
    "搞笑": "大量段子、吐槽、滑稽情节，基调轻松",
    "无脑": "冲突解决不依赖策略或信息，靠实力碾压或对手犯低级错误",
    "小白": "文字直白、人物扁平、情节套路化，解释性叙述多于描写",
}


# ---------------------------------------------------------------- the card first (2026-10-10)
#
# A book card (src/retrieval/cards.py) already states the genre, three to four elements and five style
# scales, read by a model from twelve whole chapters. Where the card speaks to a trope, its answer is
# cheaper and better founded than a fresh reading of six 700-character windows, so check_trope consults
# it first and reads the text only when the card is silent. Two asymmetries are deliberate:
#   * a style scale or the genre is single-valued, so a *different* value is evidence of "no";
#   * the element list is sparse (about three per book), so a *missing* element is not evidence of
#     anything and the text is read instead.
CARD_STYLE_RULES: dict[str, tuple[str, tuple[str, ...]]] = {  # trope -> (scale, values that mean yes)
    "后宫": ("感情线", ("多女主",)),
    "压抑": ("基调", ("沉重压抑",)),
    "搞笑": ("基调", ("轻松搞笑",)),
    "开局无敌": ("主角起点", ("开局无敌",)),
    "言情": ("感情线", ("单女主主线",)),
}
CARD_GENRE_RULES: dict[str, tuple[str, ...]] = {"玄幻": ("玄幻",), "言情": ("言情",)}  # trope -> genres that mean yes
CARD_ELEMENT_RULES: dict[str, tuple[str, ...]] = {  # trope -> elements any of which means yes
    "金手指": ("系统", "随身空间", "重生"),
    "灵异": ("鬼怪灵异",),
    "超能力": ("异能",),
    "克苏鲁": ("克苏鲁诡异",),
    "争霸": ("争霸建国",),
}
CARD_SUBGENRE_RULES: dict[str, tuple[str, ...]] = {"灵异": ("灵异民俗",), "超能力": ("都市异能",)}
# 爽度 is three-valued with a wide middle: only the extremes decide.
CARD_SCALE_EXTREMES: dict[str, tuple[str, str, str]] = {"爽文": ("爽度", "高", "低")}  # trope -> (scale, yes value, no value)


def card_verdict(card: Any, trope: str) -> dict[str, Any] | None:
    """What the book card says about a trope, as a check_trope result, or None when it is silent."""

    if card is None:
        return None
    style = getattr(card, "style", {}) or {}
    elements = set(getattr(card, "elements", []) or [])
    genre = getattr(card, "genre", "") or ""
    subgenre = getattr(card, "subgenre", "") or ""

    def answer(verdict: str, confidence: str, basis: str) -> dict[str, Any]:
        return {"verdict": verdict, "quotes": [], "confidence": confidence, "reason": f"书卡：{basis}", "source": "card"}

    if trope in CARD_ELEMENT_RULES and elements & set(CARD_ELEMENT_RULES[trope]):
        return answer("yes", "high", "元素 " + "、".join(sorted(elements & set(CARD_ELEMENT_RULES[trope]))))
    if trope in CARD_SUBGENRE_RULES and subgenre in CARD_SUBGENRE_RULES[trope]:
        return answer("yes", "high", f"题材 {genre}·{subgenre}")
    if trope in CARD_GENRE_RULES:
        if genre in CARD_GENRE_RULES[trope]:
            return answer("yes", "high", f"题材 {genre}")
        if genre and genre != "其他" and trope not in CARD_STYLE_RULES:
            return answer("no", "medium", f"题材 {genre}")
    if trope in CARD_STYLE_RULES:
        scale, yes_values = CARD_STYLE_RULES[trope]
        value = style.get(scale, "")
        if value in yes_values:
            return answer("yes", "high", f"{scale} {value}")
        if value:
            return answer("no", "medium", f"{scale} {value}")
    if trope in CARD_SCALE_EXTREMES:
        scale, yes_value, no_value = CARD_SCALE_EXTREMES[trope]
        value = style.get(scale, "")
        if value == yes_value:
            return answer("yes", "high", f"{scale} {value}")
        if value == no_value:
            return answer("no", "medium", f"{scale} {value}")
    return None


@dataclass(frozen=True)
class TropeVerdict:
    verdict: str
    quotes: tuple[str, ...]
    confidence: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "quotes": list(self.quotes), "confidence": self.confidence, "reason": self.reason}


def build_trope_prompt(trope: str, definition: str, evidence: str) -> str:
    return (
        "下面是一本中文网文的作品简介和从全书不同位置采样的若干段落。请判断这本书是否属于下面这个标签。\n\n"
        f"标签：{trope}\n"
        f"判 yes 的依据：{definition}\n\n"
        "规则：只根据给出的文字判断，不要用你对这本书的任何先验知识。看到符合依据的内容判 yes；"
        "完全没有相关迹象判 no；只有间接迹象、或采样段落不足以判断时判 unclear。"
        "引用最多三段你依据的原文（每段不超过 60 字），没有就给空列表。\n\n"
        "只输出一个 JSON 对象，不要其他文字：\n"
        '{"verdict": "yes|no|unclear", "confidence": "high|medium|low", "quotes": ["..."], "reason": "一句话"}\n\n'
        f"【采样文本】\n{evidence}"
    )


def parse_trope_verdict(text: str) -> TropeVerdict:
    data = extract_json_object(text)
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    quotes = data.get("quotes") or []
    if not isinstance(quotes, list):
        quotes = [str(quotes)]
    confidence = str(data.get("confidence", "low")).strip().lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    return TropeVerdict(
        verdict=verdict,
        quotes=tuple(str(quote)[:120] for quote in quotes[:3]),
        confidence=confidence,
        reason=str(data.get("reason", ""))[:200],
    )


def trope_cache_key(novel_id: str, trope: str, model: str) -> str:
    return f"{TROPE_PROMPT_VERSION}|{model}|{novel_id}|{trope}"


class CachedTropeJudge:
    """Samples evidence with its own seed, asks the model, caches by (novel, trope, prompt version, model)."""

    def __init__(
        self,
        transport: ChatTransport,
        raw_text_lookup: Callable[[str], str | None],
        model_name: str,
        cache_path: Path = DEFAULT_TROPE_CACHE_PATH,
        glossary: dict[str, str] = TROPE_GLOSSARY,
        windows: int = 6,
        window_chars: int = 700,
        max_tokens: int = 400,
        cards: Mapping[str, Any] | None = None,
    ) -> None:
        self.cards = cards or {}
        self.transport = transport
        self.raw_text_lookup = raw_text_lookup
        self.model_name = model_name
        self.cache_path = cache_path
        self.glossary = glossary
        self.windows = windows
        self.window_chars = window_chars
        self.max_tokens = max_tokens
        self._cache: dict[str, dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        cache: dict[str, dict[str, Any]] = {}
        with self.cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    cache[record["key"]] = record["verdict"]
        return cache

    def _append_cache(self, key: str, verdict: dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "verdict": verdict}, ensure_ascii=False) + "\n")

    def judge(self, novel_id: str, trope: str) -> dict[str, Any]:
        from src.agent.tools import ToolError

        definition = self.glossary.get(trope)
        if definition is None:
            return {"novel_id": novel_id, "trope": trope, "verdict": "unknown_trope", "known_tropes": sorted(self.glossary)}
        from_card = card_verdict(self.cards.get(novel_id), trope)
        if from_card is not None:
            return {"novel_id": novel_id, "trope": trope, **from_card}
        key = trope_cache_key(novel_id, trope, self.model_name)
        cached = self._cache.get(key)
        if cached is not None:
            return {"novel_id": novel_id, "trope": trope, **cached, "cached": True, "source": "text"}
        text = self.raw_text_lookup(novel_id)
        if not text:
            raise ToolError(f"没有这本书的原文：{novel_id}")
        evidence = sample_judge_evidence(
            text, novel_id, windows=self.windows, window_chars=self.window_chars, seed_salt=TROPE_EVIDENCE_SALT
        )
        response = self.transport.complete(build_trope_prompt(trope, definition, evidence), max_tokens=self.max_tokens)
        try:
            verdict = parse_trope_verdict(response).to_dict()
        except (ValueError, json.JSONDecodeError, TypeError):
            # Not cached: a parse failure is the model's failure on this call, not a fact about the book.
            return {"novel_id": novel_id, "trope": trope, "verdict": "unclear", "quotes": [], "confidence": "low", "reason": "判定输出无法解析"}
        self._cache[key] = verdict
        self._append_cache(key, verdict)
        return {"novel_id": novel_id, "trope": trope, **verdict, "cached": False, "source": "text"}
