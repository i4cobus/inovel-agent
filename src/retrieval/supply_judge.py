"""The judge side of the candidate-supply evaluation: evidence tiers, prompt, verdict cache, spend guard.

Each (query, book) pair gets one call that answers two things at once: how well the book satisfies
the positive preference (0 / 1 / 2) and, for every meta negative, whether the book violates it.
In-text negatives (系统, 异能, ...) are never asked: the full-text density table already decides them
with better coverage than any excerpt, and the flag is shown to the judge as settled context.

Evidence tiers (2026-10-10, undecided until a pilot compares them):

    T1  card + synopsis + density flags
    T2  T1 + 2 independent 700-char windows
    T3  T1 + 5 independent 700-char windows (the v1 sampler)
    T4  T1 + 3 independent 1,500-char windows (whole scenes)

Independent windows come from chapters the digest did not use (``src.evidence``), so the judge
reads text neither the index nor the card builder saw. Verdicts are cached on the pair, the
evidence hash, the model and the prompt version.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.config import DATA_DIR
from src.judge import BudgetGuard, JudgeTransport
from src.preferences import merged_density_from_table, violation_from_density

SUPPLY_JUDGE_PROMPT_VERSION = "supply_judge_v1"
DEFAULT_SUPPLY_JUDGE_CACHE = DATA_DIR / "cache" / "supply_judge_cache.jsonl"
DEFAULT_SUPPLY_JUDGE_MODEL = "qwen3.8-max"
DEFAULT_JUDGE_MAX_TOKENS = 400
PROMPT_OVERHEAD_TOKENS = 450  # instructions + fields, measured roughly on the v1 prompt
CHARS_PER_TOKEN = 1.4  # Chinese prose on the Qwen tokenizer: ~0.7 tokens per character

# tier -> (independent windows, characters per window)
EVIDENCE_TIERS: dict[str, tuple[int, int]] = {"T1": (0, 0), "T2": (2, 700), "T3": (5, 700), "T4": (3, 1500)}
WINDOW_SEPARATOR = "\n\n——\n\n"


@dataclass(frozen=True)
class SupplyJudgeTask:
    query_id: str
    query: str
    novel_id: str
    title: str
    evidence: str
    tier: str
    positives: tuple[str, ...] = ()
    negatives_meta: tuple[str, ...] = ()
    density_flags: tuple[tuple[str, str], ...] = ()  # (term, 违反 | 未见 | 灰区)


def density_flags(densities_row: Mapping[str, float] | None, negatives_in_text: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """The check_term rule per in-text negative, as words the judge can read."""

    flags = []
    for term in negatives_in_text:
        if densities_row is None:
            flags.append((term, "无词频数据"))
            continue
        verdict = violation_from_density(merged_density_from_table(densities_row, term))
        flags.append((term, "违反" if verdict is True else "未见" if verdict is False else "灰区"))
    return tuple(flags)


def independent_windows(raw_text: str, novel_id: str, windows: int, window_chars: int) -> list[str]:
    """Chapters the digest did not use, one window each; empty when there is no text."""

    if not raw_text or windows <= 0:
        return []
    from src.evidence import judge_chapter_indices
    from src.profile import character_window_excerpts, trim_to_sentence
    from src.split_chapters import split_chapters

    chapters = split_chapters(raw_text)
    indices = judge_chapter_indices(novel_id, chapters, windows=windows, seed_salt="supply:")
    out: list[str] = []
    if indices:
        for index in indices:
            chapter = chapters[index]
            body = trim_to_sentence(chapter.text, window_chars)
            if body:
                title = str(chapter.title or "").strip()
                out.append(f"{title}\n{body}" if title else body)
    else:
        from src.evidence import judge_window_fractions

        out.extend(character_window_excerpts(raw_text, excerpt_chars=window_chars, fractions=judge_window_fractions(windows)))
    return out


def build_evidence(tier: str, card_text: str, blurb: str, raw_text: str = "", novel_id: str = "") -> str:
    """Assemble what the judge reads for one book at the given tier."""

    if tier not in EVIDENCE_TIERS:
        raise ValueError(f"tier must be one of {sorted(EVIDENCE_TIERS)}, got {tier!r}")
    parts: list[str] = []
    if card_text.strip():
        parts.append(f"【书卡（离线由模型从 12 章档案归纳，可能有错）】\n{card_text.strip()}")
    if blurb.strip():
        parts.append(f"【作者简介】\n{blurb.strip()}")
    windows, chars = EVIDENCE_TIERS[tier]
    for piece in independent_windows(raw_text, novel_id, windows, chars):
        parts.append(f"【正文摘录】\n{piece}")
    return WINDOW_SEPARATOR.join(parts)


def build_supply_judge_prompt(task: SupplyJudgeTask) -> str:
    positives = "、".join(task.positives) if task.positives else task.query
    negatives = "、".join(task.negatives_meta) if task.negatives_meta else "（无）"
    flags = "；".join(f"{term}：{flag}" for term, flag in task.density_flags) if task.density_flags else "（无）"
    violation_keys = ", ".join(f'"{n}": true|false' for n in task.negatives_meta) or ""
    return (
        "你是中文网络小说推荐系统的评审员。下面是一本候选书的材料，请判断它对这位读者的偏好有多合适。\n\n"
        "【原则】\n"
        "1. 材料只是全书的一小部分。出现即证据，未出现不算证据；材料没提到的要素不能推断全书没有。\n"
        "2. 证据不足时降低 confidence，不要因此压低 positive。按材料能支持的最合理判断打分。\n"
        "3. 书卡是另一个模型离线归纳的，可以参考，但与简介或摘录冲突时以简介和摘录为准。\n\n"
        "【positive 判据】\n"
        "2 = 就是这类书：材料体现了正向偏好的主要部分（题材、流派、设定、主角类型、氛围、节奏）。\n"
        "1 = 沾边：题材大方向对但侧重不同，或只满足一部分偏好。\n"
        "0 = 不是：题材或核心设定明显不符。\n\n"
        "【violations 判据】对每条负向偏好单独判：true = 材料里出现了该要素（出现即算，不必是主线）；false = 没出现。\n"
        "文中词类的负向偏好已经用全文词频判过，列在「词频判定」里，你不用再判它们。\n\n"
        f"读者的检索词：{task.query}\n"
        f"正向偏好：{positives}\n"
        f"负向偏好（要你判的）：{negatives}\n"
        f"词频判定（已定，仅供参考）：{flags}\n"
        f"候选书名：{task.title}\n\n"
        f"{task.evidence}\n\n"
        "只输出 JSON，不要 markdown，不要解释：\n"
        '{"positive": 0|1|2, "violations": {' + violation_keys + '}, "confidence": "high|medium|low", "reason": "一句话依据"}'
    )


def parse_supply_verdict(text: str, negatives_meta: Sequence[str]) -> dict[str, Any] | None:
    """The verdict dict, or None when the response has no usable JSON."""

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, Mapping):
        return None
    try:
        positive = max(0, min(2, int(data.get("positive", 0))))
    except (TypeError, ValueError):
        return None
    raw_violations = data.get("violations") if isinstance(data.get("violations"), Mapping) else {}
    violations = {}
    for negative in negatives_meta:
        value = raw_violations.get(negative, False)
        violations[negative] = value if isinstance(value, bool) else str(value).strip().lower() in ("true", "yes", "是", "1")
    confidence = str(data.get("confidence", "low")).lower()
    return {
        "positive": positive,
        "violations": violations,
        "confidence": confidence if confidence in ("high", "medium", "low") else "low",
        "reason": str(data.get("reason", ""))[:300],
    }


def supply_cache_key(task: SupplyJudgeTask, model: str) -> str:
    payload = {
        "query_id": task.query_id,
        "query": task.query,
        "novel_id": task.novel_id,
        "negatives_meta": list(task.negatives_meta),
        "evidence": hashlib.sha256(task.evidence.encode("utf-8")).hexdigest(),
        "tier": task.tier,
        "model": model,
        "prompt_version": SUPPLY_JUDGE_PROMPT_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_verdict_cache(path: Path = DEFAULT_SUPPLY_JUDGE_CACHE) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            cache[record["key"]] = record["verdict"]
    return cache


def append_verdict(path: Path, key: str, task: SupplyJudgeTask, verdict: Mapping[str, Any], model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"key": key, "query_id": task.query_id, "novel_id": task.novel_id, "tier": task.tier, "model": model, "verdict": dict(verdict)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def estimate_prompt_tokens(task: SupplyJudgeTask) -> int:
    return PROMPT_OVERHEAD_TOKENS + int(len(task.evidence) / CHARS_PER_TOKEN) + int(len(task.query) / CHARS_PER_TOKEN)


@dataclass
class SupplyJudgeSummary:
    requested: int = 0
    cache_hits: int = 0
    judged: int = 0
    parse_failed: int = 0
    request_failed: int = 0
    skipped_over_budget: int = 0
    spent: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ResultCallback = Callable[[SupplyJudgeTask, Mapping[str, Any]], None]


def run_supply_judgements(
    tasks: Sequence[SupplyJudgeTask],
    transport: JudgeTransport,
    model: str,
    budget: BudgetGuard,
    cache_path: Path = DEFAULT_SUPPLY_JUDGE_CACHE,
    workers: int = 6,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    on_result: ResultCallback | None = None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], SupplyJudgeSummary]:
    """Judge every task once, reusing the cache and stopping at the budget cap.

    Returns verdicts keyed by (query_id, novel_id). A failed or unparseable response leaves the pair
    absent rather than recorded as 0, so a dead request never reads as "not relevant"."""

    summary = SupplyJudgeSummary(requested=len(tasks))
    verdicts: dict[tuple[str, str], dict[str, Any]] = {}
    cache = load_verdict_cache(cache_path)
    pending: list[tuple[str, SupplyJudgeTask]] = []
    for task in tasks:
        key = supply_cache_key(task, model)
        if key in cache:
            verdicts[(task.query_id, task.novel_id)] = cache[key]
            summary.cache_hits += 1
            if on_result:
                on_result(task, cache[key])
        else:
            pending.append((key, task))
    if not pending:
        return verdicts, summary
    lock = threading.Lock()

    def run(entry: tuple[str, SupplyJudgeTask]) -> None:
        key, task = entry
        if budget.exhausted():
            with lock:
                summary.skipped_over_budget += 1
            return
        try:
            text, usage = transport.complete_with_usage(build_supply_judge_prompt(task), max_tokens)
        except Exception:  # noqa: BLE001 - a dead request is counted, not raised, so the run keeps going
            with lock:
                summary.request_failed += 1
            return
        spent = budget.record(usage)
        verdict = parse_supply_verdict(text, task.negatives_meta)
        with lock:
            summary.spent = spent
            summary.prompt_tokens += usage.prompt_tokens
            summary.completion_tokens += usage.completion_tokens
            if verdict is None:
                summary.parse_failed += 1
                return
            summary.judged += 1
            verdicts[(task.query_id, task.novel_id)] = verdict
            append_verdict(cache_path, key, task, verdict, model)
        if on_result:
            on_result(task, verdict)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        list(executor.map(run, pending))
    return verdicts, summary
