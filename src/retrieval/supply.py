"""候选供给评估：按 agent 的实际用法衡量索引阶段。

``search_books(query, k=10)`` 收到的是 agent 写的正向偏好，返回 10 本候选；负向约束由
agent 用 check_term / check_trope 在候选里核，最后至少推 3 本。所以索引的任务不是排序竞赛，
而是候选供给：top-k 里有多少真满足正向偏好，扣掉违反负向的之后还剩不剩 3 本。

这里把评估拆成四块，前三块不花钱：

1. 查询集：agent 任务的正向词（主体）、同一意图的改写、轨迹里真实发出的 search_books 查询、
   以及 v1 的 59 条旧查询（只为锚点）。
2. 池：每个配置对每条查询取 top-``depth``，并集就是要判的（查询，书）对。
3. 免 judge 指标：锚点 Hit@k、卡一致率@k、改写重合度、密度表清洁率、查询延迟。
4. judge 指标（见 ``supply_judge``）：正向精确率@k、可行@k。

2026-10-10 定：旧的 v1 强相关集弃用，judge 用 Qwen3.8-Max，池 top-20，100 对人工校准。
"""

from __future__ import annotations

import json
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.config import PROJECT_ROOT
from src.evaluation import title_matches_anchor
from src.preferences import constraint_violation_from_densities, is_rule_checkable, parse_preference_query
from src.retrieval.card_schema import ELEMENT_ALIASES, ELEMENTS, GENRES, SUBGENRE_ALIASES, SUBGENRE_TO_GENRE

SUPPLY_K = 10
SUPPLY_DEPTH = 20
MIN_FEASIBLE = 3  # the agent must end with at least three recommendations
DEFAULT_TASK_PATHS = (
    PROJECT_ROOT / "eval" / "agent" / "tasks" / "constrained_rec.jsonl",
    PROJECT_ROOT / "eval" / "agent" / "tasks" / "memory.jsonl",
)
DEFAULT_LEGACY_QUERIES_PATH = PROJECT_ROOT / "eval" / "eval_queries.jsonl"
DEFAULT_REWRITES_PATH = PROJECT_ROOT / "eval" / "retrieval_supply" / "rewrites.jsonl"
DEFAULT_SUPPLY_RESULTS_DIR = PROJECT_ROOT / "eval" / "results" / "retrieval_supply"
POOL_FILE = "pool.jsonl"
QUERIES_FILE = "queries.jsonl"


@dataclass(frozen=True)
class SupplyQuery:
    """One query as ``search_books`` would receive it, plus the constraints it must serve."""

    query_id: str
    task_id: str  # queries of one intent share it (task, its rewrites, its trajectory queries)
    query: str
    positives: tuple[str, ...] = ()
    negatives_in_text: tuple[str, ...] = ()
    negatives_meta: tuple[str, ...] = ()
    source: str = "task"  # task | rewrite | trajectory | legacy
    anchors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SupplyQuery":
        return cls(
            query_id=str(data["query_id"]),
            task_id=str(data.get("task_id") or data["query_id"]),
            query=str(data["query"]),
            positives=tuple(data.get("positives") or ()),
            negatives_in_text=tuple(data.get("negatives_in_text") or ()),
            negatives_meta=tuple(data.get("negatives_meta") or ()),
            source=str(data.get("source") or "task"),
            anchors=tuple(data.get("anchors") or ()),
        )


def split_negatives(terms: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """In-text negatives (the density rule can score them) versus meta negatives (a reader must judge)."""

    in_text, meta = [], []
    for term in terms:
        term = str(term).strip()
        if not term:
            continue
        (in_text if is_rule_checkable(term) else meta).append(term)
    return tuple(in_text), tuple(meta)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def task_queries(paths: Sequence[Path] = DEFAULT_TASK_PATHS) -> list[SupplyQuery]:
    """One query per agent task: its positive terms joined, exactly what the tool's query parser keeps."""

    out: list[SupplyQuery] = []
    for path in paths:
        for record in _read_jsonl(Path(path)):
            positives = tuple(str(p) for p in record.get("positives") or [])
            if not positives:
                continue
            in_text, meta = split_negatives(list(record.get("negatives_in_text") or []) + list(record.get("negatives_meta") or []))
            task_id = str(record["task_id"])
            out.append(SupplyQuery(query_id=task_id, task_id=task_id, query=" ".join(positives), positives=positives, negatives_in_text=in_text, negatives_meta=meta, source="task"))
    return out


def legacy_queries(path: Path = DEFAULT_LEGACY_QUERIES_PATH) -> list[SupplyQuery]:
    """The v1 queries: kept for their anchor titles, the only model-free truth in the benchmark."""

    out: list[SupplyQuery] = []
    for record in _read_jsonl(path):
        in_text, meta = split_negatives(record.get("unwanted") or [])
        positives = tuple(str(w) for w in record.get("wanted") or [])
        query_id = f"legacy:{record['query_id']}"
        out.append(
            SupplyQuery(
                query_id=query_id,
                task_id=query_id,
                query=str(record["query"]),
                positives=positives or tuple(parse_preference_query(str(record["query"])).positive_terms),
                negatives_in_text=in_text,
                negatives_meta=meta,
                source="legacy",
                anchors=tuple(record.get("anchor_titles") or ()),
            )
        )
    return out


def rewrite_queries(path: Path, tasks: Mapping[str, SupplyQuery]) -> list[SupplyQuery]:
    """Paraphrases of a task's intent (``{"task_id", "variant", "query"}`` per line); constraints come from the task."""

    out: list[SupplyQuery] = []
    for record in _read_jsonl(path):
        base = tasks.get(str(record.get("task_id")))
        if base is None or not str(record.get("query", "")).strip():
            continue
        variant = str(record.get("variant") or len(out))
        out.append(SupplyQuery(query_id=f"{base.task_id}#{variant}", task_id=base.task_id, query=str(record["query"]).strip(), positives=base.positives, negatives_in_text=base.negatives_in_text, negatives_meta=base.negatives_meta, source="rewrite"))
    return out


def trajectory_queries(paths: Sequence[Path], tasks: Mapping[str, SupplyQuery]) -> list[SupplyQuery]:
    """The search_books queries the agent actually emitted, read from trajectory JSONL; one per distinct (task, query).

    Reads the eval runner's ``trajectories.redacted.jsonl`` (one trajectory per session, ``task_id``
    like ``rec-004/s0`` with the bare id in ``metadata``) as well as a plain per-task record. These
    are the most faithful paraphrases there are: the agent's own model wrote them from the user's
    message, so no separate rewriting model is needed (2026-10-10)."""

    out: list[SupplyQuery] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for record in _read_jsonl(Path(path)):
            metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
            task_id = str(metadata.get("task_id") or str(record.get("task_id") or "").split("/s")[0])
            base = tasks.get(task_id)
            if base is None:
                continue
            for step in record.get("steps") or []:
                for observation in step.get("observations") or []:
                    if observation.get("tool") != "search_books":
                        continue
                    query = str((observation.get("arguments") or {}).get("query") or "").strip()
                    if not query or (task_id, query) in seen:
                        continue
                    seen.add((task_id, query))
                    out.append(SupplyQuery(query_id=f"{task_id}@{len([q for q in out if q.task_id == task_id]) + 1}", task_id=task_id, query=query, positives=base.positives, negatives_in_text=base.negatives_in_text, negatives_meta=base.negatives_meta, source="trajectory"))
    return out


# ---------------------------------------------------------------- pool


@dataclass(frozen=True)
class PoolRow:
    query_id: str
    config: str
    rank: int
    novel_id: str
    title: str
    score: float = 0.0
    section_kind: str = ""
    card_match: str = ""  # set by the card-filtered searcher


def build_pool(searchers: Sequence[Any], queries: Sequence[SupplyQuery], depth: int = SUPPLY_DEPTH) -> tuple[list[PoolRow], dict[str, list[float]]]:
    """Top-``depth`` of every searcher for every query, plus per-config query latencies (ms).

    The query reaches each searcher verbatim, exactly as the tool sends it (no term stripping since 2026-10-11), so the pool measures what
    ``search_books`` would have returned."""

    rows: list[PoolRow] = []
    latencies: dict[str, list[float]] = defaultdict(list)
    for searcher in searchers:
        name = getattr(searcher, "name", "searcher")
        for query in queries:
            started = time.perf_counter()
            results = searcher.search(query.query, depth)
            latencies[name].append((time.perf_counter() - started) * 1000)
            for position, row in enumerate(results, start=1):
                rows.append(
                    PoolRow(
                        query_id=query.query_id,
                        config=name,
                        rank=int(row.get("rank") or position),
                        novel_id=str(row.get("novel_id", "")),
                        title=str(row.get("title_guess", "") or row.get("title", "")),
                        score=float(row.get("score", 0.0) or 0.0),
                        section_kind=str(row.get("section_kind", "") or ""),
                        card_match=str(row.get("card_match", "") or ""),
                    )
                )
    return rows, dict(latencies)


def write_pool(rows: Sequence[PoolRow], queries: Sequence[SupplyQuery], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / QUERIES_FILE).write_text("".join(json.dumps(q.to_dict(), ensure_ascii=False) + "\n" for q in queries), encoding="utf-8")
    (out_dir / POOL_FILE).write_text("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def load_pool(out_dir: Path) -> tuple[list[PoolRow], list[SupplyQuery]]:
    rows = [PoolRow(**record) for record in _read_jsonl(out_dir / POOL_FILE)]
    queries = [SupplyQuery.from_dict(record) for record in _read_jsonl(out_dir / QUERIES_FILE)]
    return rows, queries


def pool_pairs(rows: Sequence[PoolRow], depth: int = SUPPLY_DEPTH) -> dict[tuple[str, str], str]:
    """Distinct (query_id, novel_id) -> title over every config's top-``depth``: the set a judge must label."""

    pairs: dict[tuple[str, str], str] = {}
    for row in rows:
        if row.rank <= depth:
            pairs.setdefault((row.query_id, row.novel_id), row.title)
    return pairs


def top_k(rows: Sequence[PoolRow], k: int) -> dict[str, dict[str, list[PoolRow]]]:
    """config -> query_id -> its first k rows in rank order."""

    out: dict[str, dict[str, list[PoolRow]]] = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda r: (r.config, r.query_id, r.rank)):
        if row.rank <= k:
            out[row.config][row.query_id].append(row)
    return out


# ---------------------------------------------------------------- vocabulary targets in a query


def query_genre_targets(positives: Iterable[str]) -> tuple[set[str], set[str]]:
    """(genres, sub-genres) a query names outright, through the card vocabulary and its aliases."""

    genres: set[str] = set()
    subgenres: set[str] = set()
    for term in positives:
        term = str(term).strip()
        if term in GENRES:  # a bare genre in a query means the genre, not the alias table's default sub-genre
            genres.add(term)
            continue
        sub = SUBGENRE_ALIASES.get(term, term)
        if sub in SUBGENRE_TO_GENRE:
            subgenres.add(sub)
            genres.add(SUBGENRE_TO_GENRE[sub])
    return genres, subgenres


def query_element_targets(positives: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for term in positives:
        term = str(term).strip()
        label = ELEMENT_ALIASES.get(term, term)
        if label in ELEMENTS:
            out.add(label)
    return out


# ---------------------------------------------------------------- metrics


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def density_violates(densities: Mapping[str, Mapping[str, float]] | None, novel_id: str, terms: Sequence[str]) -> bool | None:
    """The check_term rule for a candidate: True violates, False clean, None grey or unknown book / no in-text negatives."""

    if not densities or not terms:
        return None
    row = densities.get(novel_id)
    if row is None:
        return None
    return constraint_violation_from_densities(row, list(terms))


def judge_free_metrics(
    rows: Sequence[PoolRow],
    queries: Sequence[SupplyQuery],
    k: int = SUPPLY_K,
    cards: Mapping[str, Any] | None = None,
    densities: Mapping[str, Mapping[str, float]] | None = None,
    latencies: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per config: anchor Hit@k, card genre / element consistency@k, rewrite overlap@k, density-clean@k, latency."""

    by_id = {q.query_id: q for q in queries}
    report: dict[str, dict[str, Any]] = {}
    for config, per_query in top_k(rows, k).items():
        anchor_hits: list[float] = []
        genre_shares: list[float] = []
        element_shares: list[float] = []
        clean_shares: list[float] = []
        sets_by_task: dict[str, list[set[str]]] = defaultdict(list)
        for query_id, top in per_query.items():
            query = by_id.get(query_id)
            if query is None:
                continue
            ids = [row.novel_id for row in top]
            for anchor in query.anchors:
                anchor_hits.append(1.0 if any(title_matches_anchor(row.title, anchor) for row in top) else 0.0)
            if cards:
                genres, subgenres = query_genre_targets(query.positives)
                carded = [cards[n] for n in ids if n in cards]
                if (genres or subgenres) and carded:
                    genre_shares.append(sum(1 for c in carded if c.subgenre in subgenres or c.genre in genres) / len(carded))
                wanted_elements = query_element_targets(query.positives)
                if wanted_elements and carded:
                    element_shares.append(sum(1 for c in carded if wanted_elements & set(c.elements)) / len(carded))
            if densities and query.negatives_in_text:
                verdicts = [density_violates(densities, n, query.negatives_in_text) for n in ids]
                known = [v for v in verdicts if v is not None]
                if known:
                    clean_shares.append(sum(1 for v in known if v is False) / len(known))
            sets_by_task[query.task_id].append(set(ids))
        overlaps: list[float] = []
        for variants in sets_by_task.values():
            for i in range(len(variants)):
                for j in range(i + 1, len(variants)):
                    union = variants[i] | variants[j]
                    overlaps.append(len(variants[i] & variants[j]) / len(union) if union else 1.0)
        lat = list((latencies or {}).get(config, []))
        report[config] = {
            "queries": len(per_query),
            f"anchor_hit@{k}": _mean(anchor_hits),
            "anchors": len(anchor_hits),
            f"card_genre_consistency@{k}": _mean(genre_shares),
            "genre_queries": len(genre_shares),
            f"card_element_consistency@{k}": _mean(element_shares),
            "element_queries": len(element_shares),
            f"rewrite_overlap@{k}": _mean(overlaps),
            "rewrite_pairs": len(overlaps),
            f"density_clean@{k}": _mean(clean_shares),
            "density_queries": len(clean_shares),
            "latency_ms_p50": round(statistics.median(lat), 1) if lat else None,
        }
    return report


def candidate_ok(verdict: Mapping[str, Any] | None, density: bool | None, strict: bool) -> bool | None:
    """Does a judged candidate count as usable? None when there is no verdict for it."""

    if verdict is None:
        return None
    label = int(verdict.get("positive", 0))
    satisfied = label == 2 if strict else label >= 1
    violated = density is True or any(bool(v) for v in (verdict.get("violations") or {}).values())
    return satisfied and not violated


def judged_metrics(
    rows: Sequence[PoolRow],
    queries: Sequence[SupplyQuery],
    verdicts: Mapping[tuple[str, str], Mapping[str, Any]],
    k: int = SUPPLY_K,
    densities: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per config: positive precision@k (strict = label 2, lenient = label >= 1), feasible@k (>= 3 usable), coverage."""

    by_id = {q.query_id: q for q in queries}
    report: dict[str, dict[str, Any]] = {}
    for config, per_query in top_k(rows, k).items():
        strict_prec: list[float] = []
        lenient_prec: list[float] = []
        feasible: list[float] = []
        feasible_lenient: list[float] = []
        judged = total = 0
        for query_id, top in per_query.items():
            query = by_id.get(query_id)
            if query is None:
                continue
            labels = []
            ok_strict = ok_lenient = 0
            for row in top:
                total += 1
                verdict = verdicts.get((query_id, row.novel_id))
                if verdict is None:
                    continue
                judged += 1
                labels.append(int(verdict.get("positive", 0)))
                density = density_violates(densities, row.novel_id, query.negatives_in_text)
                ok_strict += bool(candidate_ok(verdict, density, strict=True))
                ok_lenient += bool(candidate_ok(verdict, density, strict=False))
            if not labels:
                continue
            strict_prec.append(sum(1 for l in labels if l == 2) / len(labels))
            lenient_prec.append(sum(1 for l in labels if l >= 1) / len(labels))
            feasible.append(1.0 if ok_strict >= MIN_FEASIBLE else 0.0)
            feasible_lenient.append(1.0 if ok_lenient >= MIN_FEASIBLE else 0.0)
        report[config] = {
            "queries_judged": len(strict_prec),
            f"positive_precision@{k}": _mean(strict_prec),
            f"positive_precision_lenient@{k}": _mean(lenient_prec),
            f"feasible@{k}": _mean(feasible),
            f"feasible_lenient@{k}": _mean(feasible_lenient),
            "judged_coverage": round(judged / total, 4) if total else None,
        }
    return report


def format_metrics(report: Mapping[str, Mapping[str, Any]]) -> str:
    if not report:
        return ""
    columns = list(next(iter(report.values())).keys())
    lines = ["| config | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for config, metrics in report.items():
        lines.append(f"| {config} | " + " | ".join("" if metrics[c] is None else str(metrics[c]) for c in columns) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- human calibration


def calibration_sample(
    verdicts: Mapping[tuple[str, str], Mapping[str, Any]], n: int = 100, seed: int = 7
) -> list[tuple[str, str]]:
    """``n`` judged pairs for a human to label, stratified over (positive label, any violation) so that
    agreement is measured on every cell rather than on the easy majority."""

    buckets: dict[tuple[int, bool], list[tuple[str, str]]] = defaultdict(list)
    for pair, verdict in verdicts.items():
        violated = any(bool(v) for v in (verdict.get("violations") or {}).values())
        buckets[(int(verdict.get("positive", 0)), violated)].append(pair)
    rng = random.Random(seed)
    for pairs in buckets.values():
        rng.shuffle(pairs)
    picked: list[tuple[str, str]] = []
    order = sorted(buckets)
    while len(picked) < n and any(buckets[b] for b in order):
        for bucket in order:
            if buckets[bucket] and len(picked) < n:
                picked.append(buckets[bucket].pop())
    return picked


def weighted_kappa(a: Sequence[int], b: Sequence[int], categories: Sequence[int] = (0, 1, 2)) -> float | None:
    """Linear-weighted Cohen's kappa for ordinal labels."""

    if not a or len(a) != len(b):
        return None
    cats = list(categories)
    index = {c: i for i, c in enumerate(cats)}
    size = len(cats)
    observed = [[0.0] * size for _ in cats]
    for x, y in zip(a, b):
        observed[index[x]][index[y]] += 1
    total = float(len(a))
    row = [sum(observed[i]) for i in range(size)]
    col = [sum(observed[i][j] for i in range(size)) for j in range(size)]
    weight = [[abs(i - j) / (size - 1) for j in range(size)] for i in range(size)]
    disagreement_obs = sum(weight[i][j] * observed[i][j] for i in range(size) for j in range(size)) / total
    disagreement_exp = sum(weight[i][j] * row[i] * col[j] for i in range(size) for j in range(size)) / (total * total)
    if disagreement_exp == 0:
        return None
    return round(1 - disagreement_obs / disagreement_exp, 4)


def calibration_agreement(human: Sequence[Mapping[str, Any]], judge: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    """Judge versus human on the calibration pairs: positive-label agreement and per-pair violation agreement.

    ``human`` rows carry query_id, novel_id, positive (0/1/2), violations ({neg: bool}), needed_more (bool)."""

    labels_h: list[int] = []
    labels_j: list[int] = []
    violation_pairs = violation_agree = 0
    needed_more = 0
    for row in human:
        verdict = judge.get((str(row["query_id"]), str(row["novel_id"])))
        if verdict is None or row.get("positive") in (None, ""):
            continue
        labels_h.append(int(row["positive"]))
        labels_j.append(int(verdict.get("positive", 0)))
        needed_more += 1 if row.get("needed_more") else 0
        for negative, value in (row.get("violations") or {}).items():
            if value in (None, ""):
                continue
            violation_pairs += 1
            violation_agree += 1 if bool(value) == bool((verdict.get("violations") or {}).get(negative, False)) else 0
    n = len(labels_h)
    return {
        "pairs": n,
        "positive_exact": round(sum(1 for x, y in zip(labels_h, labels_j) if x == y) / n, 4) if n else None,
        "positive_adjacent": round(sum(1 for x, y in zip(labels_h, labels_j) if abs(x - y) <= 1) / n, 4) if n else None,
        "positive_weighted_kappa": weighted_kappa(labels_h, labels_j) if n else None,
        "violation_pairs": violation_pairs,
        "violation_agreement": round(violation_agree / violation_pairs, 4) if violation_pairs else None,
        "needed_more_share": round(needed_more / n, 4) if n else None,
    }
