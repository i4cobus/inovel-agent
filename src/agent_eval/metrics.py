"""Hard metrics over trajectories: everything here is computable from the redacted
trajectory, the task's expectations, the density table and the final memory state.
No model is consulted. Soft metrics (relevance, trope compliance) are the judge's job.
"""

from __future__ import annotations

import re

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from src.agent_eval.tasks import chapter_number
from src.preferences import TERM_SYNONYMS, constraint_violation_from_densities, is_rule_checkable

FAILURE_LABELS = (
    "format",  # did not end with finish
    "no_recommendation",
    "ungrounded_id",  # recommended a novel_id that never came back from search_books
    "constraint_violated",  # a recommended book fails the density rule for an in-text negative
    "tool_skipped",  # did not run check_term / check_trope on every recommended book
    "memory_missed",  # expected persistent memory entry absent
    "memory_overpersisted",  # one-off constraint written as persistent
    "budget_exhausted",
    "loop",
    "model_error",
)


def recommended_ids(trajectory: Mapping[str, Any]) -> list[str]:
    structured = trajectory.get("structured") or {}
    recs = structured.get("recommendations") or []
    return [str(r.get("novel_id", "")) for r in recs if isinstance(r, Mapping) and r.get("novel_id")]


def tool_calls(trajectory: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for step in trajectory.get("steps", []):
        for call in step.get("tool_calls", []):
            if call.get("name") == name and isinstance(call.get("arguments"), Mapping):
                calls.append(dict(call["arguments"]))
    return calls


def searched_ids(trajectory: Mapping[str, Any]) -> set[str]:
    """novel_ids any search tool put in front of the model (search_books, similar_books; list or {"results": [...]})."""

    ids: set[str] = set()
    for step in trajectory.get("steps", []):
        for obs in step.get("observations", []):
            if obs.get("tool") not in ("search_books", "similar_books"):
                continue
            result = obs.get("result")
            rows = result.get("results") if isinstance(result, Mapping) else result
            if isinstance(rows, list):
                ids.update(str(r.get("novel_id", "")) for r in rows if isinstance(r, Mapping))
    return ids


def checked_terms(trajectory: Mapping[str, Any]) -> dict[str, set[str]]:
    """novel_id -> in-text terms the agent ran check_term on."""

    out: dict[str, set[str]] = {}
    for args in tool_calls(trajectory, "check_term"):
        terms = args.get("terms") or []
        if isinstance(terms, list):
            out.setdefault(str(args.get("novel_id", "")), set()).update(str(t) for t in terms)
    return out


def checked_tropes(trajectory: Mapping[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for args in tool_calls(trajectory, "check_trope"):
        out.setdefault(str(args.get("novel_id", "")), set()).add(str(args.get("trope", "")))
    return out


def _requested_forms(requested: Iterable[str]) -> set[str]:
    """A requested term plus the surface forms check_term merges for it (修仙 → 筑基、金丹 …)."""

    forms: set[str] = set()
    for term in requested:
        term = str(term).strip()
        if term:
            forms.add(term)
            forms.update(TERM_SYNONYMS.get(term, ()))
    return forms


def _is_requested(term: str, forms: set[str]) -> bool:
    term = term.strip()
    return bool(term) and any(term == f or term in f or f in term for f in forms)


def unrequested_checks(trajectory: Mapping[str, Any], requested: Iterable[str]) -> int:
    """check_term / check_trope calls on a term nobody asked for: neither a negative of the session (task
    negatives plus what memory held), nor a surface form of one (the synonym table, or a string that contains
    or is contained in one, 直播间 for 直播), so the only source was the model's own initiative. Seen in the
    first real conversations on 2026-10-12, where 后宫 was checked in a request that never mentioned it."""

    forms = _requested_forms(requested)
    count = 0
    for args in tool_calls(trajectory, "check_term"):
        count += sum(1 for term in (args.get("terms") or []) if not _is_requested(str(term), forms))
    for args in tool_calls(trajectory, "check_trope"):
        if not _is_requested(str(args.get("trope", "")), forms):
            count += 1
    return count


def rule_violations(ids: Iterable[str], terms: list[str], densities: Mapping[str, Mapping[str, float]]) -> dict[str, bool | None]:
    checkable = [t for t in terms if is_rule_checkable(t)]
    if not checkable:
        return {}
    return {
        novel_id: (constraint_violation_from_densities(densities[novel_id], checkable) if novel_id in densities else None)
        for novel_id in ids
    }


@dataclass
class SessionScore:
    termination: str
    steps: int
    prompt_tokens: int
    completion_tokens: int
    n_recs: int
    grounded: bool
    in_corpus: bool
    term_coverage: float | None
    trope_coverage: float | None
    violation_rate: float | None
    unchecked_rate: float | None
    unrequested_checks: int = 0
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "labels": list(self.labels)}


def score_session(
    trajectory: Mapping[str, Any],
    negatives_in_text: list[str],
    negatives_meta: list[str],
    densities: Mapping[str, Mapping[str, float]],
    min_recommendations: int = 1,
    require_recommendations: bool = True,
    remembered: Iterable[str] = (),
) -> SessionScore:
    termination = str(trajectory.get("termination", ""))
    recs = recommended_ids(trajectory)
    labels: list[str] = []
    if termination != "finish":
        labels.append("format")
    if termination == "max_steps":
        labels.append("budget_exhausted")
    if termination in ("loop", "model_error"):
        labels.append(termination)

    searched = searched_ids(trajectory)
    grounded = all(r in searched for r in recs) if recs else True
    in_corpus = all(r in densities for r in recs) if recs else True
    if recs and not grounded:
        labels.append("ungrounded_id")
    if require_recommendations and len(recs) < min_recommendations:
        labels.append("no_recommendation")

    term_cov = trope_cov = violation = unchecked = None
    if recs and negatives_in_text:
        checked = checked_terms(trajectory)
        covered = [all(t in checked.get(r, set()) for t in negatives_in_text) for r in recs]
        term_cov = sum(covered) / len(covered)
        verdicts = rule_violations(recs, negatives_in_text, densities)
        decided = [v for v in verdicts.values() if v is not None]
        violation = (sum(1 for v in decided if v) / len(decided)) if decided else None
        unchecked = sum(1 for v in verdicts.values() if v is None) / len(verdicts)
        if violation:
            labels.append("constraint_violated")
    if recs and negatives_meta:
        checked_t = checked_tropes(trajectory)
        covered = [all(t in checked_t.get(r, set()) for t in negatives_meta) for r in recs]
        trope_cov = sum(covered) / len(covered)
    if recs and ((term_cov is not None and term_cov < 1.0) or (trope_cov is not None and trope_cov < 1.0)):
        labels.append("tool_skipped")

    return SessionScore(
        termination=termination,
        steps=len(trajectory.get("steps", [])),
        prompt_tokens=sum(int(s.get("prompt_tokens", 0)) for s in trajectory.get("steps", [])),
        completion_tokens=sum(int(s.get("completion_tokens", 0)) for s in trajectory.get("steps", [])),
        n_recs=len(recs),
        grounded=grounded,
        in_corpus=in_corpus,
        term_coverage=term_cov,
        trope_coverage=trope_cov,
        violation_rate=violation,
        unchecked_rate=unchecked,
        unrequested_checks=unrequested_checks(trajectory, [*negatives_in_text, *negatives_meta, *remembered]),
        labels=labels,
    )


def memory_labels(expect: Mapping[str, Any], memory_state: Mapping[str, Any]) -> list[str]:
    """Check the memory file against a session's persist / must_not_persist expectations."""

    negatives = memory_state.get("negative") or []
    persistent_values = [str(e.get("value", "")) for e in negatives if e.get("persistent", True)]
    labels: list[str] = []
    for term in expect.get("persist", []):
        if not any(term in value for value in persistent_values):
            labels.append("memory_missed")
            break
    for term in expect.get("must_not_persist", []):
        if any(term in value for value in persistent_values):
            labels.append("memory_overpersisted")
            break
    return labels



# ---------------------------------------------------------------------------
# book_qa: the answer must come from the book and the citations must be real chapters.

ANSWER_SUPPORT_THRESHOLD = 0.25  # provisional; calibrate on dev (share of the answer's CJK bigrams found in the cited chapters)
ABSTAIN_RE = re.compile(r"(未收录|没有收录|不在摘要|摘要(里|中)?(没有|不包含|没收|未)|无法(回答|确认|查到|核实)|没有(找到|查到|看到)(这|那)一?章|不在(收录|可查|覆盖)范围|只收录了|超出(了)?(摘要|收录))")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
QA_LABELS = ("book_unresolved", "no_citation", "ungrounded_citation", "gold_not_cited", "answer_unsupported", "fabricated_answer")


def cited(trajectory: Mapping[str, Any]) -> list[tuple[str, str]]:
    structured = trajectory.get("structured") or {}
    out = []
    for item in structured.get("citations") or []:
        if isinstance(item, dict):
            out.append((str(item.get("novel_id", "")), str(item.get("chapter", "")).strip()))
    return out


def asked_ids(trajectory: Mapping[str, Any]) -> set[str]:
    return {str(c.get("novel_id", "")) for c in tool_calls(trajectory, "ask_book")}


def _norm_heading(heading: str) -> str:
    return re.sub(r"\s+", "", heading)


def match_chapter(citation: str, heading: str) -> bool:
    """Same chapter number when both carry one (「第184章」 ≡ 「第184章 龙济光的计划」); else the same text, spaces ignored."""

    a, b = chapter_number(citation), chapter_number(heading)
    if a is not None and b is not None:
        return a == b
    return _norm_heading(citation) == _norm_heading(heading) or (len(citation) >= 4 and _norm_heading(citation) in _norm_heading(heading))


def cjk_bigrams(text: str) -> set[str]:
    chars = "".join(_CJK_RE.findall(text))
    return {chars[i : i + 2] for i in range(len(chars) - 1)}


def answer_support(answer: str, sources: Iterable[str], exclude: str = "") -> float | None:
    """Share of the answer's CJK bigrams (those not already in the question) that occur in the source texts; None when too short to tell."""

    grams = cjk_bigrams(answer) - cjk_bigrams(exclude)
    if len(grams) < 8:
        return None
    pool = cjk_bigrams("\n".join(sources))
    return round(sum(1 for g in grams if g in pool) / len(grams), 4)


def score_qa_session(trajectory: Mapping[str, Any], expect: Mapping[str, Any], chapters: list[tuple[str, str, str]], user_message: str = "") -> dict[str, Any]:
    """One book_qa session. ``chapters`` are the digest's (kind, heading, text) for the gold book."""

    termination = str(trajectory.get("termination", ""))
    labels: list[str] = []
    if termination != "finish":
        labels.append("format")
    if termination == "max_steps":
        labels.append("budget_exhausted")
    if termination in ("loop", "model_error"):
        labels.append(termination)

    novel_id = str(expect.get("novel_id", ""))
    answerable = bool(expect.get("answerable", True))
    gold = [str(g) for g in expect.get("gold_chapters", [])]
    answer = str(trajectory.get("final_answer") or "")
    headings = [h for _, h, _ in chapters]
    texts = {h: t for _, h, t in chapters}

    resolved = novel_id in asked_ids(trajectory)
    if not resolved:
        labels.append("book_unresolved")

    citations = cited(trajectory)
    grounded_headings: list[str] = []
    all_grounded = True
    for cited_id, chapter in citations:
        hit = next((h for h in headings if match_chapter(chapter, h)), None) if cited_id == novel_id else None
        if hit is None:
            all_grounded = False
        else:
            grounded_headings.append(hit)
    if citations and not all_grounded:
        labels.append("ungrounded_citation")

    gold_cited: bool | None = None
    support: float | None = None
    abstained: bool | None = None
    if answerable:
        if not citations:
            labels.append("no_citation")
        gold_cited = any(any(match_chapter(c, g) for g in gold) for _, c in citations)
        if citations and not gold_cited:
            labels.append("gold_not_cited")
        sources = [texts[h] for h in grounded_headings] or [texts[h] for h in headings if any(match_chapter(h, g) for g in gold)]
        support = answer_support(answer, sources, exclude=user_message)
        if support is not None and support < ANSWER_SUPPORT_THRESHOLD:
            labels.append("answer_unsupported")
    else:
        abstained = bool(ABSTAIN_RE.search(answer))
        if not abstained:
            labels.append("fabricated_answer")

    return {
        "termination": termination,
        "steps": len(trajectory.get("steps", [])),
        "prompt_tokens": sum(int(s.get("prompt_tokens", 0)) for s in trajectory.get("steps", [])),
        "completion_tokens": sum(int(s.get("completion_tokens", 0)) for s in trajectory.get("steps", [])),
        "resolved": resolved,
        "n_citations": len(citations),
        "citations_grounded": all_grounded if citations else None,
        "gold_cited": gold_cited,
        "answer_support": support,
        "abstained": abstained,
        "labels": labels,
    }


def referenced_ids(trajectory: Mapping[str, Any]) -> set[str]:
    """Books the agent looked at in this session: get_profile, ask_book, check_term, check_trope, similar_books."""

    ids = set()
    for name in ("get_profile", "ask_book", "check_term", "check_trope", "similar_books"):
        ids |= {str(c.get("novel_id", "")) for c in tool_calls(trajectory, name)}
    return ids


def session_state_labels(expect: Mapping[str, Any], trajectory: Mapping[str, Any], trajectories: list[Mapping[str, Any]]) -> list[str]:
    """Cross-session checks for session_state follow-ups; the earlier session's recommendations are the reference."""

    labels: list[str] = []
    recs = recommended_ids(trajectory)
    if "disjoint_from" in expect:
        earlier = recommended_ids(trajectories[int(expect["disjoint_from"])]) if len(trajectories) > int(expect["disjoint_from"]) else []
        if set(recs) & set(earlier):
            labels.append("repeat_recommended")
    if "refers_to" in expect:
        earlier = recommended_ids(trajectories[0]) if trajectories else []
        position = int(expect["refers_to"])
        if len(earlier) >= position and earlier[position - 1] not in referenced_ids(trajectory):
            labels.append("reference_missed")
    if "exclude_index" in expect:
        earlier = recommended_ids(trajectories[0]) if trajectories else []
        position = int(expect["exclude_index"])
        if len(earlier) >= position and earlier[position - 1] in recs:
            labels.append("excluded_recommended")
    return labels


def score_task(
    task: Mapping[str, Any],
    trajectories: list[Mapping[str, Any]],
    memory_states: list[Mapping[str, Any]],
    densities: Mapping[str, Mapping[str, float]],
    digests: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """One row per task: per-session scores, memory checks, and the task-level pass flag."""

    sessions = task["sessions"]
    rows: list[dict[str, Any]] = []
    labels: list[str] = []
    for session, trajectory, memory_state in zip(sessions, trajectories, memory_states):
        expect = session.get("expect", {})
        if task["kind"] == "clarify":
            score = score_session(trajectory, [], [], densities, require_recommendations=False)
            structured = trajectory.get("structured") or {}
            if not structured.get("asks_user"):
                score.labels.append("no_clarification")
            if score.n_recs > int(expect.get("max_recommendations", 0)):
                score.labels.append("recommended_without_asking")
            labels.extend(score.labels)
            rows.append(score.to_dict())
            continue
        if task["kind"] == "session_state":
            min_recs = int(expect.get("min_recommendations", 0))
            score = score_session(trajectory, [], [], densities, min_recommendations=max(min_recs, 1), require_recommendations=min_recs > 0)
            score.labels.extend(session_state_labels(expect, trajectory, trajectories))
            labels.extend(score.labels)
            rows.append(score.to_dict())
            continue
        if task["kind"] == "book_qa":
            book = (digests or {}).get(str(expect.get("novel_id", "")))
            if book is None:
                raise KeyError(f"no digest for {expect.get('novel_id')} (pass the digest table to score book_qa tasks)")
            row = score_qa_session(trajectory, expect, book["chapters"], user_message=str(session.get("user_message", "")))
            labels.extend(row["labels"])
            rows.append(row)
            continue
        if task["kind"] == "memory":
            # Memory sessions state their own expectations: a one-off constraint must not be demanded of the next
            # session (it fell back to the task-level terms until 2026-10-12, which scored the oneoff variant's second
            # session as tool_skipped by construction). A term the user persists in this very message is also binding now.
            must_pass = list(expect.get("must_pass", expect.get("persist", [])))
        else:
            must_pass = task.get("negatives_in_text", [])
        meta = task.get("negatives_meta", []) if task["kind"] == "constrained_rec" else []
        min_recs = int(expect.get("min_recommendations", 0))
        remembered = [str(e.get("value", "")) for kind in ("negative", "positive") for e in (memory_state.get(kind) or []) if isinstance(e, dict)]
        remembered += [t for s in sessions for t in (s.get("expect", {}).get("persist", []) + s.get("expect", {}).get("must_pass", []))]
        score = score_session(trajectory, must_pass, meta, densities, min_recommendations=max(min_recs, 1), require_recommendations=min_recs > 0, remembered=remembered)
        mem = memory_labels(expect, memory_state)
        score.labels.extend(mem)
        labels.extend(score.labels)
        rows.append(score.to_dict())
    if len(trajectories) < len(sessions):
        labels.append("model_error")
    passed = not labels
    return {
        "task_id": task["task_id"],
        "kind": task["kind"],
        "variant": task.get("variant", ""),
        "split": task.get("split", ""),
        "pass": passed,
        "labels": sorted(set(labels)),
        "sessions": rows,
        "steps": sum(r["steps"] for r in rows),
        "unrequested_checks": sum(r.get("unrequested_checks", 0) for r in rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
    }


def aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Pass rates and label counts, overall and per kind / variant."""

    def summarise(subset: list[Mapping[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {"n": 0}
        session_rows = [s for r in subset for s in r["sessions"]]
        violation = [s["violation_rate"] for s in session_rows if s.get("violation_rate") is not None]
        term_cov = [s["term_coverage"] for s in session_rows if s.get("term_coverage") is not None]
        trope_cov = [s["trope_coverage"] for s in session_rows if s.get("trope_coverage") is not None]
        qa_rows = [s for s in session_rows if "resolved" in s]
        grounded = [s["citations_grounded"] for s in qa_rows if s.get("citations_grounded") is not None]
        gold_cited = [s["gold_cited"] for s in qa_rows if s.get("gold_cited") is not None]
        support = [s["answer_support"] for s in qa_rows if s.get("answer_support") is not None]
        abstained = [s["abstained"] for s in qa_rows if s.get("abstained") is not None]

        def rate(values: list[Any]) -> float | None:
            return round(sum(1 for v in values if v) / len(values), 4) if values else None

        return {
            "n": len(subset),
            "pass_rate": round(sum(1 for r in subset if r["pass"]) / len(subset), 4),
            "finish_rate": round(sum(1 for s in session_rows if s["termination"] == "finish") / len(session_rows), 4),
            "mean_steps_per_session": round(sum(s["steps"] for s in session_rows) / len(session_rows), 2),
            "mean_prompt_tokens_per_session": round(sum(s["prompt_tokens"] for s in session_rows) / len(session_rows)),
            "violation_rate_macro": round(sum(violation) / len(violation), 4) if violation else None,
            "check_term_coverage": round(sum(term_cov) / len(term_cov), 4) if term_cov else None,
            "check_trope_coverage": round(sum(trope_cov) / len(trope_cov), 4) if trope_cov else None,
            "unrequested_checks_per_session": round(sum(s.get("unrequested_checks", 0) for s in session_rows) / len(session_rows), 2),
            "book_resolved_rate": rate([s["resolved"] for s in qa_rows]) if qa_rows else None,
            "citation_grounded_rate": rate(grounded),
            "gold_cited_rate": rate(gold_cited),
            "answer_support_mean": round(sum(support) / len(support), 4) if support else None,
            "abstain_rate": rate(abstained),
            "labels": dict(Counter(label for r in subset for label in r["labels"])),
        }

    out = {"overall": summarise(list(rows))}
    for key in ("kind", "variant"):
        values = sorted({str(r.get(key, "")) for r in rows})
        out[key] = {value: summarise([r for r in rows if str(r.get(key, "")) == value]) for value in values}
    return out


def format_summary(summary: Mapping[str, Any]) -> str:
    columns = ["n", "pass_rate", "finish_rate", "mean_steps_per_session", "mean_prompt_tokens_per_session", "violation_rate_macro", "check_term_coverage", "check_trope_coverage", "unrequested_checks_per_session"]
    if any(summary["overall"].get(c) is not None for c in ("book_resolved_rate", "citation_grounded_rate", "gold_cited_rate", "answer_support_mean", "abstain_rate")):
        columns += ["book_resolved_rate", "citation_grounded_rate", "gold_cited_rate", "answer_support_mean", "abstain_rate"]
    lines = ["| group | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    lines.append("| overall | " + " | ".join(str(summary["overall"].get(c)) for c in columns) + " |")
    for key in ("kind", "variant"):
        for name, stats in summary.get(key, {}).items():
            lines.append(f"| {key}={name} | " + " | ".join(str(stats.get(c)) for c in columns) + " |")
    labels = summary["overall"].get("labels", {})
    if labels:
        lines.append("")
        lines.append("failure labels: " + ", ".join(f"{k}={v}" for k, v in sorted(labels.items(), key=lambda kv: -kv[1])))
    return "\n".join(lines)
