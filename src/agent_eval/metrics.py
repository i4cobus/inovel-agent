"""Hard metrics over trajectories: everything here is computable from the redacted
trajectory, the task's expectations, the density table and the final memory state.
No model is consulted. Soft metrics (relevance, trope compliance) are the judge's job.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from src.preferences import constraint_violation_from_densities, is_rule_checkable

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
    ids: set[str] = set()
    for step in trajectory.get("steps", []):
        for obs in step.get("observations", []):
            if obs.get("tool") == "search_books" and isinstance(obs.get("result"), list):
                ids.update(str(r.get("novel_id", "")) for r in obs["result"] if isinstance(r, Mapping))
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


def score_task(task: Mapping[str, Any], trajectories: list[Mapping[str, Any]], memory_states: list[Mapping[str, Any]], densities: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """One row per task: per-session scores, memory checks, and the task-level pass flag."""

    sessions = task["sessions"]
    rows: list[dict[str, Any]] = []
    labels: list[str] = []
    for session, trajectory, memory_state in zip(sessions, trajectories, memory_states):
        expect = session.get("expect", {})
        must_pass = list(expect.get("must_pass", task.get("negatives_in_text", []))) if task["kind"] == "memory" else task.get("negatives_in_text", [])
        meta = task.get("negatives_meta", []) if task["kind"] == "constrained_rec" else []
        min_recs = int(expect.get("min_recommendations", 0))
        score = score_session(trajectory, must_pass, meta, densities, min_recommendations=max(min_recs, 1), require_recommendations=min_recs > 0)
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
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
    }


def aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Pass rates and label counts, overall and per kind / variant."""

    def summarise(subset: list[Mapping[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {"n": 0}
        session_rows = [s for r in subset for s in r["sessions"]]
        violation = [s["violation_rate"] for s in session_rows if s["violation_rate"] is not None]
        term_cov = [s["term_coverage"] for s in session_rows if s["term_coverage"] is not None]
        trope_cov = [s["trope_coverage"] for s in session_rows if s["trope_coverage"] is not None]
        return {
            "n": len(subset),
            "pass_rate": round(sum(1 for r in subset if r["pass"]) / len(subset), 4),
            "finish_rate": round(sum(1 for s in session_rows if s["termination"] == "finish") / len(session_rows), 4),
            "mean_steps_per_session": round(sum(s["steps"] for s in session_rows) / len(session_rows), 2),
            "mean_prompt_tokens_per_session": round(sum(s["prompt_tokens"] for s in session_rows) / len(session_rows)),
            "violation_rate_macro": round(sum(violation) / len(violation), 4) if violation else None,
            "check_term_coverage": round(sum(term_cov) / len(term_cov), 4) if term_cov else None,
            "check_trope_coverage": round(sum(trope_cov) / len(trope_cov), 4) if trope_cov else None,
            "labels": dict(Counter(label for r in subset for label in r["labels"])),
        }

    out = {"overall": summarise(list(rows))}
    for key in ("kind", "variant"):
        values = sorted({str(r.get(key, "")) for r in rows})
        out[key] = {value: summarise([r for r in rows if str(r.get(key, "")) == value]) for value in values}
    return out


def format_summary(summary: Mapping[str, Any]) -> str:
    columns = ["n", "pass_rate", "finish_rate", "mean_steps_per_session", "mean_prompt_tokens_per_session", "violation_rate_macro", "check_term_coverage", "check_trope_coverage"]
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
