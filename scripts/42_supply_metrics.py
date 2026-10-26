"""Candidate-supply bench, step 3: judged metrics per configuration.

    uv run python scripts/42_supply_metrics.py --verdicts eval/results/retrieval_supply/verdicts_T1_qwen3.8-max.jsonl
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from src.agent.backends import DEFAULT_DENSITY_PATH, load_density_table
from src.retrieval.supply import DEFAULT_SUPPLY_RESULTS_DIR, MIN_FEASIBLE, RECOMMENDED_CONFIG, SUPPLY_DEPTH, SUPPLY_K, format_metrics, judged_metrics, load_pool, recommendation_metrics, request_feasibility

app = typer.Typer(add_completion=False)
console = Console()


def load_verdicts(path: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            out[(str(record["query_id"]), str(record["novel_id"]))] = record
    return out


@app.command()
def main(
    pool_dir: Path = typer.Option(DEFAULT_SUPPLY_RESULTS_DIR),
    verdicts: Path = typer.Option(..., help="verdicts_<tier>_<model>.jsonl written by script 41."),
    density: Path = typer.Option(DEFAULT_DENSITY_PATH),
    k: int = typer.Option(SUPPLY_K, help="Precision / feasible@k cut-off per query."),
    depth: int = typer.Option(SUPPLY_DEPTH, help="How deep the seen results count towards a request's feasibility (the agent may ask for up to 20)."),
    label: str | None = typer.Option(None, help="Row label in metrics_judged.jsonl; defaults to the verdicts file stem."),
) -> None:
    rows, queries = load_pool(pool_dir)
    densities = load_density_table(density) if density.exists() else None
    judged = load_verdicts(verdicts)
    report = judged_metrics(rows, queries, judged, k=k, densities=densities)
    console.print(format_metrics(report))
    record = {"label": label or verdicts.stem, "at": datetime.now(timezone.utc).isoformat(), "k": k, "configs": report}
    by_kind = judged_by_kind(rows, queries, judged, k, densities)
    if by_kind:
        console.print("per kind (agent_seen):")
        console.print(format_metrics(by_kind))
        record["by_kind"] = by_kind
    if any(r.config == RECOMMENDED_CONFIG for r in rows):
        recommended = recommendation_metrics(rows, queries, judged, densities=densities)
        console.print("recommendations:")
        console.print(format_metrics(recommended))
        record["recommendations"] = recommended
        for strict in (False, True):
            feasible = request_feasibility(rows, queries, judged, densities=densities, k=depth, needed=MIN_FEASIBLE, strict=strict)
            path = pool_dir / ("supply_feasible_strict.json" if strict else "supply_feasible.json")
            path.write_text(json.dumps(feasible, ensure_ascii=False, indent=0), encoding="utf-8")
            console.print(f"feasible ({'strict' if strict else 'lenient'}): {sum(feasible.values())}/{len(feasible)} tasks -> {path}")
    out = pool_dir / "metrics_judged.jsonl"
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    console.print(f"-> {out}")


def judged_by_kind(rows, queries, judged, k, densities) -> dict[str, dict]:
    """The judged metrics of the agent_seen config split by task kind; empty when the queries carry no kind."""

    out: dict[str, dict] = {}
    for kind in sorted({q.kind for q in queries if q.kind}):
        subset = [q for q in queries if q.kind == kind]
        ids = {q.query_id for q in subset}
        report = judged_metrics([r for r in rows if r.query_id in ids], subset, judged, k=k, densities=densities)
        for config, metrics in report.items():
            out[f"{kind} / {config}"] = metrics
    return out


if __name__ == "__main__":
    app()
