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
from src.retrieval.supply import DEFAULT_SUPPLY_RESULTS_DIR, SUPPLY_K, format_metrics, judged_metrics, load_pool

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
    k: int = typer.Option(SUPPLY_K),
    label: str | None = typer.Option(None, help="Row label in metrics_judged.jsonl; defaults to the verdicts file stem."),
) -> None:
    rows, queries = load_pool(pool_dir)
    densities = load_density_table(density) if density.exists() else None
    report = judged_metrics(rows, queries, load_verdicts(verdicts), k=k, densities=densities)
    console.print(format_metrics(report))
    out = pool_dir / "metrics_judged.jsonl"
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"label": label or verdicts.stem, "at": datetime.now(timezone.utc).isoformat(), "k": k, "configs": report}, ensure_ascii=False) + "\n")
    console.print(f"-> {out}")


if __name__ == "__main__":
    app()
