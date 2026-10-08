"""Stage 2: one digest per novel (opening + chapter titles + middle windows + ending).

    uv run python scripts/02_build_digests.py --overwrite --max-workers 16
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.config import DEFAULT_OUTPUT_PATH
from src.digest import DEFAULT_DIGEST_PATH, build_digests

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    inventory: Path = typer.Option(DEFAULT_OUTPUT_PATH),
    out: Path = typer.Option(DEFAULT_DIGEST_PATH),
    limit: int | None = typer.Option(None),
    overwrite: bool = typer.Option(False),
    max_workers: int | None = typer.Option(None),
) -> None:
    if out.exists() and not overwrite:
        raise typer.BadParameter(f"{out} exists; pass --overwrite")
    result = build_digests(inventory_path=inventory, limit=limit, max_workers=max_workers)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.frame.to_parquet(out, index=False)
    table = Table(title="Digest build")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("digests written", str(len(result.frame)))
    for reason, count in result.skipped.items():
        table.add_row(f"skipped {reason}", str(count))
    table.add_row("ZXCS boilerplate detected", str(result.boilerplate_detected))
    table.add_row("ZXCS lines removed", str(result.boilerplate_lines_removed))
    if not result.frame.empty:
        table.add_row("avg digest chars", f"{result.frame['profile_text'].str.len().mean():.0f}")
        table.add_row("avg chapters", f"{result.frame['estimated_chapter_count'].mean():.0f}")
        table.add_row("with ending marker", str(int((result.frame['ending_status'] == '完结标记').sum())))
        table.add_row("chapter-structured", str(int((result.frame['used_chapter_indices'] != '[]').sum())))
    console.print(table)
    console.print(f"-> {out}")


if __name__ == "__main__":
    app()
