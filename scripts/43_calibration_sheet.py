"""Candidate-supply bench, step 4: the 100-pair human calibration of the judge.

    uv run python scripts/43_calibration_sheet.py make --verdicts eval/results/retrieval_supply/verdicts_T1_qwen3.8-max.jsonl --tier T1
    uv run python scripts/43_calibration_sheet.py score --sheet eval/results/retrieval_supply/calibration_sheet.csv --verdicts ...

``make`` samples pairs stratified over the judge's labels and writes a CSV with the same evidence
the judge read, blank columns for the human (positive 0/1/2, one column per meta negative, needed_more),
and no judge labels. ``score`` reads the filled sheet back and reports agreement.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import typer
from rich.console import Console

from src.agent.backends import LazyRawText
from src.config import DEFAULT_OUTPUT_PATH
from src.retrieval.cards import DEFAULT_CARDS_PATH, load_cards
from src.retrieval.supply import DEFAULT_SUPPLY_RESULTS_DIR, calibration_agreement, calibration_sample, load_pool, pool_pairs
from src.retrieval.supply_judge import EVIDENCE_TIERS, build_evidence
from src.vector_index import DEFAULT_PROFILES_PATH

app = typer.Typer(add_completion=False)
console = Console()


def _verdicts(path: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            out[(str(record["query_id"]), str(record["novel_id"]))] = record
    return out


@app.command()
def make(
    verdicts: Path = typer.Option(...),
    tier: str = typer.Option("T1", help=f"Evidence shown to the human, same tier as the judge: {sorted(EVIDENCE_TIERS)}"),
    n: int = typer.Option(100),
    seed: int = typer.Option(7),
    pool_dir: Path = typer.Option(DEFAULT_SUPPLY_RESULTS_DIR),
    cards: Path = typer.Option(DEFAULT_CARDS_PATH),
    digests: Path = typer.Option(DEFAULT_PROFILES_PATH),
    inventory: Path = typer.Option(DEFAULT_OUTPUT_PATH),
    out: Path | None = typer.Option(None, help="Defaults to <pool_dir>/calibration_sheet.csv"),
) -> None:
    rows, queries = load_pool(pool_dir)
    by_id = {q.query_id: q for q in queries}
    titles = pool_pairs(rows)
    picked = calibration_sample(_verdicts(verdicts), n=n, seed=seed)
    card_map = load_cards(cards) if cards.exists() else {}
    raw = LazyRawText(inventory) if EVIDENCE_TIERS[tier][0] > 0 else None
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("s41", Path(__file__).with_name("41_supply_judge.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["s41"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    blurbs = module._load_blurbs(digests, {novel_id for _, novel_id in picked})
    negatives = sorted({neg for query_id, _ in picked for neg in by_id[query_id].negatives_meta})
    out = out or pool_dir / "calibration_sheet.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "novel_id", "title", "query", "positives", "negatives_meta", "evidence", "human_positive", *[f"violates:{n}" for n in negatives], "needed_more", "notes"])
        for query_id, novel_id in picked:
            query = by_id[query_id]
            card_text = card_map[novel_id].text() if novel_id in card_map else ""
            raw_text = raw(novel_id) if raw is not None else ""
            evidence = build_evidence(tier, card_text, blurbs.get(novel_id, ""), raw_text or "", novel_id)
            writer.writerow([query_id, novel_id, titles.get((query_id, novel_id), ""), query.query, "、".join(query.positives), "、".join(query.negatives_meta), evidence, "", *["" if n in query.negatives_meta else "n/a" for n in negatives], "", ""])
    console.print(f"{len(picked)} pairs -> {out}  (fill human_positive 0/1/2, violates:* true/false where not n/a, needed_more 1 if you had to open the digest)")


@app.command()
def score(sheet: Path = typer.Option(...), verdicts: Path = typer.Option(...)) -> None:
    human = []
    with sheet.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            violations = {}
            for column, value in row.items():
                if column.startswith("violates:") and value.strip().lower() not in ("", "n/a"):
                    violations[column.split(":", 1)[1]] = value.strip().lower() in ("true", "1", "yes", "是")
            human.append({"query_id": row["query_id"], "novel_id": row["novel_id"], "positive": int(row["human_positive"]) if row["human_positive"].strip() else None, "violations": violations, "needed_more": row.get("needed_more", "").strip() in ("1", "true", "是")})
    report = calibration_agreement(human, _verdicts(verdicts))
    console.print(report)
    Path(sheet).with_name("calibration_agreement.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    app()
