"""Split the judge-labelled books into a validation half and a held-out half, and pick the card pilot.

The v1 judge labelled 753 (trope, book) pairs over 646 books. Books are split by a
stable hash of novel_id; the pilot takes the validation-half books with the most
positive (violated) labels first, so per-trope precision and recall are measurable
on 120 cards. The held-out half is never used to tune the card prompt.

    uv run python scripts/33b_select_card_pilot.py --size 120
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import typer
from rich.console import Console

from src.config import PROJECT_ROOT
from src.preferences import META_LABEL_NEGATIVES

app = typer.Typer(add_completion=False)
console = Console()
SPLIT_PATH = PROJECT_ROOT / "eval" / "agent" / "cards_label_split.json"
PILOT_PATH = PROJECT_ROOT / "eval" / "agent" / "cards_pilot_ids.txt"


def load_judge_labels() -> dict[str, dict[str, bool]]:
    """trope -> {novel_id -> judge said the book violates the trope}; single-meta-negative queries only."""

    queries = {}
    with (PROJECT_ROOT / "eval" / "eval_queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                queries[record["query_id"]] = record
    labels: dict[str, dict[str, bool]] = {}
    for csv_path in sorted((PROJECT_ROOT / "eval" / "results").rglob("eval_results_judged*.csv")):
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                verdict = row.get("judge_constraint_violation", "")
                if verdict == "":
                    continue
                unwanted = queries[row["query_id"]]["unwanted"]
                metas = [t for t in unwanted if t in META_LABEL_NEGATIVES]
                if len(metas) == 1 and len(unwanted) == 1:
                    labels.setdefault(metas[0], {})[row["novel_id"]] = verdict == "True"
    return labels


def split_books(novel_ids: set[str]) -> dict[str, list[str]]:
    validate = sorted(n for n in novel_ids if int(hashlib.md5(f"cards:{n}".encode()).hexdigest(), 16) % 2 == 0)
    heldout = sorted(novel_ids - set(validate))
    return {"validate": validate, "heldout": heldout}


@app.command()
def main(size: int = typer.Option(120), split_out: Path = typer.Option(SPLIT_PATH), pilot_out: Path = typer.Option(PILOT_PATH)) -> None:
    labels = load_judge_labels()
    per_book: dict[str, list[tuple[str, bool]]] = {}
    for trope, books in labels.items():
        for novel_id, violated in books.items():
            per_book.setdefault(novel_id, []).append((trope, violated))
    split = split_books(set(per_book))
    # Half the pilot from the books with the most positive labels, half from books with the most
    # negative labels, so both precision and false-positive rate have something to stand on.
    by_pos = sorted(split["validate"], key=lambda n: (-sum(v for _, v in per_book[n]), -len(per_book[n]), n))
    by_neg = sorted(split["validate"], key=lambda n: (-sum(not v for _, v in per_book[n]), -len(per_book[n]), n))
    pilot: list[str] = []
    for n in by_pos[: size // 2]:
        pilot.append(n)
    for n in by_neg:
        if len(pilot) >= size:
            break
        if n not in pilot:
            pilot.append(n)
    split_out.parent.mkdir(parents=True, exist_ok=True)
    split_out.write_text(json.dumps({**split, "pilot": pilot, "rule": "md5('cards:'+novel_id) even -> validate"}, ensure_ascii=False, indent=1), encoding="utf-8")
    pilot_out.write_text("\n".join(pilot) + "\n", encoding="utf-8")
    pairs = lambda ids: sum(len(per_book[n]) for n in ids)  # noqa: E731
    positives = lambda ids: sum(v for n in ids for _, v in per_book[n])  # noqa: E731
    console.print(f"labelled books {len(per_book)}, pairs {pairs(per_book)}, positives {positives(per_book)}")
    console.print(f"validate {len(split['validate'])} books / {pairs(split['validate'])} pairs / {positives(split['validate'])} positives")
    console.print(f"heldout  {len(split['heldout'])} books / {pairs(split['heldout'])} pairs / {positives(split['heldout'])} positives")
    console.print(f"pilot    {len(pilot)} books / {pairs(pilot)} pairs / {positives(pilot)} positives -> {pilot_out}")
    covered: dict[str, list[int]] = {}
    for n in pilot:
        for trope, v in per_book[n]:
            covered.setdefault(trope, [0, 0])[0] += 1
            covered[trope][1] += int(v)
    console.print("pilot per trope (n/positives): " + ", ".join(f"{t} {a}/{b}" for t, (a, b) in sorted(covered.items(), key=lambda kv: -kv[1][0])))


if __name__ == "__main__":
    app()
