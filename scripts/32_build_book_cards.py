"""Build book cards with a local model, resumably, and report how it used the vocabulary.

    uv run python scripts/32_build_book_cards.py --only-ids eval/agent/cards_pilot_ids.txt --workers 2   # pilot
    uv run python scripts/32_build_book_cards.py --report                                                 # vocabulary usage

Every card is appended to data/cache/book_cards.jsonl as it is produced, so a
killed run resumes where it stopped; the parquet is rewritten at the end and
every --checkpoint cards.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console

from src.agent.backends import DEFAULT_CHAT_BASE_URL
from src.chat_transport import HTTPChatTransport
from src.config import PROJECT_ROOT
from src.retrieval.cards import DEFAULT_CARD_CACHE_PATH, DEFAULT_CARD_MAX_CHARS, DEFAULT_CARDS_PATH, CardBuilder, cards_to_frame, load_cards, vocabulary_report
from src.vector_index import DEFAULT_PROFILES_PATH

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    model: str = typer.Option("qwen3.5:9b"),
    base_url: str = typer.Option(DEFAULT_CHAT_BASE_URL),
    reasoning_effort: str | None = typer.Option("none", help="Ollama: 'none' disables thinking; pass 'off' to send nothing (hosted APIs)."),
    extra_body: str = typer.Option("", help='JSON merged into every request body, e.g. {"enable_thinking": false} for Bailian.'),
    profiles: Path = typer.Option(DEFAULT_PROFILES_PATH),
    out: Path = typer.Option(DEFAULT_CARDS_PATH),
    cache: Path = typer.Option(DEFAULT_CARD_CACHE_PATH),
    limit: int | None = typer.Option(None, help="First N profiles (pilot)."),
    only_ids: Path | None = typer.Option(None, help="Text file of novel_ids to build, one per line."),
    exclude_ids: Path | None = typer.Option(None, help="Text file of novel_ids to skip."),
    sample: int | None = typer.Option(None, help="Random sample of N books (after --only-ids / --exclude-ids)."),
    seed: int = typer.Option(7, help="Seed for --sample."),
    workers: int = typer.Option(2, help="Concurrent requests to the model server."),
    max_chars: int = typer.Option(DEFAULT_CARD_MAX_CHARS, help="Digest characters sent per book (default: the whole digest)."),
    max_tokens: int = typer.Option(1200, help="Output budget per card; a card cut off here fails to parse."),
    checkpoint: int = typer.Option(200, help="Rewrite the parquet every N cards."),
    report: bool = typer.Option(False, help="Only report vocabulary usage of the existing parquet."),
    reparse: bool = typer.Option(False, help="Re-parse the cached raw responses of the selected books with the current parser (no model calls) and rewrite the parquet."),
) -> None:
    if report:
        _report(out)
        return
    import pandas as pd

    frame = pd.read_parquet(profiles, columns=["novel_id", "profile_text"])
    if only_ids is not None:
        wanted = {line.strip() for line in only_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
        frame = frame[frame["novel_id"].astype(str).isin(wanted)]
    if exclude_ids is not None:
        skip = {line.strip() for line in exclude_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
        frame = frame[~frame["novel_id"].astype(str).isin(skip)]
    if sample is not None:
        frame = frame.sample(n=min(sample, len(frame)), random_state=seed).sort_index()
    if limit is not None:
        frame = frame.head(limit)
    items = [(str(r.novel_id), str(r.profile_text)) for r in frame.itertuples(index=False)]
    # Ollama switches thinking off through reasoning_effort; Bailian's OpenAI-compatible endpoint
    # wants {"enable_thinking": false} in the body instead (--extra-body), with no reasoning_effort.
    effort = reasoning_effort if reasoning_effort not in (None, "", "off") else None
    body = json.loads(extra_body) if extra_body else {}
    transport = HTTPChatTransport(model=model, base_url=base_url, reasoning_effort=effort, extra_body=body, timeout=300.0)
    builder = CardBuilder(transport, model, cache_path=cache, max_chars=max_chars, max_tokens=max_tokens)
    from src.retrieval.cards import card_cache_key

    already = sum(1 for novel_id, _ in items if card_cache_key(novel_id, model) in builder.cache)
    console.print(f"books: {len(items)}  cached: {already}  model: {model}  workers: {workers}")
    if reparse:
        count = builder.reparse(items)
        cards = [builder.build_one(novel_id, text) for novel_id, text in items if card_cache_key(novel_id, model) in builder.cache]
        _write(cards, out)
        console.print(f"reparsed {count} cached responses; wrote {len(cards)} cards -> {out}")
        return

    done: list = []
    started = time.perf_counter()
    errors = 0

    def on_result(card) -> None:
        nonlocal errors
        done.append(card)
        if card.error:
            errors += 1
        n = len(done)
        if n % 25 == 0 or n == len(items):
            elapsed = time.perf_counter() - started
            console.print(f"  {n}/{len(items)}  {elapsed / max(n - already, 1):.1f}s/new card  errors={errors}")
        if n % checkpoint == 0:
            _write(done, out)

    cards = builder.build_many(items, workers=workers, on_result=on_result)
    _write(cards, out)
    console.print(f"wrote {len(cards)} cards ({errors} with errors) -> {out} in {time.perf_counter() - started:.0f}s")
    if builder.failures:
        errors_path = out.with_name(out.stem + "_errors.jsonl")
        with errors_path.open("w", encoding="utf-8") as handle:
            for failure in builder.failures:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        kinds: dict[str, int] = {}
        for failure in builder.failures:
            kind = failure["error"].split(":")[0]
            kinds[kind] = kinds.get(kind, 0) + 1
        console.print(f"failures by kind: {kinds} -> {errors_path}")


def _write(cards: list, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cards_to_frame(c for c in cards if not c.error).to_parquet(out, index=False)


def _report(cards_path: Path) -> None:
    cards = load_cards(cards_path)
    summary = vocabulary_report(cards)
    out = cards_path.with_name(cards_path.stem + "_vocabulary.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    console.print(f"cards {summary['cards']}  elements/card {summary['elements_per_card']}  distinct keywords {summary['keywords_distinct']}")
    console.print("genres:", summary["genres"])
    console.print("dropped (outside the vocabulary):", summary["dropped"])
    console.print("elements the text did not back:", summary["unsupported_elements"])
    console.print(f"-> {out}")


if __name__ == "__main__":
    app()
