"""Candidate-supply bench, step 2: judge the pool (costs API money; --dry-run first).

    uv run python scripts/41_supply_judge.py --dry-run --tier T1
    uv run python scripts/41_supply_judge.py --tier T4 --sample 150 --cap 15 --api-key-file ~/.config/aliyun.key

Evidence per tier comes from the cards, the digest's synopsis and (T2–T4) chapters the digest did
not use, read from the raw text via the inventory. Verdicts are cached; a rerun only pays for new
pairs. --cap is the hard spend ceiling in the price unit you pass (CNY for Bailian list prices).
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from src.agent.backends import DEFAULT_DENSITY_PATH, LazyRawText, load_density_table
from src.chat_transport import HTTPChatTransport
from src.config import DEFAULT_OUTPUT_PATH
from src.judge import BudgetGuard, PricePerMillion
from src.retrieval.cards import DEFAULT_CARDS_PATH, load_cards
from src.retrieval.supply import DEFAULT_SUPPLY_RESULTS_DIR, SUPPLY_DEPTH, load_pool, pool_pairs
from src.retrieval.supply_judge import (
    DEFAULT_SUPPLY_JUDGE_CACHE,
    DEFAULT_SUPPLY_JUDGE_MODEL,
    EVIDENCE_TIERS,
    SupplyJudgeTask,
    build_evidence,
    density_flags,
    estimate_prompt_tokens,
    run_supply_judgements,
)
from src.vector_index import DEFAULT_PROFILES_PATH

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    pool_dir: Path = typer.Option(DEFAULT_SUPPLY_RESULTS_DIR),
    tier: str = typer.Option("T1", help=f"Evidence tier: {sorted(EVIDENCE_TIERS)}"),
    model: str = typer.Option(DEFAULT_SUPPLY_JUDGE_MODEL),
    base_url: str = typer.Option("", help="OpenAI-compatible endpoint; INOVELREC_LLM_BASE_URL when empty."),
    api_key_file: Path | None = typer.Option(None),
    price_in: float = typer.Option(12.0, help="Price per million input tokens (Qwen3.8-Max list: 12 CNY)."),
    price_out: float = typer.Option(36.0),
    cap: float = typer.Option(20.0, help="Hard spend ceiling, same unit as the prices."),
    depth: int = typer.Option(SUPPLY_DEPTH),
    sample: int | None = typer.Option(None, help="Judge a random sample of this many pairs (pilot)."),
    seed: int = typer.Option(7),
    only_source: str | None = typer.Option(None, help="Restrict to queries of one source: task | rewrite | trajectory | legacy."),
    include_legacy: bool = typer.Option(False, help="Also judge the v1 queries' pairs; by default they only serve the anchor metric (a 0.6B smoke pool had 6,593 pairs with them, ~60% more)."),
    cards: Path = typer.Option(DEFAULT_CARDS_PATH),
    digests: Path = typer.Option(DEFAULT_PROFILES_PATH),
    inventory: Path = typer.Option(DEFAULT_OUTPUT_PATH, help="For raw-text windows (T2–T4)."),
    density: Path = typer.Option(DEFAULT_DENSITY_PATH),
    cache: Path = typer.Option(DEFAULT_SUPPLY_JUDGE_CACHE),
    workers: int = typer.Option(6),
    dry_run: bool = typer.Option(False, help="Build the tasks, print pair counts and the token / cost estimate, call nothing."),
) -> None:
    import os

    rows, queries = load_pool(pool_dir)
    by_id = {q.query_id: q for q in queries}
    pairs = pool_pairs(rows, depth)
    if only_source:
        pairs = {pair: title for pair, title in pairs.items() if by_id[pair[0]].source == only_source}
    elif not include_legacy:
        pairs = {pair: title for pair, title in pairs.items() if by_id[pair[0]].source != "legacy"}
    keys = sorted(pairs)
    if sample is not None and sample < len(keys):
        keys = random.Random(seed).sample(keys, sample)
    console.print(f"pairs: {len(keys)} (of {len(pairs)} in the pool at depth {depth}), tier {tier}")

    novel_ids = {novel_id for _, novel_id in keys}
    card_map = load_cards(cards) if cards.exists() else {}
    blurbs = _load_blurbs(digests, novel_ids)
    densities = load_density_table(density) if density.exists() else {}
    raw = LazyRawText(inventory) if EVIDENCE_TIERS[tier][0] > 0 else None
    tasks: list[SupplyJudgeTask] = []
    evidence_cache: dict[str, str] = {}
    for query_id, novel_id in keys:
        query = by_id[query_id]
        if novel_id not in evidence_cache:
            card_text = card_map[novel_id].text() if novel_id in card_map else ""
            raw_text = raw(novel_id) if raw is not None else ""
            evidence_cache[novel_id] = build_evidence(tier, card_text, blurbs.get(novel_id, ""), raw_text or "", novel_id)
        tasks.append(
            SupplyJudgeTask(
                query_id=query_id,
                query=query.query,
                novel_id=novel_id,
                title=pairs[(query_id, novel_id)],
                evidence=evidence_cache[novel_id],
                tier=tier,
                positives=query.positives,
                negatives_meta=query.negatives_meta,
                density_flags=density_flags(densities.get(novel_id), query.negatives_in_text),
            )
        )
    prompt_tokens = sum(estimate_prompt_tokens(t) for t in tasks)
    est_cost = (prompt_tokens * price_in + len(tasks) * 150 * price_out) / 1_000_000
    without_card = sum(1 for t in tasks if "【书卡" not in t.evidence)
    console.print(f"estimated input tokens {prompt_tokens:,} (~{prompt_tokens // max(len(tasks), 1):,}/pair), cost ≈ {est_cost:.1f} at {price_in}/{price_out} per M; pairs without a card: {without_card}")
    if dry_run:
        return

    api_key = api_key_file.read_text(encoding="utf-8").strip() if api_key_file else None
    transport = HTTPChatTransport(model=model, base_url=base_url or os.environ.get("INOVELREC_LLM_BASE_URL", ""), api_key=api_key, extra_body={"enable_thinking": False}, timeout=180.0)
    budget = BudgetGuard(limit_usd=cap, prices=PricePerMillion(price_in, price_out))
    done = 0

    def on_result(task: SupplyJudgeTask, verdict: dict) -> None:
        nonlocal done
        done += 1
        if done % 50 == 0:
            console.print(f"  {done}/{len(tasks)} spent {budget.spent_usd:.2f}")

    verdicts, summary = run_supply_judgements(tasks, transport, model, budget, cache_path=cache, workers=workers, on_result=on_result)
    out = pool_dir / f"verdicts_{tier}_{model}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for (query_id, novel_id), verdict in sorted(verdicts.items()):
            handle.write(json.dumps({"query_id": query_id, "novel_id": novel_id, **verdict}, ensure_ascii=False) + "\n")
    (pool_dir / f"judge_summary_{tier}_{model}.json").write_text(json.dumps({"at": datetime.now(timezone.utc).isoformat(), **summary.to_dict()}, ensure_ascii=False, indent=1), encoding="utf-8")
    console.print(summary.to_dict())
    console.print(f"-> {out}")


def _load_blurbs(digests: Path, novel_ids: set[str]) -> dict[str, str]:
    """novel_id -> the digest's synopsis section (header stripped), for the ids needed."""

    import pyarrow.parquet as pq

    from src.retrieval.multivector import synopsis_preview

    if not digests.exists():
        return {}
    out: dict[str, str] = {}
    for batch in pq.ParquetFile(digests).iter_batches(batch_size=512, columns=["novel_id", "sections_json"]):
        for record in batch.to_pylist():
            novel_id = str(record["novel_id"])
            if novel_id not in novel_ids:
                continue
            for section in json.loads(record["sections_json"] or "[]"):
                if section.get("kind") == "blurb":
                    out[novel_id] = synopsis_preview(str(section["text"]), 800)
                    break
    return out


if __name__ == "__main__":
    app()
