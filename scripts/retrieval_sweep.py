"""Build and benchmark several book-index configurations in one PC run.

A thin driver over 30_build_book_indexes.py and 31_retrieval_bench.py so one
``pc.sh run`` covers a whole sweep. Each configuration is skipped at the build
step when its directory already holds a build_summary.json, so a sweep can be
re-run after a crash without re-encoding.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import typer

from src.config import INDEX_DIR

app = typer.Typer(add_completion=False)

SWEEPS: dict[str, list[dict[str, object]]] = {
    "0p6b": [
        {"name": "single_0p6b", "model": "Qwen/Qwen3-Embedding-0.6B", "dense": "single", "dtype": "fp32", "batch_size": 8},
        {"name": "multi_0p6b", "model": "Qwen/Qwen3-Embedding-0.6B", "dense": "multi", "dtype": "fp32", "batch_size": 128},
    ],
    "4b": [
        {"name": "single_4b", "model": "Qwen/Qwen3-Embedding-4B", "dense": "single", "dtype": "bf16", "batch_size": 4},
        {"name": "multi_4b", "model": "Qwen/Qwen3-Embedding-4B", "dense": "multi", "dtype": "bf16", "batch_size": 32},
    ],
}


def run(argv: list[str]) -> None:
    print(f"[sweep] $ {' '.join(argv)}", flush=True)
    started = time.perf_counter()
    subprocess.run([sys.executable, *argv], check=True)
    print(f"[sweep] done in {time.perf_counter() - started:.0f}s", flush=True)


@app.command()
def main(
    sweep: str = typer.Argument("0p6b", help=f"One of {sorted(SWEEPS)}"),
    device: str = typer.Option("cuda:0"),
    limit: int | None = typer.Option(None, help="Smoke run: first N profiles only."),
    depth: int = typer.Option(1000),
) -> None:
    if sweep not in SWEEPS:
        raise typer.BadParameter(f"sweep must be one of {sorted(SWEEPS)}")
    for config in SWEEPS[sweep]:
        out_dir = INDEX_DIR / str(config["name"]) if limit is None else INDEX_DIR / f"{config['name']}_smoke{limit}"
        if (out_dir / "build_summary.json").exists():
            print(f"[sweep] {out_dir} already built, skipping build", flush=True)
        else:
            argv = [
                "scripts/30_build_book_indexes.py",
                "--out-dir", str(out_dir),
                "--dense", str(config["dense"]),
                "--bm25",
                "--model", str(config["model"]),
                "--device", device,
                "--dtype", str(config["dtype"]),
                "--batch-size", str(config["batch_size"]),
                "--overwrite",
            ]
            if limit is not None:
                argv += ["--limit", str(limit)]
            run(argv)
        run(["scripts/31_retrieval_bench.py", str(out_dir), "--device", device, "--depth", str(depth)])
    results = Path("eval/results/retrieval_bench/results.jsonl")
    if results.exists():
        print("[sweep] results so far:", flush=True)
        for line in results.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            print(json.dumps({k: record[k] for k in ("config", "anchor_hit@10", "anchor_hit@50", "anchor_median_rank", "recall@20_macro")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    app()
