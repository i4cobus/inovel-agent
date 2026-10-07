# inovel-agent: a conversational agent over a Chinese web-novel corpus, and how to evaluate it

> English summary. The Chinese [`README.md`](README.md) is the primary one, and all
> other documents are in Chinese.
>
> Since October 2026 this repository has moved from "post-training a web-novel
> reranker with SFT + GRPO" to "a web-novel conversational agent plus an agent
> evaluation suite". The design and every decision with its alternatives are in
> [`docs/agent-plan.md`](docs/agent-plan.md). The previous system is fully
> described in [`docs/v1-reranker/`](docs/v1-reranker/README.md) and its code
> is at tag `v1-reranker`.

Corpus: 7,653 Chinese web novels, about 36 GB of plain text, private, not in git.

## What is here now

| path | contents |
|---|---|
| `src/` | the retrieval primitives that survived (ingest, profiles, embedding, FAISS index, single-query search), the constraint rule (`preferences.py`), evidence sampling (`evidence.py`), the judge and its calibration (`judge.py`, `evaluation.py`), an OpenAI-compatible transport (`chat_transport.py`) |
| `src/agent/` | the agent loop, six tools, single-user memory, the context budget, structured trajectories with redaction |
| `src/retrieval/` | multi-vector book index, in-house BM25 over jieba tokens, reciprocal-rank hybrid, and the retrieval benchmark |
| `scripts/` | 01–03 corpus, profiles, index; 10–12 annotation and splits; 15–16 rule blind-spot table and term-density table; 30–31 build book indexes and run the retrieval benchmark |
| `eval/` | the frozen v1 queries, human annotations, and under `eval/results/` the judged rows of every v1 arm |
| `docs/agent-plan.md` | the agent design: goals and trade-offs, architecture, evaluation, phases |
| `docs/v1-reranker/` | the previous system's README, architecture and evaluation notes |

Retrieval and RAG are the technical core: the corpus is raw text with no tags or
metadata, so every structured signal has to be extracted at index time. The agent
consumes retrieval; the evaluation measures both. Passage-level RAG, the offline
per-book card, the evaluation task set and the Streamlit chat are not started
yet; they follow the phases in `docs/agent-plan.md` §7.

```bash
uv sync
uv run pytest      # CPU only, no model downloads
```
