# inovel-agent: a conversational agent over a Chinese web-novel corpus, and how to evaluate it

> English summary. The Chinese [`README.md`](README.md) is the primary one, and all
> other documents are in Chinese.

Corpus: 7,653 Chinese web novels, about 36 GB of plain text, private, not in git.

## Repository layout

| path | contents |
|---|---|
| `src/` | the retrieval primitives that survived (ingest, profiles, embedding, FAISS index, single-query search), the constraint rule (`preferences.py`), evidence sampling (`evidence.py`), the judge and its calibration (`judge.py`, `evaluation.py`), an OpenAI-compatible transport (`chat_transport.py`) |
| `src/agent/` | the agent loop, six tools, single-user memory, the context budget, structured trajectories with redaction |
| `src/retrieval/` | multi-vector book index, in-house BM25 over jieba tokens, reciprocal-rank hybrid, and the retrieval benchmark |
| `scripts/` | 01–02 corpus and profiles; 10–12 annotation and splits; 15–16 rule blind-spot table and term-density table; 30–31 build book indexes and run the retrieval benchmark |
| `eval/` | the frozen v1 queries, human annotations, and under `eval/results/` the judged rows of every v1 arm |
| `docs/agent-plan.md` | the design document: goals and trade-offs, architecture, evaluation, phases |
| `docs/retrieval-bench.md` | the book-level retrieval benchmark: configurations, numbers, provenance, decision |
| `docs/v1-reranker/` | the reranker post-training work (SFT + GRPO): method, notes and results |

Retrieval and RAG are the technical core: the corpus is raw text with no tags or
metadata, so every structured signal has to be extracted at index time. The agent
consumes retrieval; the evaluation measures both. Passage-level RAG, the offline
per-book card, the evaluation task set and the Streamlit chat are not started
yet; they follow the phases in `docs/agent-plan.md` §7.

```bash
uv sync
uv run pytest      # CPU only, no model downloads
```
