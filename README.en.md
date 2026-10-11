# inovel-agent: a conversational agent over a Chinese web-novel corpus, and how to evaluate it

> English summary. The Chinese [`README.md`](README.md) is the primary one, and all
> other documents are in Chinese.

Corpus: 7,655 Chinese web novels, about 36 GB of plain text with no tags or metadata,
private, not in git. Retrieval and RAG are the technical core: every structured signal
is extracted offline from the text, and the agent only consumes those artefacts.

## Data flow

```
raw text ──02──▶ digest (~40k chars per book: synopsis + 4 whole opening chapters + 100 chapter titles
                 + 6 whole middle chapters + 2 whole ending chapters)
                  ├──32──▶ book card (an LLM fills a closed three-layer vocabulary per book)
                  └──30──▶ multi-vector index (digest cut into ≤1,500-char paragraph chunks + the card, Qwen3-Embedding-4B)
                             └──33──▶ pooled single-vector index (the agent's default)
agent (a hosted API model; local Qwen3.5-9B as the comparison arm) ──▶ search_books / similar_books / get_profile / ask_book / check_term / check_trope / set_aside / memory_read / memory_write
```

**Digest** (`src/digest.py`): a deterministic sample that stands in for a book of several
hundred thousand to several million characters. Its version is part of the card cache key
and of the index metadata, so stale pairs cannot be mixed.

**Book cards** (`src/retrieval/cards.py`, `card_schema.py`): one structured record per
book, produced by an LLM from the whole digest. Three layers with a closed vocabulary:
genre (12 top-level, 41 sub-genres, each sub-genre in exactly one genre; the model fills
only the sub-genre), elements (40, in four groups; every element must carry a ≤20-char
verbatim quote that the parser checks against the digest, unverified ones are kept apart),
and five style scales (protagonist structure, 爽度, tone, romance line, starting point).
The vocabulary was audited line by line against the Qidian/Zongheng taxonomies rather than
copied; anything outside it is dropped and the reason recorded. Cards feed the index as one
section, `get_profile` (card first, then synopsis and opening) and `check_trope` (the card answers
what its genre, elements and style scales can; the text is sampled only when the card is silent).

**Index** (`src/retrieval/multivector.py`, `pooling.py`, `hybrid.py`, `bm25.py`): one
vector per chunk, a book scores as its best chunk; the single-vector index averages a
book's chunks per section kind and then weights the kinds, which takes seconds on CPU per
weighting. BM25 over jieba tokens and reciprocal-rank fusion are kept as reference
configurations. Embeddings: Qwen3-Embedding-4B in bf16 (batch 8, 1,536-token cap on a
16 GB card).

**Agent** (`src/agent/`): an OpenAI-compatible tool-calling loop with nine tools plus
`finish`, single-user long-term memory plus per-conversation session state, a context
budget with compaction of old results, and redacted structured trajectories. Users talk in
natural language; the agent writes the retrieval query itself and it reaches the embedder
verbatim (no parsing, no term stripping). Finding books: `search_books` with hard filters
from the book cards (`genre` / `elements` / `style`) when the user names them, and
`exclude_shown` for "show me others"; `similar_books` searches with a book's own vector.
Checking: negatives never reach the retriever; `check_term` (full-text term densities) and
`check_trope` (card first, sampled text when the card is silent) verify candidates, and
`get_profile` (card, synopsis, opening) answers positive preferences. In-book questions:
`ask_book` retrieves passages within one book over its digest chapters (opening 4, middle 6,
ending 2, chapter list), reusing the multi-vector index's chunk vectors through a memory map
rather than building a new index, and answers cite chapter headings; what the digest does not
cover is reported as such. Conversation: session state records what was recommended and set
aside so "the second one" and "something else" resolve (`set_aside`); a request too vague to
act on ends with a question (`asks_user`).

## Evaluation

- **Index stage**: measured as candidate supply for the agent. Queries come from the
  agent's task positives and the search strings it actually emits; metrics are top-10
  positive precision, feasibility (do at least three candidates survive the negative
  constraints), anchor hits, card consistency and paraphrase robustness; relevance comes
  from pooled LLM judging calibrated on 100 human-labelled pairs. In progress; the v1
  benchmark in `docs/retrieval-bench.md` is history.
- **Agent stage**: constrained-recommendation and cross-turn-memory tasks
  (`eval/agent/tasks/`); trajectory metrics, failure taxonomy and judge in
  `docs/agent-plan.md` §4.

## Repository layout

| path | contents |
|---|---|
| `src/digest.py` | digest rules and version |
| `src/retrieval/` | cards and vocabulary, multi-vector index and chunking, pooling, BM25, hybrid, benchmark |
| `src/agent/` | loop, tools, memory, context budget, trajectories |
| rest of `src/` | corpus cleaning and chapter splitting, embedding and FAISS, the constraint rule (`preferences.py`), evidence sampling (`evidence.py`), judge and calibration (`judge.py`, `evaluation.py`), transport |
| `scripts/` | 01–02 corpus and digest; 30 indexes, 31 benchmark, 32 cards, 33 pooling, 34–36 task synthesis / agent eval / metrics, 37 card review page; 10–16 are v1 annotation leftovers |
| `eval/` | task sets, the v1 queries and human labels, judged rows under `eval/results/` |
| `docs/agent-plan.md` | design document |
| `docs/card-schema-v3.md`, `docs/book-cards-pilot.md` | card vocabulary and prompt, pilot history |
| `docs/retrieval-bench.md` | v1 retrieval benchmark and section-kind attribution |
| `docs/first-trajectories.md` | first real trajectories |
| `docs/v1-reranker/` | the predecessor: reranker post-training (SFT + GRPO) |

```bash
uv sync
uv run pytest      # CPU only, no model downloads
```

Model calls read `INOVELREC_LLM_API_KEY` or `--api-key-file`; no credentials live in
the repository. Development happens on a Mac; card building and index building run as
scheduled tasks on a Windows machine with an RTX 4080.

## Status (2026-10-11)

Digest v2, book cards v3.2 (7,410 cards; 245 books rejected by the endpoint's content filter are
left without one), the 4B multi-vector index (294,485 vectors including the card sections) and
the pooled single-vector index are built; the agent defaults to `single_4b`. The agent layer has
been reworked for natural-language conversation (card filters, similar books, session state,
`ask_book`, question-style finish) and the main model moved to a hosted API, with the local 9B
kept as a comparison arm. Neither the dev evaluation nor the index-stage candidate-supply
evaluation has produced numbers yet. A build-on-demand full-book passage index, follow-up and
clarification task types, and a UI are next.
