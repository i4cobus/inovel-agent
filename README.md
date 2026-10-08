# inovel-agent：网文领域的对话 Agent 与 Agent 评测

> English summary: [`README.en.md`](README.en.md)

语料：7,653 本中文网文，约 36 GB 纯文本，私有，不进 git。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `src/agent/` | agent loop、六个工具、单用户记忆、上下文预算、带脱敏的结构化轨迹 |
| `src/retrieval/` | 多向量书籍索引、自实现 BM25、RRF 混合、检索基准 |
| `src/` | 保留下来的检索原语（ingest、profile、embed、vector_index、search）、约束规则（`preferences.py`）、证据采样（`evidence.py`）、judge 与校准（`judge.py`、`evaluation.py`）、OpenAI 兼容 transport（`chat_transport.py`） |
| `scripts/` | 01–02 建语料与 profile；10–12 标注与划分；15–16 规则盲区与词频表；30–31 建书籍级索引与跑检索基准 |
| `eval/` | 旧评测查询、人工标注、以及 `eval/results/` 下每个 arm 的判分行 |
| `docs/agent-plan.md` | 设计文档：目标、取舍、架构、评测、分期 |
| `docs/retrieval-bench.md` | 书籍级检索基准：配置、数字、出处、决定 |
| `docs/v1-reranker/` | 重排器后训练（SFT + GRPO）的说明、方法与结果 |

技术核心是检索与 RAG：语料是没有任何标签的原始文本，结构化信号都要在建索引时提炼。
段落级 RAG、离线书卡、评测任务集、Streamlit chat 尚未开始，按 `docs/agent-plan.md`
第 7 节分期推进。

```bash
uv sync
uv run pytest      # CPU，无模型下载
```
