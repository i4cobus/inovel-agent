# inovel-agent：网络小说对话 Agent

> English summary: [`README.en.md`](README.en.md)

语料：7,655 本中文网文，约 36 GB 纯文本，没有任何标签或元数据，私有，不进 git。
技术核心是检索与 RAG：所有结构化信号都要在离线阶段从原文里提炼出来，agent 只消费这些产物。

## 数据流

```
原始文本 ──02──▶ digest（每本约 4 万字：简介 + 开头 4 整章 + 100 个章节名 + 中段 6 整章 + 结尾 2 整章）
                  │
                  ├──32──▶ 书卡（LLM 按三层词表为每本书出一张结构化卡）
                  │
                  └──30──▶ 多向量索引（digest 按段落切成 ≤1,500 字块 + 卡段，Qwen3-Embedding-4B）
                             └──33──▶ 池化单向量索引（agent 默认用这个）
agent（API 模型为主，本地 Qwen3.5-9B 作对照）──▶ search_books / similar_books / get_profile / ask_book / check_term / check_trope / set_aside / memory_read / memory_write
```

### digest

一本网文几十万到几百万字，不可能整本进模型。`src/digest.py` 按确定性规则抽一份代表全书的档案：
作者简介、开头四整章、一百个等距采样的章节名、六个等距位置的整章、结尾两整章，单章上限 8,000 字。
digest 是书卡和索引共同的输入，版本号写进卡的缓存键和索引的 metadata，新旧不会混用。

### 书卡（`src/retrieval/cards.py`、`card_schema.py`）

每本书一张卡，由 LLM 读整份 digest 生成，字段分三层，词表封闭：

- **题材**：12 个一级、41 个二级，每个二级只属于一个一级，模型只填二级，一级由程序推出；
- **元素**：40 个，分主角来路 / 外挂与体系 / 流派 / 世界设定四组，每个元素必须附一条 ≤20 字的原文摘录，程序在 digest 里核对，核不上或抄了词表定义的降为「待核」；
- **风格**：五个标尺，主角结构、爽度、基调、感情线（无 / 单女主辅线 / 单女主主线 / 多女主）、主角起点，模型直接判。

词表参考起点、纵横的分类体系逐条审定，不照搬、不自造；二级边界有说明行，旧名有别名表。
模型给词表外的词一律丢弃并记录原因。设计经过 v1 → v3.2 多轮，过程见 `docs/card-schema-v3.md`、`docs/book-cards-pilot.md`。
卡用在三处：作为索引里的一个段参与检索；`get_profile` 先给卡再给简介和开头；`check_trope` 先查卡（题材、元素、风格五维能直接回答的标签），卡没说的再采样原文让模型判。

### 索引（`src/retrieval/multivector.py`、`pooling.py`、`hybrid.py`、`bm25.py`）

多向量：digest 的每个段按段落边界切成尽量等长、≤1,500 字的块，章节标题行在每块重复，卡单独一段；每块一个向量，一本书按最佳块得分。
单向量：把一本书的块向量先按段类求均值、再按段类加权合成一个向量，CPU 上几秒就能换一组权重。
另有 jieba 分词的 BM25 和 RRF 融合，作对照配置。嵌入模型 Qwen3-Embedding-4B bf16，在 16 GB 显卡上 batch 8、序列上限 1,536。

### agent（`src/agent/`）

OpenAI 兼容的 tool-calling loop，九个工具加 `finish`，单用户长期记忆加会话状态，上下文预算与旧结果压缩，结构化轨迹并对原文脱敏。用户用自然语言对话；查询措辞由 agent 自己写，原样进向量检索，不再经过任何解析或去词。

- 找书：`search_books` 按自然语言描述检索，用户点名题材、元素、风格时用书卡做硬过滤（`genre` / `elements` / `style`），「换几本」用 `exclude_shown`；`similar_books` 用一本书自己的向量找相近的书。
- 核查：负向约束不进检索，`check_term`（全文词频表）和 `check_trope`（先查书卡，卡上没有再读原文）在候选里核；正向偏好读 `get_profile`（书卡、简介、开头）。
- 进书问答：`ask_book` 在一本书的 digest 章节（开头 4、中段 6、结尾 2 和目录）里做段落检索，向量直接从多向量索引里按书取（内存映射，不建新索引），回答带章节引用；覆盖之外的要如实说未收录。
- 对话：会话状态记录已推荐、已排除的书，让「第二本」「换几本」「这本看过了」成立（`set_aside`）；需求太模糊时 `finish` 以提问结束（`asks_user`）。

## 评测

- **索引阶段**：按 agent 的实际用法评「候选供给」，查询集来自 agent 的任务正向词与真实发出的检索词，指标是 top-10 的正向精确率、可行性（扣掉违反负向的还剩不剩 3 本）、锚点命中、卡一致率、改写鲁棒性；相关性由池化判定加 100 对人工校准给出。正在实现，旧的 v1 基准（`docs/retrieval-bench.md`）只作历史。
- **agent 阶段**：约束推荐与跨轮记忆两类任务（`eval/agent/tasks/`），轨迹指标、失败分类与 judge 见 `docs/agent-plan.md` 第 4 节。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `src/digest.py` | digest 规则与版本 |
| `src/retrieval/` | 书卡与词表、多向量索引与切块、池化、BM25、混合检索、检索基准 |
| `src/agent/` | agent loop、工具、记忆、上下文预算、轨迹 |
| `src/` 其余 | 语料清洗与分章、嵌入与 FAISS、约束规则（`preferences.py`）、证据采样（`evidence.py`）、judge 与校准（`judge.py`、`evaluation.py`）、transport |
| `scripts/` | 01–02 语料与 digest；30 建索引、31 检索基准、32 建书卡、33 池化、34–36 任务合成 / agent 评测 / 指标、37 书卡审阅页；10–16 为 v1 遗留的标注与词频表 |
| `eval/` | 评测任务集、v1 查询与人工标注、`eval/results/` 下的判分行 |
| `docs/agent-plan.md` | 设计文档：目标、取舍、架构、评测、分期 |
| `docs/card-schema-v3.md`、`docs/book-cards-pilot.md` | 书卡词表与 prompt 的定稿、试点过程 |
| `docs/retrieval-bench.md` | v1 检索基准的配置、数字与段类归因 |
| `docs/first-trajectories.md` | 首批真实轨迹 |
| `docs/v1-reranker/` | 前身项目：重排器后训练（SFT + GRPO） |

## 运行

```bash
uv sync
uv run pytest                                            # CPU，无模型下载
uv run python scripts/02_build_digests.py                # digest
uv run python scripts/32_build_book_cards.py --model qwen3.8-flash --base-url <OpenAI 兼容地址> --no-thinking
uv run python scripts/30_build_book_indexes.py --dense multi --cards data/processed/book_cards_flash.parquet --dtype bf16 --batch-size 8
#   书卡还没建好时：先不带 --cards 建，之后 scripts/30b_append_card_sections.py --index-dir <目录> 只嵌入书卡段并追加（平铺内积索引，结果与一起建相同）
uv run python scripts/33_pool_single_from_multi.py --kind-mean
uv run python scripts/chat.py --model qwen3.7-plus --base-url <OpenAI 兼容地址> --api-key-file ~/.config/aliyun.key --no-thinking
uv run python scripts/chat.py                            # 默认本地 Ollama qwen3.5:9b
uv run python scripts/35_run_agent_eval.py --run-id dev-01 --tasks eval/agent/tasks/constrained_rec.jsonl --model qwen3.7-plus --base-url <地址> --api-key-file <key> --no-thinking --workers 8
```

模型调用走 `INOVELREC_LLM_API_KEY` 或 `--api-key-file`，仓库里不含任何凭证。
开发在 Mac，建卡与建索引等长任务在一台 RTX 4080 的 Windows 机器上以计划任务运行。

## 现状（2026-10-11）

digest v2、书卡 v3.2（7,410 张，245 本被接口内容审核拒收、不补）、4B 多向量索引（294,485 向量，含书卡段）与池化单向量索引均已建成，agent 默认用 `single_4b`。
agent 层按自然语言对话改造完毕（书卡过滤、相似书、会话状态、`ask_book`、提问式结束），主模型改为 API 模型，本地 9B 留作对照；dev 评测与索引阶段的候选供给评测尚未跑出数字。
按需建全书段落索引、追问与澄清类评测任务、界面属于下一步。
