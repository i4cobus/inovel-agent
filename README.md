# inovel-agent：网络小说对话 Agent

> English summary: [`README.en.md`](README.en.md)

一个面向私有网文库的对话 agent：用户用自然语言说想看什么，agent 自己检索、核查、读档案，给出带依据的推荐，也能回答某本书的情节问题。
语料 7,655 本中文网文、约 36 GB 纯文本，没有任何标签或元数据，私有，不进 git。
技术核心是离线提炼与检索：所有结构化信号（摘要、书卡、向量）都在离线阶段从原文里提炼，agent 只消费这些产物。

## 演示

演示视频（2 分 48 秒，无声）：[`demo/demo.mp4`](demo/demo.mp4)

https://github.com/i4cobus/inovel-agent/raw/master/demo/demo.mp4

![界面](demo/ui.jpg)


## 数据流

```
原始文本 ──02──▶ digest（每本约 4 万字：简介 + 开头 4 整章 + 100 个章节名 + 中段 6 整章 + 结尾 2 整章）
                  │
                  ├──32──▶ 书卡（LLM 按三层封闭词表为每本书出一张结构化卡）
                  │
                  └──30──▶ 多向量索引（digest 按段落切成 ≤1,500 字块 + 书卡段，Qwen3-Embedding-4B）
                             └──33──▶ 池化单向量索引（agent 默认用这个）

agent（Qwen3.8-Flash；本地 Qwen3.5-9B 作对照）
  ├─ 工具：search_books / similar_books / get_profile / ask_book / check_term / check_trope / set_aside / memory_read / memory_write / finish
  └─ 会话：每个会话自己的记忆、已推荐编号、对话记录 ──▶ 网页界面（scripts/serve.py）
```

### digest（`src/digest.py`）

一本网文几十万到几百万字，不可能整本进模型。按确定性规则抽一份代表全书的档案：
作者简介、开头四整章、一百个等距采样的章节名、六个等距位置的整章、结尾两整章，单章上限 8,000 字。
digest 是书卡和索引共同的输入，版本号写进卡的缓存键和索引的 metadata，新旧不会混用。

### 书卡（`src/retrieval/cards.py`、`card_schema.py`）

每本书一张卡，由 LLM 读整份 digest 生成，字段分三层，词表封闭：

- **题材**：12 个一级、41 个二级，每个二级只属于一个一级，模型只填二级，一级由程序推出；
- **元素**：40 个，分主角来路 / 外挂与体系 / 流派 / 世界设定四组，每个元素必须附一条 ≤20 字的原文摘录，程序在 digest 里核对，核不上或抄了词表定义的降为「待核」；
- **风格**：五个标尺，主角结构、爽度、基调、感情线（无 / 单女主辅线 / 单女主主线 / 多女主）、主角起点。

词表参考起点、纵横的分类体系逐条审定，不照搬、不自造；模型给词表外的词一律丢弃并记录原因。
设计经过 v1 → v3.2 多轮，见 `docs/card-schema-v3.md`、`docs/book-cards-pilot.md`。
全量 7,655 本出了 7,410 张卡，245 本被接口的内容审核拒收，不补。

卡用在四处：作为索引里的一个段参与检索；`search_books` 的硬过滤；`get_profile` 先给卡再给简介和开头；`check_trope` 先查卡（题材、元素、风格五维能直接回答的标签），卡没说的再采样原文让模型判。

### 索引（`src/retrieval/multivector.py`、`pooling.py`、`hybrid.py`、`catalog.py`）

- **多向量**：digest 的每个段按段落边界切成尽量等长、≤1,500 字的块，章节标题行在每块重复，书卡单独一段；每块一个向量，一本书按最佳块得分。全量 294,485 个向量。书卡段可以在主索引建完后单独嵌入追加，平铺内积索引与一起建结果相同。
- **单向量**：一本书的块向量先按段类求均值、再按段类加权合成一个向量，CPU 上几秒就能换一组权重。agent 的 `search_books` 用它。
- **过滤**：书卡过滤走 FAISS 的 id 选择器做精确过滤，不是检索后再筛，所以「仙侠 + 凡人流」命中 40 本时返回的就是这 40 本里最相关的。
- **对照**：jieba 分词的 BM25 和 RRF 融合保留为对照配置，agent 不用。
- 嵌入模型 Qwen3-Embedding-4B bf16，16 GB 显卡上 batch 8、序列上限 1,536，全量建索引约 8 小时。

### agent（`src/agent/`）

OpenAI 兼容的 tool-calling loop，九个工具加 `finish`。用户用自然语言对话；检索措辞由 agent 自己写，原样进向量检索，不经过任何解析或去词。

- **找书**：`search_books` 按自然语言描述检索，用户点名题材、元素、风格时用书卡做硬过滤，「换几本」用 `exclude_shown`；`similar_books` 用一本书自己的向量找相近的书。
- **核查**：负向条件不进检索，而是在候选里核：`check_term` 查全文词频表（系统、异能、僵尸这类文中词），`check_trope` 先查书卡、卡上没有再读原文（后宫、圣母、压抑这类标签，27 个可判）；只核查用户说了的或记忆里记着的条件。正向偏好读 `get_profile`（书卡、简介、开头）判断。
- **进书问答**：`ask_book` 在一本书的 digest 章节里做段落检索，向量直接从多向量索引里按书取（内存映射，不建新索引），回答带章节引用；覆盖之外的要如实说未收录。
- **对话**：会话状态记录已推荐（带编号）、已排除的书，让「第二本」「换几本」「这本看过了」成立；需求太模糊时以提问结束（`asks_user`）；每一轮都以 `finish` 交付结构化结果（回答、推荐列表、引用）。
- **记忆**：`memory_write` 分长期（「以后都不看后宫」）和仅本次；长期条目持久化并每轮写进系统提示。
- **上下文**：工具结果有字符预算，更早的检索结果压缩成「id|书名」；跨轮只重放最近 8 轮的用户话和回答。
- **兜底**：一轮最多 10 步，连续两步不调工具即终止，与上一步完全相同的调用即终止，模型接口错误重试 3 次后终止。

### 会话与界面（`src/agent/sessions.py`、`web.py`、`static/index.html`、`scripts/serve.py`）

一个会话就是一套独立上下文：换一类书，另开一个；另一个人用，也只需新开一个。
实现上每个会话是共享后端（编码器、索引、档案、词频表、judge）之上的一个 agent 分叉，自己持有记忆文件、会话状态和对话记录，存在 `data/sessions/<id>/`，服务重启后恢复。
Starlette 提供接口，一页前端：左栏会话，中间对话与逐步展示的过程（每次工具调用的参数与一行结果），右栏记忆、已推荐、已排除；回答里点书名滑出书卡面板。消息走 server-sent events，每一步完成即推到页面。不对话时没有任何模型调用。

## 评测

两层分开评，因为失败要能归因：是索引没把候选给够，还是 agent 没按流程用工具。

- **agent 阶段**（`src/agent_eval/`、`eval/agent/tasks/`、`docs/agent-dev-rounds.md`）：两类程序可判的任务。约束推荐（题材 + 正向特征 + 1–2 个负向条件，查推荐数、推荐是否来自检索结果、每个负向是否逐本核查、词频规则能判的是否违反）和跨轮记忆（长期 / 累积 / 仅本次三种变体，查记忆文件与后续推荐）。任务由模板和词表合成，固定种子切分 test 75 / dev 20。两轮 dev：Qwen3.8-Flash 0.55 → 0.75，Qwen3.7-Plus 0.30 → 0.70，中间修了五处（finish 截断、仅陈述偏好的轮次不结束、标签核查覆盖、元素误填题材、记忆任务的评分规则）；Flash 定为主模型。test 尚未跑。
- **索引阶段**（`src/retrieval/supply.py`、`supply_judge.py`，`docs/retrieval-supply.md`）：按 agent 的实际用法评「候选供给」，查询取 agent 在轨迹里真实发出的检索词，指标是 top-k 正向精确率、可行性（扣掉违反负向的还剩不剩 3 本）、锚点命中、卡一致率、改写鲁棒性；相关性由池化判定加人工校准给出。代码在，4B 索引上尚未跑。
- 当前任务集只覆盖「流程对不对」，不覆盖「推荐得好不好」和真实用户请求；完整评估（真实感请求集、书内问答、追问与澄清、推荐质量 judge）是下一阶段。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `src/digest.py` | digest 规则与版本 |
| `src/retrieval/` | 书卡与词表、过滤目录、多向量索引与切块、池化、BM25、混合检索、候选供给评测 |
| `src/agent/` | loop、工具、记忆、会话状态、段落检索、上下文预算、轨迹；会话存储、Web 接口与页面 |
| `src/agent_eval/` | 任务合成、并发运行、硬指标 |
| `src/` 其余 | 语料清洗与分章、嵌入与 FAISS、约束规则（`preferences.py`）、证据采样、judge 与校准、transport |
| `scripts/` | 01–02 语料与 digest；30/30b 建索引与追加书卡段、31 检索基准、32 建书卡、33 池化、34–36 任务合成 / agent 评测 / 指标、37 书卡审阅页、40–43 候选供给评测；`chat.py` 终端对话、`serve.py` 网页服务；10–16 为 v1 遗留 |
| `eval/` | 评测任务集与运行记录（`eval/agent/runs/`，轨迹已脱敏）、v1 查询与人工标注 |
| `docs/agent-plan.md` | 设计文档：目标、取舍、架构、评测、分期 |
| `docs/agent-rework.md`、`docs/agent-dev-rounds.md` | agent 层改造说明；两轮 dev 的数字、修正与模型选择 |
| `docs/card-schema-v3.md`、`docs/book-cards-pilot.md` | 书卡词表与 prompt 的定稿、试点过程 |
| `docs/retrieval-supply.md`、`docs/retrieval-bench.md` | 候选供给评测设计；v1 检索基准（历史） |
| `docs/v1-reranker/` | 前身项目：重排器后训练（SFT + GRPO） |

## 运行

```bash
uv sync
uv run pytest                                            # 250 个测试，CPU，无模型下载
uv run python scripts/02_build_digests.py                # digest
uv run python scripts/32_build_book_cards.py --model qwen3.8-flash --base-url <OpenAI 兼容地址> --no-thinking
uv run python scripts/30_build_book_indexes.py --dense multi --cards data/processed/book_cards_flash.parquet --dtype bf16 --batch-size 8
uv run python scripts/33_pool_single_from_multi.py --kind-mean
uv run python scripts/serve.py --base-url <地址> --api-key-file <key> --no-thinking --port 8765   # 网页界面，默认只绑 127.0.0.1
uv run python scripts/chat.py --base-url <地址> --api-key-file <key> --no-thinking                # 终端对话
uv run python scripts/35_run_agent_eval.py --run-id dev-01 --tasks eval/agent/tasks/constrained_rec.jsonl --base-url <地址> --api-key-file <key> --no-thinking --workers 6
uv run python scripts/36_agent_metrics.py --run-id dev-01 --tasks eval/agent/tasks/constrained_rec.jsonl   # 硬指标，CPU
```

模型调用走 `INOVELREC_LLM_API_KEY` 或 `--api-key-file`，端点走 `INOVELREC_LLM_BASE_URL`，仓库里不含任何凭证。
开发在 Mac；建卡、建索引、跑 agent 的任务在一台 RTX 4080 的 Windows 机器上以计划任务运行，界面也在那台机器上起，通过 ssh 隧道访问。

## 数据与版权

语料是私有的网文全文，不进 git，也不出现在任何提交的产物里：评测运行记录里的轨迹经过脱敏（原文段落替换为指纹），书卡只含词表内的标签和 ≤20 字的核对摘录。
界面上展示的是书名、书卡字段、作者简介和 agent 自己的话；`get_profile` 的开头摘录和 `ask_book` 的段落只给模型看，页面的过程区不展示原文。

## 现状（2026-10-12）

已完成：digest v2、书卡 v3.2（7,410 张）、4B 多向量索引与池化单向量索引、按自然语言对话改造的 agent 层、两轮 dev 评测与模型选择、多会话网页界面并在真实索引上跑通。
进行中：agent 侧 test 运行与 9B 对照；索引阶段候选供给评测在 4B 索引上的首轮。
下一步：真实感请求集与新任务类型（书内问答、追问、澄清）、推荐质量 judge；按需建全书段落索引。
