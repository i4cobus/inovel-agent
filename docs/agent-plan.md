# 设计文档：从网文重排器到网文对话 Agent

> 状态：草稿，边写边讨论。2026-10-07 起草。
> 标注为「估算」的数字都没有实测，第一次跑完就换成实数。

## 1. 目标与边界

把本仓从「一个固定管线的推荐系统 + 重排器后训练」演进为「网文领域的对话 agent +
agent 评测体系」，用于 2026 年秋招 agent 算法岗。对照目标岗位 JD 的覆盖面：

| JD 要点 | 本项目的对应 | 状态 |
|---|---|---|
| Context | 工具结果压缩、记忆按需注入、会话压缩 | 新做 |
| Planning | 自写 agent loop，多步任务（推荐后追问、跨会话） | 新做 |
| Tool Use | 七个工具（含记忆读写），四个由现有 Stage 1–4 原语改造 | 新做 |
| RAG | 两级：书籍级 profile 检索（已有）+ 书内段落级检索（新） | 扩展 |
| Memory | 用户长期记忆（偏好、禁忌、已读）+ 会话短期摘要 | 新做 |
| 自动化 Eval | 四类任务、程序校验的硬判据 | 新做 |
| Trajectory Analysis | 结构化轨迹日志 + 固定失败标签集 | 新做 |
| LLM-as-Judge | 四个软指标，沿用人工标注校准流程 | 沿用方法 |
| Synthetic Data | 任务合成（模板 + 本地模型 + 可验证过滤） | 沿用方法 |
| SFT / RL | 已完成的 SFT + 四轮 GRPO（见 `docs/post-training.md`） | 已有，不新跑 |
| Trajectory Learning | **不做** | 砍 |
| 论文复现 | **不做** | 砍 |

**不做的理由**：轨迹学习是所有项中 API 花费最高（teacher 轨迹收集）且在本项目规模下最
难出统计结论的一项；SFT / RL 这条 JD 要求由已完成的后训练工作覆盖。论文复现通常要重复
跑实验，算力不允许。两项都在面试里作为「数据收齐后的下一步」讲，附显存估算。

### 算力边界

| 资源 | 规格 | 用途 |
|---|---|---|
| Mac M4 Pro | 24 GB 统一内存 | 开发；候选运行 Qwen3-30B-A3B 4-bit |
| jacob-desktop RTX 4080 | 实际可用约 14.4 GiB | 嵌入、建索引、评测；运行 8B / 14B |
| API | 按次报预算 | 只用于 judge 软指标子集 |

PC 显卡目前有其他任务在用，**启用前先问**。

## 2. 决策记录

每条都列被放弃的选项和代价，便于回头翻。

| # | 决策 | 放弃了什么 | 为什么 |
|---|---|---|---|
| D1 | 原地演进本仓，不新开 | 一个干净的新仓 | 后训练工作正好覆盖 JD 的 SFT/RL 条，一个仓讲完整故事。旧系统打 tag `v1-reranker` 保持可寻址 |
| D2 | 重排层（`rank.py` 等）从运行时删除，检索原语保留为工具 | 现成的推荐质量（teacher 32B 水平的重排） | 逐候选 LLM 打分就是 agent 自己该做的判断；4B 重排器 8 GB 也和 agent 模型不能共存 |
| D3 | 嵌入模型由锚点检索测试选定（0.6B / 4B / 8B 同台比），不预设 0.6B | 与旧评测各 arm 的可比性 | 8B 嵌入与 agent 模型不能同驻 4080，但建索引是离线批处理，运行时只编码查询，可以放 CPU 或 Mac。所以显存不是选型依据，检索质量才是，见 3.7 |
| D4 | 自写最小 agent loop，不用框架 | 框架的流式、并行工具、持久化 | 算法岗面试问的是 loop 内部怎么设计 |
| D5 | agent 模型全本地，API 只做 judge | 强模型的 demo 效果 | 开发和评测循环零花费；弱模型让评测体系有失败可分析 |
| D6 | 砍轨迹 SFT 和论文复现 | JD 两个要点的直接证据 | 见第 1 节 |
| D7 | 任务集先 150 test + 40 dev，后扩 | 第一版的统计功效 | 合成脚本和硬判据写好后扩量是机械的 |
| D8 | 段落级索引只建子集（先 200 本） | 全语料问答 | 全量约 3,600 万块，不可行；子集外的书成为「不可答」评测用例 |

## 3. 架构

```text
用户（Streamlit chat）
  → agent loop（src/agent/loop.py）
      ├─ 上下文管理（预算、压缩、记忆注入）
      ├─ 工具调度（OpenAI 兼容 tool calling，经 http_matcher 的 ChatTransport）
      └─ 轨迹记录（JSONL，每步一行）
工具
  search_books      书籍级：query → 嵌入 → FAISS（profiles）→ 候选 + 300 字预览
  get_profile       novel_id → 完整 profile（截断到预算）
  check_term        novel_id + 文中词 → 规则判定（preferences.py 的规则 + 词频表）
  check_trope       novel_id + 元标签 → 采样证据 + 模型判定（带缓存）
  ask_book          novel_id + 问题 → 该书段落级索引检索 → 带章节出处的段落
  memory_read / memory_write   用户长期记忆
```

### 3.1 Agent loop

- 消息列表 = system（角色、工具使用规范、记忆摘要）+ 对话历史 + 当前轮的工具往返。
- 每步：调模型 → 若返回 tool calls 则逐个执行并追加 observation → 否则视为最终回答。
- 终止条件：模型给出无工具调用的回答；或达到步数上限（初定 10）；或连续两次同参数调用
  同一工具（视为循环，强制终止并记录）。
- 温度 0，评测时可复现。
- `enable_thinking` 作为配置项，默认关（沿用现有 JSON 解析的教训），评测时作为对照变量。
- 最终回答要求自然语言 + 结构化附件（推荐列表或引用列表的 JSON），后者供硬判据解析。
  解析失败记为「格式失败」，不猜。

### 3.2 工具定义

| 工具 | 参数 | 返回 | 实现来源 |
|---|---|---|---|
| `search_books` | `query: str`, `k: int ≤ 20` | `[{novel_id, title, preview, score}]` | `search.semantic_search` + `vector_index.load_id_map` |
| `get_profile` | `novel_id` | `{title, profile}`，profile 截到 1,200 字 | `novel_profiles.parquet` |
| `check_term` | `novel_id`, `terms: list[str]` | `{violates: bool \| null, evidence: {term: density}}` | `preferences.constraint_violation_from_densities` + 词频表 |
| `check_trope` | `novel_id`, `trope: str` | `{verdict: yes\|no\|unclear, quotes: [str], confidence}` 或 `unknown_trope` | 新：全书采样约 5,000 字（`evidence.py` 窗口采样 + 简介）+ 元标签定义表 + 模型判断，按 `(novel_id, trope, prompt_version)` 缓存 |
| `ask_book` | `novel_id`, `question` | `[{chapter_idx, chapter_title, passage, score}]` 或 `not_indexed` | 新：`src/rag/` |
| `memory_read` | 无 | 用户记忆 JSON | 新：`src/agent/memory.py` |
| `memory_write` | `kind: positive\|negative\|read\|note`, `value`, `persistent: bool` | 确认 | 同上 |

**约束校验分两层，可靠度不同，agent 必须知道自己在哪一层。**

- `check_term` 是确定性规则，只对「系统、异能、僵尸」这类文中会出现的词有效（上一版
  测得召回 63%）。对「后宫、爽文、圣母」这类元标签返回 `null`：这些词在书里基本不出现，
  数词频没有意义。旧评测 59 条查询里 42 条的约束是元标签，这是主要情况而不是边角。
- `check_trope` 是概率性判断。读全书不可能，所以从全书按固定比例采样约 5,000 字（含
  简介），对照一份手写的元标签定义表（后宫 = 主角与多名异性并存感情线；种马、爽文、
  圣母、无脑各一条）让模型给结论和证据引文。会漏掉后期才展开的线，这是方法的固有
  上限。缓存让调过的书不再花费，用着用着就变成一张标签表；200 本子集可在空闲时预热。
  放弃的方案是离线给全部 7,653 本打固定词表的标签：代码相同但词表定死、一次性花一天
  量级显卡时间。
- 两个防线：`check_trope` 的采样窗口和 judge 的证据窗口用不同种子，否则工具和裁判在看
  同一批文字；`check_trope` 在 agent 写出来之前先对旧评测的元标签违规标签（judge 行 +
  人工标注）算精确率和召回率，按标签分开报，不达标就换模型或加采样量，不能先用了再说。
- 真实产品里这件事靠平台标签解决，这里不爬，用模型打标顶替。

`null` 和 `unclear` 都是故意暴露的：规则的盲区是上一版的核心发现，agent 必须知道工具
什么时候不可信，自己去读 profile 或调 `ask_book`。

查询扩展（`DOMAIN_HINTS`）不再是独立模块。agent 自己改写查询多调几次 `search_books`
就是查询扩展，这也是 Planning 的一部分。

### 3.3 段落级 RAG

- 子集：200 本，从语料里分层抽（按体裁），固定随机种子，列表进 git。
- 分块：先按 `split_chapters` 切章，章内按约 800 字、重叠 100 字切窗。每块带
  `novel_id, chapter_idx, chapter_title, char_offset`。
- 索引：**每本书一个 FAISS 文件**，而不是一个大索引加过滤。`ask_book` 总是带 `novel_id`，
  按书加载最简单；200 个小文件也便于只建一部分先跑通。
- 嵌入：与书籍级同一个模型（由 3.7 选定），走 `embed.encode_documents` 的角色化编码。
- 规模（估算）：200 本约 2 亿字 → 约 25 万块 → 0.6B 时索引约 1 GB，4B（2560 维）约 2.5 GB。

**不做 `chapter_range`。** 它服务两种需求，都不做：防剧透过滤（元数据筛选，便宜，但 agent
要从措辞推范围，多一个失败模式，收益是一类任务）；范围阅读/摘要（「前十章讲了什么」不是
检索问题，查询没有内容词，要按范围做 map-reduce，每问分钟级）。`ask_book` 只保留
`novel_id` 和 `question`，问答任务只有开放检索问答一种。

### 3.4 记忆

- 面向单用户，不做多用户隔离。长期记忆是一个文件 `data/memory/user.json`，字段
  `preferences.positive`、`preferences.negative`、`read_titles`、`notes`，每条带时间戳和
  来源轮次。新会话开始时注入摘要到 system。评测跑记忆类任务时每条任务前重置这个文件，
  这是测试 harness 的事，不是产品设计。
- 短期：会话内超过上下文预算时，把早期轮次压缩成摘要，工具 observation 只保留结论。
- 一次性约束 vs 长期偏好的区分由 `memory_write` 的 `persistent` 参数承担，agent 根据
  用户措辞判断（「这次」「以后都」）。评测里专门有「一次性约束不该被持久化」的用例。

agent 写记忆不需要用户确认，但回答里要显式说明「已记住……」，让用户能纠正。

### 3.5 上下文管理

| 项 | 预算（初值） |
|---|---|
| `search_books` 每条 preview | 300 字（沿用 `id_map` 的 preview） |
| `get_profile` | 1,200 字 |
| `ask_book` 每段 | 800 字，最多 5 段 |
| 记忆注入 | 500 字摘要 |
| 会话压缩阈值 | 模型上下文窗口的 60% |

这些数字进配置，评测里不调。

### 3.6 界面

Streamlit chat。每轮展示最终回答，侧栏或折叠区展示轨迹（每步 thought、tool call、
observation、耗时）。不做流式。`src/streamlit_app.py` 重写，`annotate_app.py` 保留给
judge 校准标注。

**公开 demo 只放截图**，关键信息（书名、原文段落、作者）打码。语料私有，README 已声明
不进 git，这一点不变。

### 3.7 检索质量：用锚点测试选嵌入模型和索引方式

旧系统的书籍级检索本身就弱，重排层在扛。`eval/results/anchor_ranks_baseline.json`
是 8B 嵌入、单向量 profile、无重排时 55 个锚点书的检索排名：

| 指标 | 8B 嵌入、单向量 |
|---|---|
| 中位排名 | 184 |
| Hit@10 | 0.145 |
| Hit@50 | 0.273 |
| Hit@200 | 0.509 |

删掉重排层后，agent 面对的就是这个检索。锚点集只有 55 条且偏向名著，是 smoke test
不是 benchmark，但它免费、不用 judge、几分钟出结果，拿来**选型**够用。在 PC 空出来后
同台比较：

| 变量 | 候选 |
|---|---|
| 嵌入模型 | Qwen3-Embedding 0.6B / 4B / 8B |
| 每本书的向量数 | 单向量（现状，8,000 字拼一起）/ 多向量（简介 + 每个采样章节各一个，取最大分） |
| 稀疏通道 | 无 / BM25（jieba 分词，CPU，免费），与稠密分 RRF 融合 |

报 Hit@10 / Hit@50 和中位排名。预期多向量和 BM25 带来的提升大于嵌入模型尺寸带来的
（估计，待测）。选定后的配置同时用于书籍级和段落级。

运行时只需编码查询：0.6B 和 4B 可以放 PC 的 CPU 或和 agent 模型同驻显存（4B 要实测），
8B 放 Mac 的话 30B-A3B 就放不下了。嵌入模型选 8B 的代价就是这个部署约束。

## 4. 评测

三条原则沿用上一版：评测标签与系统的任何优化信号隔离；judge 对人工标注校准；统计功效
先算后做。

### 4.1 任务集

test 150 条冻结，dev 40 条用于改 prompt 和 loop。**改完 prompt 回头看 test 再改就是在
test 上调参**，不允许；test 每个模型配置只跑一次。

| 类型 | test / dev | 考什么 | 硬判据 | 软判据（judge） |
|---|---|---|---|---|
| 约束推荐 | 50 / 14 | Tool Use、约束遵守 | 文中词约束：每本推荐通过规则；调用过 `check_term`。元标签约束：调用过 `check_trope`，且未推荐其判定为 yes 的书 | 相关性 0/1/2；违规率（judge 看原文证据） |
| 书内问答 | 50 / 14 | RAG | 引用章节 = 金标章节；短答案字符串匹配；不可答用例要拒答 | 正确性（对照金标）；忠实性（对照检索段落） |
| 推荐后追问 | 25 / 6 | Planning | 工具顺序合法；追问的书来自推荐结果 | 同上 |
| 跨会话记忆 | 25 / 6 | Memory | 会话 1 写了记忆；会话 2 未重述仍遵守；更新后以最新为准；一次性约束未持久化 | 无 |

**对照组**：每个模型配置都另跑一个**无工具单次回答**的 arm，即同一模型拿到
`search_books` top-20 预览后一次性作答，不能调工具、不读 profile、不查约束。主表里
agent 的每个数字旁边都放这一列。这是本项目对「做成 agent 有没有用」的直接回答，比和
旧的 32B 重排系统比更诚实，因为检索、模型、任务集都相同，唯一变量是 agent loop。

**约束推荐的循环要说明**：`check_term` 和硬判据是同一条规则，agent 调工具过滤就能
满分。硬指标衡量「会不会用工具」，质量看 judge。上一版测得规则只能看见 63% 的真实违规，
盲区表照搬，两列之差就是「工具没覆盖的部分 agent 自己补了多少」。

### 4.2 任务合成

- 约束推荐：约束词从词频表取，先做候选池预检（沿用 `arm_precheck` 思路），保证检索池里
  确实有违规书，否则任务无区分度。
- 书内问答：本地模型读子集中某一章出题，产出 `{question, answer, chapter_idx}`。
  可验证过滤：答案字符串必须在该章出现。人工抽 50 条看质量。不可答用例两种：书不在
  子集；事件是编的（由模型改写一条真问题的关键实体生成）。
- 推荐后追问、跨会话记忆：由前两类组合，追问的书限定在子集内。
- 合成全程本地模型 + 模板，不花钱。

### 4.3 轨迹指标与失败分类

每条轨迹 JSONL，每步记 `thought, tool_call, observation, tokens_in, tokens_out, latency`。

通用指标：步数、总 token、总耗时、工具调用合法率（schema）、冗余调用率（同参数重复）、
过早终止率、最终输出格式合法率。

失败标签集（固定）：错误工具、参数错误、循环、过早终止、引用编造、约束忽略、记忆丢失、
上下文溢出、格式失败。judge 打标，人工抽 50 条核对一致率。

**轨迹文件的提交规则**：`ask_book` 和 `get_profile` 的 observation 含原文，完整轨迹只留
本地。进 git 的是脱敏版：observation 替换为 `{sha256, length, novel_id, chapter_idx}`，
最终回答里的引用段落同样替换。指标全部从脱敏版可复算，这是脱敏字段设计的约束。

### 4.4 Judge

只负责四个软指标：相关性、违规、正确性、忠实性。后两个是对照金标打分。每个指标人工标
60 到 100 条，报 kappa 和一致率，不达标的指标不进主表。judge 用 API，只跑 test 集需要
软指标的子集，**跑前报模型、调用量、预估 token、prompt**。沿用 `src/judge.py` 的
`BudgetGuard` 和缓存。

### 4.5 统计与算力

- 主比较：模型配置之间逐任务配对（8B、14B、30B-A3B，thinking 开/关），报 bootstrap
  置信区间和配对检验。n=150 时能测出约 8 个百分点的差；扩到 300 后约 5 个百分点。
- 本地模型温度 0 跑一遍；方差用 dev 集跑 3 次报。
- 算力（估算）：150 任务 × 约 6 步，8B 在 4080 上一步 10 到 20 秒，一个配置约 3 到 5
  小时。dev 集一轮约 1 小时。

### 4.6 报告里的表

1. 四类任务 × 模型配置主表（硬指标全量，软指标子集）
2. 失败标签分布 × 模型配置
3. Judge 校准表
4. 规则盲区表（规则 vs judge 的违规判定分歧）
5. `check_trope` 校验表（对旧评测元标签标签的精确率 / 召回率，按标签分）
6. 工具调用效率表（步数、冗余率、token）

## 5. 代码迁移

先打 tag `v1-reranker`，再动代码。

| 模块 | 处置 |
|---|---|
| `ingest clean profile embed vector_index search split_chapters evidence preferences backends http_matcher judge evaluation annotate_app config schema text_utils splits` | 保留 |
| `rank llm_matcher explain llm_explain report app_pipeline query_expansion` | 删，运行时不再需要 |
| `grpo_reward verl_reward sft_data query_synthesis` | 删出运行时；方法与结果写入 `docs/post-training.md` |
| 脚本 04–09、13–23 | 删；01、02、03、10、11、12 保留（03 改参数重建） |
| 对应测试 | 随模块删 |
| `streamlit_app` | 重写 |

新增：

```text
src/agent/     loop.py  tools.py  memory.py  context.py  trajectory.py
src/rag/       chunk.py  index.py  retrieve.py
eval/agent/    tasks/（合成任务 JSONL）  metrics.py  failure_labels.py
scripts/       30_rebuild_book_index.py  31_build_chunk_index.py  32_synthesize_tasks.py
               33_run_agent_eval.py  34_agent_metrics.py  35_judge_agent.py
docs/          post-training.md（旧 README 的主体）  agent-plan.md（本文）
```

脚本编号从 30 起，延续「按执行顺序编号」的约定，和删掉的号段不混。

README 重写为 agent 项目；后训练作为一章链接到 `docs/post-training.md`，结果表和
`eval/results/` 原样保留。

## 6. 模型与算力

| 候选 | 放哪 | 显存 / 内存（估算） | 把握 |
|---|---|---|---|
| Qwen3-8B 4-bit | 4080 | 约 6 GB + 0.6B 嵌入 1 GB 多 | 装得下 |
| Qwen3-14B 4-bit | 4080 | 权重约 9 GB + 16k 上下文 KV 约 2 到 3 GB + 嵌入 | **不确定**，要实测 |
| Qwen3-30B-A3B 4-bit | Mac | 权重约 17 GB，激活 3B 速度快 | **不确定** 24 GB 够不够留上下文 |

- 同尺寸优先 2507 更新版（Instruct 版改善了工具调用）。开工时查最新版本。
- 服务方式：Windows 侧 Ollama 的 OpenAI 兼容端点（vLLM 不可用）。要确认 Ollama 对
  Qwen3 的 tool calling 和 thinking 开关支持。Mac 侧 Ollama 或 MLX。
- 8B 是固定 baseline；14B 和 30B-A3B 在同一任务集上实测后选主模型。

## 7. 分期

| 期 | 周 | 交付 | 可讲的点 |
|---|---|---|---|
| 1 | 1–2 | tag、删代码、锚点测试选定检索配置并重建书籍级索引、agent loop + 六个工具（无 `ask_book`）、轨迹日志、约束推荐 + 记忆两类任务、硬指标、Streamlit chat | Tool Use、Planning、Memory、自动化 Eval |
| 2 | 3–4 | 段落级索引、`ask_book`、问答 + 追问两类任务、失败分类、judge 校准、模型对比 | RAG、Trajectory Analysis、LLM-as-Judge、Context |
| 3 | 可选 | 扩到 300 条、thinking 对照、README 定稿 | 统计功效 |

每期结束 README 处于可读状态。

## 8. 待定与待实测

已定：

- 面向单用户，记忆不做隔离（3.4）。
- 公开 demo 只放打码截图；轨迹脱敏后才进 git（3.6、4.3）。

- agent 写记忆不需要用户确认，但回答里显式说明「已记住……」（3.4）。
- 文档和 README 统一用中文；现有英文文档在重写时转中文。
- `.gitignore` 不再整体忽略 `docs/`，只忽略含语料的 `docs/prompt*.md`、`docs/pre_interview.md`。
- `ask_book` 不做 `chapter_range`，防剧透和范围摘要都不做（3.3）。

待定（需要讨论）：

1. `check_trope` 的元标签定义表：先按旧评测出现过的 27 个元标签起草，再改。

待实测（PC 空出来后，按顺序）：

1. 3.7 的锚点测试：嵌入模型 × 向量数 × BM25，选定检索配置。这一步决定后面所有索引。
2. 选定嵌入模型在 4080 上的吞吐；重建书籍级索引和 200 本段落级索引的实际耗时。
3. 三个候选 agent 模型的显存占用和每步延迟。
4. Ollama 的 tool calling 与 Qwen3 配合是否稳定。
5. `check_trope` 对旧评测元标签标签的精确率 / 召回率；采样 5,000 字够不够。

## 附录 A：`check_trope` 元标签定义表（草稿）

每条是给模型的判定依据：采样文本里**看到什么就判 yes**，看不到判 no，只有间接迹象判
unclear。按三组分，第三组是读者口味标签，主观性最强，定义里给的是可观察的代理特征。

**题材类**（简介通常就能判）

| 标签 | 判 yes 的依据 |
|---|---|
| 玄幻 | 架空世界，有修炼体系或超自然力量体系，非仙侠语境（无修仙、飞升、道门词汇） |
| 灵异 | 鬼怪、灵体、诅咒、凶宅等超自然恐怖元素是主线 |
| 超能力 | 现代或近未来背景，人物拥有非修炼来源的特殊能力 |
| 克苏鲁 | 不可名状的旧日支配者、理智值、疯狂、邪神信仰等要素 |
| 言情 | 男女感情线是主线，情节围绕两人关系推进 |
| 争霸 | 主角以势力扩张、攻城略地、建国称帝为主线 |
| 宫斗 | 后宫或朝堂内部的权谋争斗是主线 |

**人物关系与设定类**

| 标签 | 判 yes 的依据 |
|---|---|
| 后宫 | 主角与两名以上异性存在并存的、被叙事认可的感情或伴侣关系 |
| 种马 | 主角与多名异性发生关系且叙事不作感情铺垫，关系对象持续增加 |
| 独狼 | 主角长期无固定伙伴，拒绝组织与同伴，独自行动是性格设定 |
| 金手指 | 主角拥有他人没有的外挂式优势（系统、随身空间、重生先知、特殊血脉）且情节依赖它 |
| 开挂 | 同金手指，优势远超同阶且几乎没有代价 |
| 开局无敌 | 故事开始时主角已处于实力顶点或立即获得顶级实力 |
| 速通 | 升级或目标达成极快，几章内跨越常规需要大量篇幅的阶段 |
| 魔改 | 以已有作品或历史为底本，改动人物命运或核心设定 |

**叙事风格与读者口味类**（主观，按代理特征判）

| 标签 | 判 yes 的依据 |
|---|---|
| 爽文 | 冲突被迅速解决，主角连续占上风，受挫后快速反转，几乎没有长期失败 |
| 打脸 | 反复出现「被轻视后当众证明实力、对方难堪」的情节模式 |
| 宠文 | 男女主之间一方对另一方无条件偏爱、纵容，几乎没有感情危机 |
| 圣母 | 主角反复为陌生人或敌人牺牲自身利益，不计后果地原谅或救助对手 |
| 玛丽苏 | 主角被几乎所有人无理由喜爱或倾慕，缺点不被叙事承认 |
| 恋爱脑 | 人物的重大决定主要由感情驱动，为感情放弃明显更重要的目标 |
| 虐主 | 主角持续遭受羞辱、重大损失或肉体精神折磨，且篇幅占比高 |
| 压抑 | 整体基调阴郁，反复出现绝望、无力、牺牲无回报的情节，少有轻松段落 |
| 狗血 | 密集的巧合、身世反转、误会、三角关系等戏剧化桥段 |
| 搞笑 | 大量段子、吐槽、滑稽情节，基调轻松 |
| 无脑 | 冲突解决不依赖策略或信息，靠实力碾压或对手犯低级错误 |
| 小白 | 文字直白、人物扁平、情节套路化，解释性叙述多于描写 |

备注：「无脑」「小白」「狗血」三条即便人工也难一致，校验表里如果这几条 kappa 过低，
就从 `check_trope` 的词表里去掉，返回 `unknown_trope` 让 agent 自己读 profile 判断。
