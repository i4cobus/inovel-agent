# agent 层任务集（2026-10-27 扩容后）

五类任务，全部由程序合成、固定种子切分，test 307 / dev 85。2026-10-08 的原始 75 test / 20 dev（约束推荐 rec-000…127 中的 64 条、记忆 mem-000…030）逐字节保留、编号与切分不变，历史数字（dev 四轮、test-flash-01、test-9b-01）仍能对照；扩展批用 seed+1 生成，编号接在原批之后，并按整段对话去重。

| 类 | 文件 | test / dev | 会话数 | 测什么 | 期望（expect） |
|---|---|---|---|---|---|
| 约束推荐 constrained_rec | `constrained_rec.jsonl` | 140 / 36 | 1 | 题材 + 正向特征 + 1–2 个负向；变体 in_text / meta / mixed 各三分之一 | 推荐 ≥3 本、来自检索、每个负向逐本核查、规则能判的不违反 |
| 跨轮记忆 memory | `memory.jsonl` | 60 / 16 | 2–3 | persist / accumulate / oneoff | 记忆文件里该有的条目、后续推荐必须通过的词、不该持久化的词 |
| 会话状态 session_state | `session_state.jsonl` | 50 / 14 | 2 | 第一轮普通请求（无负向），第二轮：more「换几本」/ refer「第二本再讲讲」/ exclude「第一本看过了」 | more：≥3 本且与第一轮不重叠；refer：对第一轮编号 2 的书调过 get_profile / ask_book 等；exclude：第一本不再出现且至少 1 本新书 |
| 澄清 clarify | `clarify.jsonl` | 12 / 4 | 1 | 「推荐点好看的」这类无法开始找书的请求，16 种说法各一条 | finish 的 asks_user=true 且不推书 |
| 书内问答 book_qa | `book_qa.jsonl` | 45 / 15 | 1 | 见 `docs/agent-book-qa.md` | 找对书、引用章节真实、落在 gold 章节、答案与原文重合、不可答时如实说 |

判分在 `src/agent_eval/metrics.py`：`score_task` 按 kind 分派；新增标签 `repeat_recommended`、`reference_missed`、`excluded_recommended`（会话状态），`no_clarification`、`recommended_without_asking`（澄清）。pass 仍是「没有任何标签」。

## 为什么扩

75 条 test 的 95% 区间是 ±8 个百分点，单个变体 ±17–20，dev 20 条 ±17；0.75 到 0.80 这种差分不出来。扩到 300 左右整体区间收到 ±4，dev 85 条能看出一处修正 10 个点以上的效果。扩容只压采样噪声，不压模板偏差：任务仍是十几个模板乘词表，报告里要写明。

## 费用

Flash 跑 test-flash-01（75 条、107 会话）¥4.1，按会话线性估：test 307 条约 430 会话、约 ¥16；dev 85 条约 ¥4。9B 对照按时间估 75 条 54 分钟 → 307 条约 3.5 小时。

## 还没做的（按顺序）

1. 打分加供给条件化（按索引层可行性 / ask_book 是否取到 gold 分组报通过率）、Wilson 区间、未经请求核查拆「同义扩展 / 离题」。
2. 索引层：从 test 轨迹抽 agent 真实 query → 建池 → Qwen3.8-Max 判满足度（无人工校准，数字标注为模型判定）→ 同一批标签给 agent 层算「推荐满足率」。
3. 新任务类先在 dev 跑一轮验证提示，再用当前提示跑一次完整 test（含书内问答）。
