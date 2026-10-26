# 书内问答任务集（book_qa）：引用是否有原文支持（2026-10-26）

agent 的 `ask_book` 回答带章节引用，之前没有任何评测核过「引用的章节是不是真的、答案是不是从那段话来的」。这一类任务补上这个空：全程序生成、全程序判分，不用 judge，不花钱。

## 问题怎么出

gold 来自 digest 本身：digest 收录了每本书的 12 个整章（开头 4、中段 6、结尾 2）和 100 个等距抽样的章节名。从 7,410 张有卡的书里选符合条件的（12 章齐全、章节标题都带「第N章」编号、书名在库内唯一、书名 ≥ 3 字；6,538 本合格），固定种子抽 60 本，五种变体轮流：

| 变体 | 用户怎么问 | gold | 测什么 |
|---|---|---|---|
| chapter_event | 《X》里「第184章 龙济光的计划」那一章讲了什么？ | 那一章 | 能不能按章节定位、引用对不对 |
| quote_locate | 《X》里有一句「……」，是在哪一章、什么情况下出现的？ | 含这句话的唯一一章（16–48 字、只在一个 digest 章节出现的叙述句） | 段落检索命中 + 引用 |
| opening | 《X》开头讲了什么？主角是谁？ | 开头 4 章中任一 | 最自然的问法 |
| ending | 《X》的结局怎么样？ | 结尾 2 章中任一 | 同上 |
| out_of_coverage | 《X》里「第37章 追击行动」那一章讲了什么？（章节名来自目录抽样、不在 12 章里） | 无 | 边界：该说「摘要未收录」而不是编 |

生成：`uv run python scripts/34_synthesize_tasks.py`（`--qa-test 45 --qa-dev 15`，和另两类任务同一个种子；旧任务文件不变），输出 `eval/agent/tasks/book_qa.jsonl`。每条任务的 `expect` 里带 `novel_id`、`answerable`、`gold_chapters`。

## 判分（`src/agent_eval/metrics.py` 的 `score_qa_session`）

从脱敏轨迹加 digest 表判，不依赖段落原文是否留在轨迹里：

- `book_unresolved`：没有对 gold 的 novel_id 调过 `ask_book`。为此 `search_books` 加了书名精确匹配：query 是书名或含书名时，那本书排第一并标 `title_match=true`（向量检索按书名找书不可靠，不加这一步会把「找错书」和「引用错」混在一起）。
- `ungrounded_citation`：finish 的 citations 里有一条对不上 digest 的任何章节（按章节号比对，「第184章」≡「第184章 龙济光的计划」；无编号时按去空格后的文本）。这就是「引用是否有原文支持」里的「有没有这一章」。
- `no_citation` / `gold_not_cited`（可答题）：没给引用；给了但没有一条落在 gold 章节上。
- `answer_unsupported`（可答题）：答案里的中文二元组（去掉问题里已有的）在被引用章节原文里的占比 `answer_support` 低于阈值。阈值 0.25 是**暂定值**，等 dev 轨迹出来后按真实回答校准；指标本身是连续量，表里报均值。
- `fabricated_answer`（不可答题）：答案里没有「未收录 / 没有收录 / 无法确认」这类说法（正则 `ABSTAIN_RE`）。
- 通用：`format`（没以 finish 结束）、`budget_exhausted`、`loop`、`model_error`。

汇总表新增列：`book_resolved_rate`、`citation_grounded_rate`、`gold_cited_rate`、`answer_support_mean`、`abstain_rate`。打分：`uv run python scripts/36_agent_metrics.py --run-id <id> --tasks eval/agent/tasks/book_qa.jsonl`（digest 表默认路径，只读任务涉及的书）。

## 配套改动

- 提示里加了一段：点名某本书时先用书名搜到 novel_id、核对书名、再 ask_book；回答只依据返回段落；覆盖之外如实说未收录。
- `search_books` 书名匹配见上；重名的书名（36 组）不参与匹配。

## 首轮 dev（2026-10-27，15 条）

14/15 通过。找对书 15/15，gold 章节被取到 12/13，引用真实 13/13，取到即引对 12/12，两条未收录章节的题都如实拒答；答案支持度 0.28–0.54（均值 0.41）。唯一失败是原句定位的一条：三次 ask_book 都没取到含那句话的章节，记 `chapter_not_retrieved`，归工具侧。对照轨迹后判分改了四处：答案指标读完整轨迹（脱敏文件把 answer 指纹化了）；引用 ask_book 返回的「简介」「章节目录」段落算真实引用；开头类问题可只靠 get_profile（档案即开头摘录）；拒答措辞表加了真实出现的几种说法。细节见 `docs/agent-dev-rounds.md` 第五轮。

## 边界与已知不足

- 「答案有原文支持」用二元组重合近似，能抓编造，抓不住「引对了章节但理解错了」；那要 judge，留到推荐质量 judge 一起做。
- quote_locate 的句子是从 digest 里抽的，用户不会真的这么问；它的价值是 gold 唯一、段落检索的命中能精确判。
- 还没跑：dev 15 条先看阈值和失败形态，再跑 test 45 条。
