# 索引阶段评估：候选供给（2026-10-10 起）

替代 v1 的书籍级检索基准（`docs/retrieval-bench.md`，只作历史）。原因：v1 的 792 对强相关全来自旧 0.6B
检索的候选池，新索引找到的新书一律算不相关；judge 与 200 条人工标注的加权 kappa 只有 0.263；
而且它量的是排序，不是 agent 需要的东西。

## 索引在 agent 里的职责

`search_books(query, k=10)` 收到的是 agent 的 LLM 自己写的正向偏好（工具里再用规则剥掉负向词），
返回 10 本候选；负向约束由 agent 用 `check_term`（全文词频表）和 `check_trope` 在候选里核，最后至少推 3 本。
所以索引做的是**候选供给**：10 本里有多少真满足正向偏好，扣掉违反负向的还剩不剩 3 本。

## 四块

1. **查询集**（`src/retrieval/supply.py`）
   - 主体：95 个 agent 任务（`eval/agent/tasks/`）的正向词拼成的检索串，约束沿用任务的 `negatives_in_text` / `negatives_meta`；
   - 轨迹：agent 真实发出的 `search_books` 查询，从脚本 35 的 `trajectories.redacted.jsonl` 读。这是最忠实的改写：agent 自己的模型按用户原话写的，
     不需要另一个改写模型（用户 10-10 指出）。要凑多种说法，就用不同 `max-steps`、温度或多跑几轮；
   - 手写改写：`eval/retrieval_supply/rewrites.jsonl`（task_id, variant, query）作为补充接口，目前为空；
   - 旧查询：v1 的 59 条，只为它们的 55 个锚点书名。
2. **池**：每个配置对每条查询取 top-20，并集就是要判的（查询，书）对。配置 = 索引目录支持的每个检索器
   （稠密 / BM25 / RRF 融合）加上稠密检索器的**卡过滤**版（`src/retrieval/cardfilter.py`：查询点了题材时，卡题材相符的先排）。
3. **免 judge 指标**：锚点 Hit@10；卡一致率@10（查询点了题材时 top-10 的卡题材相符比例，元素同理）；
   改写重合度（同一意图不同说法的 top-10 Jaccard）；密度清洁率@10（文中词负向按词频表判，未违反的比例）；查询延迟 p50。
4. **judge 指标**（`src/retrieval/supply_judge.py`）：每对一次调用，答 positive 0/1/2 和每条元标签负向是否违反；
   文中词负向由词频表判定后作为已知信息给 judge，不让它判。
   - 正向精确率@10：严格 = positive 为 2 的比例，宽松 = ≥1；
   - 可行@10：同时满足正向、无元标签违反、无词频违反的候选 ≥ 3 本的查询比例。

## 2026-10-27 改法：池就是 agent 看到的

上面的「四块」是重跑索引建池的设计；用户 10-27 定了三件事：judge 用 Qwen3.8-Max、不做人工校准（数字标明是模型判定）、
查询**只用 agent 真实发出的 `search_books`**，不引入改写。既然查询就是轨迹里的，候选也不必重跑索引：
轨迹里每次 `search_books` 的返回（含 agent 自己加的 genre / elements / style 过滤和 k）就是 agent 当时看到的候选，
`finish` 的推荐就是它最后给用户的。所以脚本 44 直接从一次 agent 评测的轨迹抽：

- 查询 = 请求会话里的每次 `search_books`（任务类 constrained_rec / memory / session_state；book_qa 的查询是书名定位、clarify 不该搜，都不算）；
  它的返回按序成为 `agent_seen` 行；会话的推荐成为 `agent_recommended` 行。
- 同一会话的查询共用 `request_id = task/sN`，judge 按 **(请求, 书)** 判一次，prompt 里给的是用户原话 + 该会话的正向偏好
  （memory 任务每个请求会话对应 positives 里的一项）+ 元标签负向；词频判定照旧作为已知信息。
- 同一套判定同时出：索引侧 `positive_precision@10` / `feasible@10`（按查询）、请求可行性（会话看到的全部候选里可用的 ≥ 3，
  写成 `supply_feasible.json` 给脚本 36 的 `--supply`，做条件化通过率）、agent 侧 **推荐满足率**（`agent_recommended` 里 positive 2 / ≥1 的比例、
  违反率、全部可用的会话比例，按任务类分）。

放弃的：配置对比（稠密 / BM25 / 融合 / 卡过滤）。那要重跑索引（PC 显卡几分钟）并把对数乘上配置数，judge 费用随之翻倍；
现在先回答「agent 看到的候选够不够、推荐的书对不对」，配置对比留给脚本 40（代码还在）。

dev-flash-05 冒烟（Mac，零费用）：122 条查询 / 85 个请求会话，agent 的 k 多为 10–15，带 genre 过滤 99 次；
(请求, 书) 对 1,199，推荐 297 本里 14 本不在看到的候选里（similar_books 来的）；卡题材一致率 0.91、密度清洁率 0.89；
`41 --dry-run` T1 估 775 token/对，Max 牌价约 ¥17.6。test 307 条按比例约 3,500 对、¥50，是否全判待用户定。

```bash
uv run python scripts/44_observed_supply.py --run-id test-flash-02                                   # 池 + 免 judge 指标（Mac 可跑）
uv run python scripts/41_supply_judge.py --pool-dir eval/results/retrieval_supply/test-flash-02 --dry-run
uv run python scripts/42_supply_metrics.py --pool-dir eval/results/retrieval_supply/test-flash-02 --verdicts .../verdicts_T1_qwen3.8-max.jsonl
uv run python scripts/36_agent_metrics.py --run-id test-flash-02 --tasks ... --supply eval/results/retrieval_supply/test-flash-02/supply_feasible.json
```

## 定下的与未定的

- judge 用 **Qwen3.8-Max**（用户定，不用 Flash）；池 **top-20**；用户标 **100 对**做校准，只标每对的 positive 和各负向是否违反，可行@10 由这两项算出。
- 证据档位未定：T1 卡 + 简介 + 词频旗标；T2 加 2 段 700 字独立原文；T3 加 5 段；T4 加 3 段 1,500 字完整场景。
  独立原文沿用 v1 采样器，避开 digest 用过的 12 章。候选做法：150 对试点 T1 对 T4，一致率 ≥ 90% 就用 T1。
- 0.6B 对 4B 不比；4B 索引直接成为默认。

## 脚本

```bash
uv run python scripts/40_supply_pool.py --index-dir data/index/single_4b --index-dir data/index/multi_4b   # 池 + 免 judge 指标
uv run python scripts/41_supply_judge.py --dry-run --tier T1                                                   # 对数、token、费用估算
uv run python scripts/41_supply_judge.py --tier T4 --sample 150 --cap 15 --api-key-file ~/.config/aliyun.key   # 试点
uv run python scripts/42_supply_metrics.py --verdicts eval/results/retrieval_supply/verdicts_T1_qwen3.8-max.jsonl
uv run python scripts/43_calibration_sheet.py make --verdicts ... --tier T1                                    # 100 对人工表
uv run python scripts/43_calibration_sheet.py score --sheet ... --verdicts ...
```

判定缓存 `data/cache/supply_judge_cache.jsonl`，键含查询、书、证据哈希、档位、模型、prompt 版本；重跑只为新对付费。
花费由端点返回的 usage 累计，`--cap` 是硬上限。
