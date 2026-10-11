# agent dev 评测：两轮，两个 API 模型（2026-10-12）

dev 集 20 条任务（约束推荐 14、记忆 6），硬指标由脚本 36 计算（`src/agent_eval/metrics.py`），不用 judge。
两个模型都走百炼 Token Plan，`enable_thinking=false`，6 并发，查询编码器 4B 在 PC 显卡上；每轮每模型约 5 分钟。

| 轮 | 模型 | pass | finish 率 | 步/会话 | 提示 token/会话 | check_term 覆盖 | check_trope 覆盖 | 失败标签 |
|---|---|---|---|---|---|---|---|---|
| 1 | Qwen3.8-Flash | 0.55 | 1.00 | 6.5 | 44.8k | 0.88 | 0.71 | tool_skipped 8，no_recommendation 1 |
| 1 | Qwen3.7-Plus | 0.30 | 0.79 | 7.0 | 49.9k | 0.79 | 0.13 | tool_skipped 9，format 5，budget_exhausted 3 |
| 2 | Qwen3.8-Flash | **0.75** | 1.00 | 6.5 | 47.1k | 0.88 | 0.89 | tool_skipped 4，no_recommendation 1 |
| 2 | Qwen3.7-Plus | 0.70 | 0.93 | 6.3 | 45.7k | 1.00 | 0.43 | tool_skipped 4，format 2，no_recommendation 2 |

第 1 轮的分数是按修正后的评分规则重算的（见下）。

## 第 1 轮暴露的问题与改动

1. **finish 被截断**：每步完成 token 上限 1,024，长回答加推荐列表超过它，参数变成非法 JSON，58 个会话里 9 个撞上。上限改 4,096，错误提示里写明「可能超长被截断」。
2. **只陈述偏好的消息没有 finish**（Plus 3 次 `answer_without_finish`）：提示改成每一轮都以 finish 结束，写完记忆用 `asks_user=true` 问用户想看什么；nudge 文案同步。
3. **标签类负向没逐本 check_trope**：两个模型都倾向于读 get_profile 的书卡就当核查过。提示明确「每个标签类负向、每本推荐书都要调一次 check_trope（它先查卡，几乎不花时间）」。Flash 的覆盖从 0.71 到 0.89，Plus 从 0.13 到 0.43。
4. **把元素当题材**（`genre="种田经营"`，Flash 每轮 5 次）：过滤器现在直接回「是元素，不是题材，放进 elements」，模型下一步就改对。
5. **评分规则错误**：记忆任务的会话若没写 `must_pass`，评分回退到任务级负向词，于是 oneoff 变体的第二个会话被要求继续核查上一轮的一次性条件，按构造必失败。改为按会话自己的期望评分；同一句话里持久化的词在本轮也算约束。

## 怎么选

Flash 在工具协议上更听话：finish 率 1.0、check_trope 覆盖更高、无 budget 耗尽；Plus 第二轮记忆类 6/6 全过，但约束推荐 14 条里仍有 2 条没给出推荐、2 条没以 finish 结束，标签核查覆盖只有 0.43。
dev 只有 20 条，一条任务等于 5 个百分点，0.75 对 0.70 本身分不出高下；分得出的是协议遵守和费用：Flash 一轮约 ¥1.2，Plus 约 ¥3.4。

结论：主模型用 Qwen3.8-Flash；Plus 和本地 9B 作对照 arm，只在 test 上各跑一次。Max 作上限 arm 留到 test。
