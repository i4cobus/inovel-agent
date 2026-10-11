# agent 层改造（2026-10-11）

起因：梳理现状时发现两件事。一是系统提示里写了 `ask_book` 和 citations，代码里没有这个工具；二是
`search_books` 把 agent 写的查询先过 v1 的偏好解析器 `retrieval_query`，对自然语言只是按标点切开再拼回，
否定词原样进了向量检索。用户随后定下方向：用户用自然语言对话，agent 自己构造查询，主模型改用 API，本地
9B 留作对照。

## 改了什么

| 项 | 之前 | 之后 |
|---|---|---|
| 查询 | `retrieval_query` 解析后送检索 | 原样送检索；工具说明改成自然语言描述的示例 |
| 书卡 | 只在 `check_trope` 逐本查 | `search_books` 的 `genre` / `elements` / `style` 硬过滤（`src/retrieval/catalog.py`，FAISS id 选择器做精确过滤），结果行带题材标签 |
| 类比找书 | 无 | `similar_books`：锚书自己的向量在单向量索引里搜（`SingleVectorSearcher.similar`） |
| 进书问答 | 提示里有、代码里没有 | `ask_book`（`src/agent/passages.py`）：按书重切 digest 段落，向量从 `multi_4b` 内存映射按行取，答不到的说未收录；`finish.citations` 改为 `{novel_id, chapter}` |
| 会话 | 只有文字历史 | `SessionState`（`src/agent/session.py`）：已推荐（编号）、已排除、已展示；`exclude_shown`、`set_aside`；注入系统提示 |
| 澄清 | 无 | `finish.asks_user` |
| 系统提示 | 推荐流程 | 能力清单、不能做的事、工具配合、流程、澄清规则 |
| 模型 | Ollama 9B | `--api-key-file`、`--no-thinking`（百炼 `enable_thinking=false`）、`--workers` 并发；`AgentBundle.fork` 让每个任务有自己的记忆、会话、工具和 loop，后端共享 |

`search_books` 的返回从列表改成 `{"results": [...], "filter"?, "filter_matches"?, "excluded"?}`；
评测指标里读检索结果的地方（`searched_ids`）同时认两种形状。

## ask_book 的边界

一本 300 章的书，digest 收录 12 章加目录，约 4% 正文。开头类（设定、金手指、主角来路）和结局类问题通常
答得了，中段具体情节多半答不了，工具返回里的 `coverage` 列出可查的章节范围。实测 `multi_4b`：按书重切的
块键与索引记录完全一致（书卡行是额外的一行，被忽略），内存映射打开 0.9 秒，取一本书的向量约 1 毫秒。
全书按需建索引是第二版，不推翻这一版。

## 待跑

dev 评测两轮（Flash 对 Plus，各约 ¥2）定主模型；轨迹里的 `search_books` 查询回灌索引阶段评估。
