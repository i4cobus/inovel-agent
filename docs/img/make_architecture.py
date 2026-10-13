"""Generate architecture.svg and architecture.en.svg next to this file (python3 docs/img/make_architecture.py)."""
from pathlib import Path

W = 1000
FONT = '-apple-system, "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Microsoft YaHei", "Segoe UI", Helvetica, Arial, sans-serif'
PAPER, INK, MUTED, EDGE = "#faf8f3", "#2b2a28", "#6b665e", "#cfc9bd"
OLIVE, VERM, TEAL = "#6b7a3a", "#b5452f", "#2f6f73"

def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

class Doc:
    def __init__(self): self.parts = []
    def box(self, x, y, w, h, lines, stroke=EDGE, fill="#ffffff", title_color=INK, size=13, dash=False):
        extra = ' stroke-dasharray="5 4"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.2"{extra}/>')
        ty = y + 20
        for i, line in enumerate(lines):
            weight = ' font-weight="600"' if i == 0 else ""
            color = title_color if i == 0 else MUTED
            fs = size if i == 0 else size - 1
            self.parts.append(f'<text x="{x+12}" y="{ty}" font-size="{fs}" fill="{color}"{weight}>{esc(line)}</text>')
            ty += 18
    def rows(self, x, y, pairs, size=13, key_w=48):
        for k, v in pairs:
            self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{VERM}" font-weight="600">{esc(k)}</text>')
            self.parts.append(f'<text x="{x+key_w}" y="{y}" font-size="{size}" fill="{INK}">{esc(v)}</text>')
            y += 21
    def layer(self, y, h, title, color):
        self.parts.append(f'<rect x="20" y="{y}" width="{W-40}" height="{h}" rx="10" fill="none" stroke="{color}" stroke-width="1.6"/>')
        self.parts.append(f'<text x="40" y="{y+24}" font-size="14" font-weight="700" fill="{color}">{esc(title)}</text>')
    def arrow(self, x1, y1, x2, y2, color=MUTED, dash=False, both=False):
        extra = ' stroke-dasharray="5 4"' if dash else ""
        start = ' marker-start="url(#ar)"' if both else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.4" marker-end="url(#ae)"{start}{extra}/>')
    def elbow(self, x1, y1, xm, y2, x2, color=MUTED):
        self.parts.append(f'<path d="M{x1},{y1} H{xm} V{y2} H{x2}" fill="none" stroke="{color}" stroke-width="1.4" marker-end="url(#ae)"/>')
    def label(self, x, y, text, color=MUTED, size=12, anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{esc(text)}</text>')
    def render(self, H):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family=\'{FONT}\'>'
                f'<defs><marker id="ae" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{MUTED}"/></marker></defs>'
                f'<rect width="{W}" height="{H}" fill="{PAPER}"/>')
        return head + "".join(self.parts) + "</svg>"

T = {
 "zh": dict(
  off="离线层 · 每本书跑一次，产物落盘", on="在线层 · 对话时才调模型", ev="评测层 · 两层分开评，失败才能归因",
  raw=["原文"], digest=["digest", "约 4 万字"],
  card=["书卡：LLM 按三层封闭词表填卡", "题材 12/41 · 元素 40（附原文摘录核对）· 风格五维"],
  multi=["多向量索引：段落切块 + 书卡段", "Qwen3-Embedding-4B，294,485 个向量"],
  single=["单向量索引：按段类均值再加权", "一本书一个向量，search_books 用它"],
  dens=["词频表", "全文词频，check_term 用"], prof=["档案表", "书卡 + 简介 + 开头"], also="其他产物：",
  shared="共享后端：编码器、两套索引、档案表、词频表、judge",
  agent="agent loop（Qwen3.8-Flash，OpenAI 兼容 tool calling）",
  rows=[("找书", "search_books（书卡硬过滤）· similar_books"), ("核查", "check_term（词频规则）· check_trope（先书卡后原文）"),
        ("了解", "get_profile · ask_book（书内段落检索，章节引用）"), ("对话", "set_aside · memory_read / memory_write · finish"),
        ("记忆", "长期条目持久化并每轮写进系统提示；仅本次的不持久化"), ("兜底", "每轮最多 10 步；工具结果有字符预算；跨轮只重放最近 8 轮")],
  sess=["会话 = loop 的一个分叉", "自己的记忆文件、已推荐编号、对话记录", "存在 data/sessions/<id>/，重启后恢复"],
  ui=["界面：Starlette + 单页", "工具调用与结果逐步推送（SSE）", "点书名看书卡"],
  idx=["索引阶段：候选供给", "查询取 agent 真实发出的检索词", "top-k 正向精确率、可行性、卡一致率"],
  ag=["agent 阶段：约束推荐 + 跨轮记忆", "程序判分：推荐数、来源、逐本核查、规则违反", "test 75 条：Flash 0.867，9B 对照 0.227"],
  key_w=48),
 "en": dict(
  off="Offline · once per book, written to disk", on="Online · a model is called only while someone is talking", ev="Evaluation · two stages scored apart so failures can be attributed",
  raw=["text"], digest=["digest", "~40k chars"],
  card=["Book card: an LLM fills a closed three-layer vocabulary", "genre 12/41 · 40 elements (quote-verified) · 5 style scales"],
  multi=["Multi-vector index: paragraph chunks + card section", "Qwen3-Embedding-4B, 294,485 vectors"],
  single=["Single-vector index: pooled per kind", "one vector per book; search_books uses it"],
  dens=["Term densities", "full text, for check_term"], prof=["Profile table", "card + synopsis + opening"], also="also derived:",
  shared="shared backends: encoder, both indexes, profiles, densities, judge",
  agent="agent loop (Qwen3.8-Flash, OpenAI-compatible tool calling)",
  rows=[("find", "search_books (hard card filters) · similar_books"), ("verify", "check_term (density rule) · check_trope (card first, then text)"),
        ("read", "get_profile · ask_book (in-book passage retrieval, chapter citations)"), ("talk", "set_aside · memory_read / memory_write · finish"),
        ("memory", "long-term entries persist and enter the system prompt; one-off ones do not"), ("bounds", "≤10 steps per turn; tool results budgeted; only the last 8 turns replayed")],
  sess=["Session = a fork of the loop", "its own memory file, numbered picks, transcript", "under data/sessions/<id>/, restored on restart"],
  ui=["UI: Starlette + one page", "tool calls and results streamed (SSE)", "click a title for its card"],
  idx=["Index stage: candidate supply", "queries the agent actually emitted", "top-k positive precision, feasibility, card consistency"],
  ag=["Agent stage: constrained recommendation + memory", "program-checked: picks, provenance, per-book checks, violations", "75 test tasks: Flash 0.867, 9B arm 0.227"],
  key_w=62),
}

def build(t):
    d = Doc()
    # ---- offline layer
    d.layer(30, 262, t["off"], OLIVE)
    d.box(40, 84, 70, 40, t["raw"], title_color=OLIVE); d.arrow(110, 104, 148, 104)
    d.box(150, 76, 120, 56, t["digest"], title_color=OLIVE)
    d.box(320, 70, 350, 56, t["card"], title_color=OLIVE)
    d.box(320, 148, 350, 56, t["multi"], title_color=OLIVE)
    d.box(700, 148, 260, 56, t["single"], title_color=OLIVE)
    d.elbow(270, 104, 295, 98, 318); d.elbow(270, 104, 295, 176, 318)
    d.arrow(670, 176, 698, 176)
    d.label(150, 254, t["also"], color=OLIVE, size=12)
    d.box(320, 226, 170, 52, t["dens"], title_color=OLIVE)
    d.box(510, 226, 160, 52, t["prof"], title_color=OLIVE)
    # ---- connector
    d.arrow(500, 292, 500, 330); d.label(512, 315, t["shared"])
    # ---- online layer
    d.layer(332, 262, t["on"], VERM)
    d.box(40, 372, 560, 196, [t["agent"]], title_color=VERM)
    d.rows(56, 418, t["rows"], key_w=t["key_w"])
    d.box(660, 372, 300, 88, t["sess"], title_color=VERM)
    d.box(660, 480, 300, 88, t["ui"], title_color=VERM)
    d.arrow(600, 416, 658, 416); d.arrow(810, 460, 810, 478)
    # ---- connector
    d.arrow(500, 594, 500, 632)
    # ---- evaluation layer
    d.layer(634, 150, t["ev"], TEAL)
    d.box(40, 674, 450, 90, t["idx"], title_color=TEAL)
    d.box(510, 674, 450, 90, t["ag"], title_color=TEAL)
    return d.render(804)

out = Path(__file__).resolve().parent
(out / "architecture.svg").write_text(build(T["zh"]), encoding="utf-8")
(out / "architecture.en.svg").write_text(build(T["en"]), encoding="utf-8")
print("written")
