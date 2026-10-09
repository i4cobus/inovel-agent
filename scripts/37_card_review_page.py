"""Write a standalone HTML page for reviewing book cards by hand, card beside its digest.

    uv run python scripts/37_card_review_page.py --sample 30
    -> data/review/card_review.html   (open in a browser; verdicts stay in that browser's
       localStorage and can be exported to JSON / imported back from the page)

Needs the digest parquet (data/processed/novel_digests.parquet), so it runs where the
data is (the PC); the page itself is a single file that opens anywhere.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from src.config import DATA_DIR, PROJECT_ROOT
from src.digest import DEFAULT_DIGEST_PATH, load_sections
from src.retrieval.card_schema import ELEMENTS, STYLE_OPTIONS
from src.retrieval.cards import DEFAULT_CARDS_PATH, load_cards

app = typer.Typer(add_completion=False)
console = Console()
DEFAULT_OUT = DATA_DIR / "review" / "card_review.html"
DEFAULT_IDS = PROJECT_ROOT / "eval" / "agent" / "cards_pilot_ids.txt"
KIND_NAMES = {"blurb": "简介", "opening": "开头", "titles": "章节名", "middle": "中段", "ending": "结尾", "card": "书卡"}


def sample_order(novel_id: str) -> str:
    return hashlib.md5(f"review:{novel_id}".encode()).hexdigest()


@app.command()
def main(
    cards: Path = typer.Option(DEFAULT_CARDS_PATH),
    digests: Path = typer.Option(DEFAULT_DIGEST_PATH),
    ids: Path = typer.Option(DEFAULT_IDS, help="novel_ids to include, one per line"),
    sample: int = typer.Option(30, help="How many books get the 抽样 mark (stable hash order)."),
    out: Path = typer.Option(DEFAULT_OUT),
) -> None:
    wanted = [line.strip() for line in ids.read_text(encoding="utf-8").splitlines() if line.strip()]
    card_map = load_cards(cards)
    frame = pd.read_parquet(digests, columns=["novel_id", "title_guess", "sections_json"])
    frame = frame[frame["novel_id"].astype(str).isin(set(wanted))]
    sections = load_sections(frame)
    titles = {str(r.novel_id): str(r.title_guess or "") for r in frame.itertuples(index=False)}
    sampled = set(sorted((n for n in wanted if n in card_map), key=sample_order)[:sample])
    rows = []
    for novel_id in wanted:
        card = card_map.get(novel_id)
        if card is None:
            continue
        rows.append(
            {
                "id": novel_id,
                "title": titles.get(novel_id, ""),
                "sampled": novel_id in sampled,
                "genre": card.genre,
                "subgenre": card.subgenre,
                "elements": list(card.elements),
                "style": dict(card.style),
                "protagonist": card.protagonist,
                "setting": card.setting,
                "tone": card.tone,
                "one_liner": card.one_liner,
                "keywords": list(card.keywords),
                "dropped": list(card.dropped),
                "evidence": dict(card.evidence),
                "sections": [{"kind": KIND_NAMES.get(s.kind, s.kind), "text": s.text} for s in sections.get(novel_id, [])],
            }
        )
    payload = json.dumps({"cards": rows, "elements": dict(ELEMENTS), "style_options": {k: list(v) for k, v in STYLE_OPTIONS.items()}}, ensure_ascii=False).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    console.print(f"{len(rows)} cards ({len(sampled)} sampled, {sum(1 for r in rows if r['sections'])} with digest) -> {out}  {out.stat().st_size / 1e6:.1f} MB")


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>书卡人工审阅</title>
<style>
:root{
  --bg:#f6f5f1; --panel:#fff; --fg:#1d1f24; --muted:#6b6f7a; --line:#dcdad3; --accent:#3b4fb8; --accent-soft:#e6e9f8;
  --yes:#2f7d4f; --yes-soft:#dff0e4; --no:#8a8e98; --no-soft:#ececea; --unclear:#b7791f; --unclear-soft:#fbefd6; --bad:#b4372f; --bad-soft:#f9e1de;
  --display:"Noto Serif SC","Songti SC","SimSun",serif; --body:"Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif; --mono:"Menlo","Consolas",monospace;
}
@media (prefers-color-scheme: dark){ :root{
  --bg:#17181c; --panel:#202228; --fg:#e8e6df; --muted:#9a9ea8; --line:#34363d; --accent:#8d9cf0; --accent-soft:#2a2f4a;
  --yes:#7fcf9a; --yes-soft:#1f3a2a; --no:#9a9ea8; --no-soft:#2b2d33; --unclear:#e2b765; --unclear-soft:#3d3119; --bad:#ef8a82; --bad-soft:#4a2320; color-scheme:dark } }
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--body);font-size:14px;line-height:1.55;padding:16px}
.app{display:grid;grid-template-rows:auto 1fr;gap:12px;height:100%;max-width:1500px;margin:0 auto}
header{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:baseline}
h1{font-family:var(--display);font-size:22px;margin:0}
.stat{font-size:12.5px;color:var(--muted);font-variant-numeric:tabular-nums}
.stat b{color:var(--fg)}
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-left:auto}
.filters select,.filters input[type=search],button{font:inherit;font-size:13px;padding:5px 8px;border:1px solid var(--line);border-radius:4px;background:var(--panel);color:var(--fg)}
button{cursor:pointer} button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.filters label{font-size:13px;display:flex;gap:5px;align-items:center}
.main{display:grid;grid-template-columns:300px minmax(0,1fr) minmax(0,1fr);gap:12px;min-height:0}
.pane{background:var(--panel);border:1px solid var(--line);border-radius:6px;min-height:0;overflow-y:auto}
.item{padding:9px 12px;border-bottom:1px solid var(--line);cursor:pointer;display:grid;gap:2px}
.item:hover{background:var(--accent-soft)} .item.on{background:var(--accent-soft);box-shadow:inset 3px 0 0 var(--accent)}
.item .t{display:flex;gap:6px;align-items:baseline;font-weight:500} .item .t span:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.item .o{font-size:12px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tag{font-size:11px;padding:0 6px;border-radius:3px;flex:none}
.tag.s{background:var(--accent-soft);color:var(--accent)} .tag.d{background:var(--yes-soft);color:var(--yes)}
.card{padding:16px 18px;display:grid;gap:14px}
h3{font-family:var(--display);font-size:19px;margin:0}
h4{font-size:11.5px;color:var(--muted);letter-spacing:.06em;text-transform:uppercase;margin:0 0 6px;font-weight:500}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{display:inline-flex;align-items:center;gap:5px;padding:2px 9px;border-radius:999px;font-size:12.5px;background:var(--no-soft);border:1px solid transparent}
.chip.g{background:var(--accent-soft);color:var(--accent);font-weight:500}
.chip.y{background:var(--yes-soft);color:var(--yes);font-weight:500} .chip.u{background:var(--unclear-soft);color:var(--unclear)} .chip.n{color:var(--muted)}
.chip.v{cursor:pointer;user-select:none} .chip.v.ok{border-color:var(--yes)} .chip.v.bad{border-color:var(--bad);background:var(--bad-soft);color:var(--bad)}
.chip .m{font-family:var(--mono);font-size:11px;opacity:.8}
.one{font-family:var(--display);font-size:15.5px;line-height:1.6;border-left:3px solid var(--accent);padding-left:12px;margin:0}
.kv{display:grid;grid-template-columns:60px 1fr;gap:5px 10px;font-size:13.5px;margin:0} .kv dt{color:var(--muted)} .kv dd{margin:0;overflow-wrap:anywhere}
.review{border-top:1px dashed var(--line);padding-top:12px;display:grid;gap:10px}
.row{display:grid;grid-template-columns:110px 1fr;gap:6px 10px;align-items:center;font-size:13px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:4px;overflow:hidden}
.seg button{border:0;border-radius:0;padding:4px 10px;background:var(--panel)} .seg button+button{border-left:1px solid var(--line)}
.seg button.on{background:var(--accent);color:#fff}
textarea,input[type=text]{font:inherit;font-size:13px;width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:4px;background:var(--panel);color:var(--fg)}
.hint{font-size:12px;color:var(--muted)}
.digest{padding:14px 18px;display:grid;gap:8px;align-content:start}
details{border:1px solid var(--line);border-radius:4px;padding:6px 10px} details[open]{background:var(--bg)}
summary{cursor:pointer;font-weight:500;font-size:13px} summary span{color:var(--muted);font-weight:400;font-size:12px;margin-left:6px}
pre{white-space:pre-wrap;font-family:var(--body);font-size:13px;line-height:1.7;margin:8px 0 0;max-height:60vh;overflow-y:auto}
.empty{color:var(--muted);padding:30px;text-align:center}
.nav{display:flex;gap:8px;align-items:center}
.nav .sp{flex:1}
@media (max-width:1100px){ .main{grid-template-columns:260px minmax(0,1fr)} .digest{grid-column:1 / -1} }
@media (max-width:700px){ body{height:auto} .app{height:auto} .main{grid-template-columns:1fr} .pane{max-height:60vh} }
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>书卡人工审阅</h1>
    <span class="stat" id="stats"></span>
    <div class="filters">
      <label><input type="checkbox" id="fSample" checked> 只看抽样</label>
      <label><input type="checkbox" id="fTodo"> 只看未审</label>
      <select id="fGenre"><option value="">全部题材</option></select>
      <select id="fTrope"><option value="">任一元素</option></select>
      <input id="fQ" type="search" placeholder="搜书名、一句话、关键词">
      <button id="btnExport">导出审阅 JSON</button>
      <label><input type="file" id="fileImport" accept="application/json" hidden><button type="button" id="btnImport">导入</button></label>
    </div>
  </header>
  <div class="main">
    <div class="pane" id="list"></div>
    <div class="pane card" id="card"><div class="empty">左侧选一本</div></div>
    <div class="pane digest" id="digest"><div class="empty">这里显示 digest 原文</div></div>
  </div>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const CARDS = DATA.cards, ELEMENTS = DATA.elements, STYLE = DATA.style_options;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const KEY = 'cardReview.verdicts.v2';
let verdicts = {}; try { verdicts = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(verdicts)); } catch (e) {} renderStats(); };
const V = id => (verdicts[id] ||= {fields:{}, elements:{}, style:{}, missing:'', note:'', done:false});

let selected = null; try { selected = localStorage.getItem('cardReview.selected'); } catch (e) {}
const genres = [...new Set(CARDS.map(c => c.genre))].sort();
$('#fGenre').innerHTML += genres.map(g => `<option>${esc(g)}</option>`).join('');
$('#fTrope').innerHTML += Object.keys(ELEMENTS).map(t => `<option>${esc(t)}</option>`).join('');
for (const id of ['fSample','fTodo','fGenre','fTrope','fQ']) $('#'+id).addEventListener('input', renderList);

function renderStats() {
  const done = CARDS.filter(c => verdicts[c.id]?.done); const sampled = CARDS.filter(c => c.sampled);
  let eOk = 0, eBad = 0, sOk = 0, sBad = 0, fOk = 0, fBad = 0;
  for (const c of done) { const v = verdicts[c.id];
    for (const x of Object.values(v.elements)) { if (x === 'ok') eOk++; else if (x === 'bad') eBad++; }
    for (const x of Object.values(v.style)) { if (x === 'ok') sOk++; else if (x) sBad++; }
    for (const x of Object.values(v.fields)) { if (x === 'ok') fOk++; else if (x) fBad++; } }
  $('#stats').innerHTML = `共 <b>${CARDS.length}</b> 张 · 抽样 <b>${sampled.length}</b> · 已审 <b>${done.length}</b>（抽样里 ${sampled.filter(c=>verdicts[c.id]?.done).length}）· 元素 对 <b>${eOk}</b> 错 <b>${eBad}</b> · 风格 对 <b>${sOk}</b> 偏 <b>${sBad}</b> · 描述字段 对 <b>${fOk}</b> 有误 <b>${fBad}</b>`;
}
function filtered() {
  const s = $('#fSample').checked, td = $('#fTodo').checked, g = $('#fGenre').value, t = $('#fTrope').value, q = $('#fQ').value.trim().toLowerCase();
  return CARDS.filter(c => (!s || c.sampled) && (!td || !verdicts[c.id]?.done) && (!g || c.genre === g) && (!t || c.elements.includes(t)) &&
    (!q || [c.title, c.subgenre, c.one_liner, c.protagonist, c.setting, ...c.keywords, ...c.elements].join(' ').toLowerCase().includes(q)));
}
function renderList() {
  const rows = filtered();
  $('#list').innerHTML = rows.length ? rows.map(c => `<div class="item${c.id===selected?' on':''}" data-id="${c.id}">
    <div class="t"><span>${esc(c.title||'（无书名）')}</span><span class="stat">${esc(c.genre)}</span>${c.sampled?'<span class="tag s">抽样</span>':''}${verdicts[c.id]?.done?'<span class="tag d">已审</span>':''}</div>
    <div class="o">${esc(c.one_liner)}</div></div>`).join('') : '<div class="empty">没有符合条件的书</div>';
  if (!rows.some(c => c.id === selected) && rows.length) select(rows[0].id); else renderCard();
  renderStats();
}
$('#list').addEventListener('click', e => { const it = e.target.closest('.item'); if (it) select(it.dataset.id); });
function select(id) { selected = id; try { localStorage.setItem('cardReview.selected', id); } catch (e) {}
  for (const it of document.querySelectorAll('.item')) it.classList.toggle('on', it.dataset.id === id); renderCard(); }
function step(d) { const rows = filtered(); const i = rows.findIndex(c => c.id === selected); const n = rows[i + d]; if (n) { select(n.id); document.querySelector(`.item[data-id="${n.id}"]`)?.scrollIntoView({block:'nearest'}); } }

const FIELDS = [['subgenre','题材二级'],['one_liner','一句话'],['protagonist','主角'],['setting','背景'],['tone','气质'],['keywords','关键词']];
function renderCard() {
  const c = CARDS.find(x => x.id === selected); const el = $('#card'), dg = $('#digest');
  if (!c) { el.innerHTML = '<div class="empty">左侧选一本</div>'; dg.innerHTML = ''; return; }
  const v = V(c.id);
  const chip = t => { const st = v.elements[t] || ''; const q = (c.evidence||{})[t]; return `<span class="chip y v ${st}" data-t="${esc(t)}" title="${esc(ELEMENTS[t]||'')}${q?'\n依据：'+esc(q):''}">${esc(t)}${q?`<span class="m">“${esc(q)}”</span>`:''}<span class="m">${st==='ok'?'✓':st==='bad'?'✗':'·'}</span></span>`; };
  const seg = (name, opts, cur, group) => `<span class="seg" data-f="${name}" data-g="${group}">${opts.map(([k,l]) => `<button type="button" data-v="${k}" class="${cur===k?'on':''}">${l}</button>`).join('')}</span>`;
  const styleRows = Object.entries(STYLE).map(([dim, opts]) => `<div class="row"><span>${esc(dim)}<br><span class="hint">卡片：<b>${esc(c.style[dim]||'（空）')}</b></span></span>${seg(dim, [['ok','对'],['off1','偏一档'],['off2','差很多']], v.style[dim]||'', 'style')}</div>`).join('');
  el.innerHTML = `
    <div class="nav"><button type="button" id="prev">‹ 上一本</button><span class="sp"></span><span class="stat">${esc(c.sampled?'抽样':'')}</span><span class="sp"></span><button type="button" id="next">下一本 ›</button></div>
    <h3>${esc(c.title||'（无书名）')}</h3>
    <div class="chips"><span class="chip g">${esc(c.genre)}${c.subgenre?' · '+esc(c.subgenre):''}</span>${Object.entries(c.style).filter(([,x])=>x).map(([d,x])=>`<span class="chip">${esc(d)}·${esc(x)}</span>`).join('')}</div>
    <p class="one">${esc(c.one_liner)}</p>
    ${(c.evidence||{}).subgenre ? `<div class="hint">二级依据：“${esc(c.evidence.subgenre)}”</div>` : ''}
    <dl class="kv"><dt>主角</dt><dd>${esc(c.protagonist)}</dd><dt>背景</dt><dd>${esc(c.setting)}</dd><dt>气质</dt><dd>${esc(c.tone)}</dd><dt>关键词</dt><dd><div class="chips">${c.keywords.map(k=>`<span class="chip">${esc(k)}</span>`).join('')||'<span class="hint">无</span>'}</div></dd>${c.dropped.length?`<dt>词表外</dt><dd class="hint">${c.dropped.map(esc).join('、')}（模型想说但词表没有，已丢弃）</dd>`:''}</dl>
    <div><h4>元素（${c.elements.length}）· 点一下 ✓ 对，再点 ✗ 错，再点清除</h4><div class="chips">${c.elements.map(chip).join('')||'<span class="hint">无</span>'}</div></div>
    <div class="review">
      <h4>人工审阅 · 风格档位</h4>
      ${styleRows}
      <h4>人工审阅 · 描述字段</h4>
      ${FIELDS.map(([f,l]) => `<div class="row"><span>${l}</span>${seg(f, [['ok','对'],['part','部分对'],['bad','错']], v.fields[f]||'', 'fields')}</div>`).join('')}
      <div class="row"><span>漏掉的元素</span><input type="text" id="missing" value="${esc(v.missing)}" placeholder="书里有但卡片没列的元素，逗号分隔"></div>
      <div class="row" style="align-items:start"><span>备注</span><textarea id="note" rows="3">${esc(v.note)}</textarea></div>
      <div class="row"><span></span><label><input type="checkbox" id="done" ${v.done?'checked':''}> 这本审完了</label></div>
      <div class="hint">审阅结果存在这个浏览器里，右上角可导出 JSON。novel_id ${esc(c.id)}</div>
    </div>`;
  el.querySelectorAll('.chip.v').forEach(ch => ch.addEventListener('click', () => { const t = ch.dataset.t; const cur = v.elements[t] || ''; v.elements[t] = cur === '' ? 'ok' : cur === 'ok' ? 'bad' : ''; if (!v.elements[t]) delete v.elements[t]; save(); renderCard(); }));
  el.querySelectorAll('.seg').forEach(sg => sg.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; const f = sg.dataset.f, bag = v[sg.dataset.g]; bag[f] = bag[f] === b.dataset.v ? '' : b.dataset.v; if (!bag[f]) delete bag[f]; save(); renderCard(); }));
  $('#missing').addEventListener('input', e => { v.missing = e.target.value; save(); });
  $('#note').addEventListener('input', e => { v.note = e.target.value; save(); });
  $('#done').addEventListener('change', e => { v.done = e.target.checked; save(); renderList(); });
  $('#prev').addEventListener('click', () => step(-1)); $('#next').addEventListener('click', () => step(1));
  dg.innerHTML = c.sections.length ? `<h4>digest 原文（${c.sections.length} 段，书卡只读了这些）</h4>` + c.sections.map((s, i) => `<details ${i===0||s.kind==='章节名'?'open':''}><summary>${esc(s.kind)}<span>${s.text.length} 字</span></summary><pre>${esc(s.text)}</pre></details>`).join('') : '<div class="empty">这本没有 digest</div>';
  dg.scrollTop = 0;
}
$('#btnExport').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify({exported_at: new Date().toISOString(), verdicts}, null, 1)], {type: 'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'card_review_verdicts.json'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
$('#btnImport').addEventListener('click', () => $('#fileImport').click());
$('#fileImport').addEventListener('change', e => { const f = e.target.files[0]; if (!f) return; const r = new FileReader();
  r.onload = () => { try { const d = JSON.parse(r.result); verdicts = d.verdicts || d; save(); renderList(); } catch (err) { $('#stats').textContent = '导入失败：不是审阅 JSON'; } }; r.readAsText(f); });
document.addEventListener('keydown', e => { if (e.target.matches('input,textarea,select')) return; if (e.key === 'j' || e.key === 'ArrowDown') step(1); if (e.key === 'k' || e.key === 'ArrowUp') step(-1); });
renderList();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app()
