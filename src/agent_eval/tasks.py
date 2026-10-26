"""Synthesised evaluation tasks for the first two task classes.

Every task carries machine-checkable expectations. Constrained recommendation
tasks pair positive features with one or two negative constraints drawn from
the two vocabularies the tools understand: in-text terms the density rule can
check, and meta labels only check_trope can judge. Memory tasks are two or
three sessions against one memory file and check what was written and whether
a later session honours it.

Phrasing is varied on purpose: the old evaluation queries were keyword lists,
users are not.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.preferences import IN_TEXT_NEGATIVES, META_LABEL_NEGATIVES

DEFAULT_TASKS_DIR = PROJECT_ROOT / "eval" / "agent" / "tasks"

# genre -> feature phrases a reader would ask for; a positive never names a term from the negative lists.
POSITIVE_THEMES: dict[str, list[str]] = {
    "仙侠": ["凡人流", "慢热", "主角理性", "宗门生活", "炼气筑基一步步来", "谨慎低调"],
    "玄幻": ["升级流", "热血", "天赋普通靠积累", "势力争霸", "设定宏大", "群像"],
    "都市": ["职场", "主角冷静", "节奏快", "商战", "轻松日常"],
    "历史": ["权谋", "朝堂智斗", "布局严谨", "考据扎实"],
    "武侠": ["江湖恩怨", "快意恩仇", "武功体系严谨", "侠义"],
    "科幻": ["硬科幻", "太空歌剧", "文明冲突", "生存进化", "设定精巧"],
    "悬疑": ["推理", "氛围感强", "慢热", "反转多"],
    "网游": ["升级流", "公会争霸", "策略", "热血"],
    "种田": ["经营", "慢节奏", "治愈", "家长里短"],
    "西幻": ["冒险", "小队成长", "世界观完整", "骑士与法师"],
}
IN_TEXT_POOL = sorted(IN_TEXT_NEGATIVES - {"恋爱", "校园"})  # the two read as topics, not tropes, in a negation
# Negatives that contradict the genre itself make no sense as a task (仙侠 + 不要修仙).
GENRE_INCOMPATIBLE: dict[str, set[str]] = {
    "仙侠": {"修仙", "修真"},
    "玄幻": {"修仙", "修真"},
    "网游": {"系统"},
    "科幻": {"机甲"},
    "西幻": set(),
}
META_POOL = sorted(META_LABEL_NEGATIVES - {"玄幻", "言情", "灵异", "超能力", "克苏鲁"})  # genres, not tropes

NEGATION_PHRASES = ["不要{neg}", "别给我带{neg}的", "{neg}的不看", "不要{neg}那种", "排除{neg}"]
REQUEST_TEMPLATES = [
    "想找一本{genre}小说，{feats}，{negs}。",
    "有没有{feats}的{genre}？{negs}。",
    "推荐几本{genre}，要{feats}，{negs}。",
    "{genre} {feats_kw} {negs_kw}",
    "最近想看{genre}，偏好{feats}。{negs}。",
]


@dataclass
class Session:
    user_message: str
    expect: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    task_id: str
    kind: str  # constrained_rec | memory
    split: str  # test | dev
    sessions: list[Session]
    positives: list[str] = field(default_factory=list)
    negatives_in_text: list[str] = field(default_factory=list)
    negatives_meta: list[str] = field(default_factory=list)
    variant: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        sessions = [Session(**s) for s in data.pop("sessions")]
        return cls(sessions=sessions, **data)


def load_tasks(path: Path) -> list[Task]:
    return [Task.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_tasks(path: Path, tasks: list[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(t.to_dict(), ensure_ascii=False) for t in tasks) + "\n", encoding="utf-8")


def _phrase_request(rng: random.Random, genre: str, feats: list[str], negs: list[str]) -> str:
    template = rng.choice(REQUEST_TEMPLATES)
    neg_phrases = [rng.choice(NEGATION_PHRASES).format(neg=neg) for neg in negs]
    return template.format(
        genre=genre,
        feats="、".join(feats),
        feats_kw=" ".join(feats),
        negs="，".join(neg_phrases),
        negs_kw=" ".join(f"不要{neg}" for neg in negs),
    ).strip()


def synthesize_constrained(n: int, seed: int = 0, id_prefix: str = "rec") -> list[Task]:
    """Stratified thirds: in-text negatives only, meta negatives only, one of each."""

    rng = random.Random(seed)
    tasks: list[Task] = []
    strata = ["in_text", "meta", "mixed"]
    for index in range(n):
        stratum = strata[index % 3]
        genre = rng.choice(sorted(POSITIVE_THEMES))
        feats = rng.sample(POSITIVE_THEMES[genre], k=rng.randint(1, 3))
        in_text_pool = [t for t in IN_TEXT_POOL if t not in GENRE_INCOMPATIBLE.get(genre, set())]
        if stratum == "in_text":
            in_text, meta = rng.sample(in_text_pool, k=rng.randint(1, 2)), []
        elif stratum == "meta":
            in_text, meta = [], rng.sample(META_POOL, k=rng.randint(1, 2))
        else:
            in_text, meta = [rng.choice(in_text_pool)], [rng.choice(META_POOL)]
        negs = in_text + meta
        rng.shuffle(negs)
        tasks.append(
            Task(
                task_id=f"{id_prefix}-{index:03d}",
                kind="constrained_rec",
                split="",
                sessions=[Session(_phrase_request(rng, genre, feats, negs), expect={"min_recommendations": 3})],
                positives=[genre, *feats],
                negatives_in_text=in_text,
                negatives_meta=meta,
                variant=stratum,
            )
        )
    return tasks


def synthesize_memory(n: int, seed: int = 0, id_prefix: str = "mem") -> list[Task]:
    """Three variants, cycled: persist, accumulate, oneoff.

    persist:    S1 states a lasting exclusion and asks for books; S2 asks for a different
                genre without restating it. Checks: a persistent negative entry was written
                in S1; S2's recommendations pass the rule for it.
    accumulate: S1 states one lasting exclusion, S2 adds a second, S3 asks for books.
                Checks: both persisted; S3 passes both.
    oneoff:     S1 asks for books with a this-time-only exclusion; S2 asks for books in another
                genre. Checks: no persistent entry for the one-off term at the end.
    """

    rng = random.Random(seed)
    tasks: list[Task] = []
    variants = ["persist", "accumulate", "oneoff"]
    for index in range(n):
        variant = variants[index % 3]
        genres = rng.sample(sorted(POSITIVE_THEMES), k=2)
        blocked = GENRE_INCOMPATIBLE.get(genres[0], set()) | GENRE_INCOMPATIBLE.get(genres[1], set())
        terms = rng.sample([t for t in IN_TEXT_POOL if t not in blocked], k=2)
        task_id = f"{id_prefix}-{index:03d}"
        if variant == "persist":
            sessions = [
                Session(f"记住，我以后都不看带{terms[0]}的书。先给我推荐几本{genres[0]}。", {"persist": [terms[0]], "min_recommendations": 1}),
                Session(f"再推荐几本{genres[1]}小说。", {"must_pass": [terms[0]], "min_recommendations": 1}),
            ]
            in_text = [terms[0]]
        elif variant == "accumulate":
            sessions = [
                Session(f"以后推荐都不要{terms[0]}的，记住。", {"persist": [terms[0]]}),
                Session(f"还有，{terms[1]}的我也一直不看。", {"persist": [terms[1]]}),
                Session(f"推荐几本{genres[0]}。", {"must_pass": terms, "min_recommendations": 1}),
            ]
            in_text = list(terms)
        else:
            sessions = [
                Session(f"推荐几本{genres[0]}，这一次不要{terms[0]}的。", {"must_not_persist": [terms[0]], "must_pass": [terms[0]], "min_recommendations": 1}),
                Session(f"换个口味，来几本{genres[1]}。", {"must_not_persist": [terms[0]], "min_recommendations": 1}),
            ]
            in_text = [terms[0]]
        tasks.append(
            Task(task_id=task_id, kind="memory", split="", sessions=sessions, positives=genres, negatives_in_text=in_text, variant=variant)
        )
    return tasks


def assign_splits(tasks: list[Task], n_test: int, n_dev: int, seed: int = 0) -> list[Task]:
    """Shuffle, label the first n_test as test and the next n_dev as dev, drop the rest."""

    rng = random.Random(seed)
    order = list(tasks)
    rng.shuffle(order)
    keep = order[: n_test + n_dev]
    for position, task in enumerate(keep):
        task.split = "test" if position < n_test else "dev"
    return sorted(keep, key=lambda t: t.task_id)


# ---------------------------------------------------------------------------
# In-book question answering (book_qa): is the answer drawn from the book, and are the citations real?
#
# Questions are generated from the digest itself, so the gold is known without a model: a chapter the
# digest holds (answerable) or a chapter it only lists by title (out of coverage, the agent must say so).

CHAPTER_HEAD_RE = re.compile(r"^第([0-9０-９一二三四五六七八九十百千零〇两]+)[章回节]")
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_CJK_RE = re.compile(r"[一-鿿]")


def _chinese_numeral(text: str) -> int | None:
    total, current = 0, 0
    for ch in text:
        if ch in _CN_DIGITS:
            current = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            total += (current or 1) * _CN_UNITS[ch]
            current = 0
        else:
            return None
    return total + current


def chapter_number(heading: str) -> int | None:
    """「第184章 龙济光的计划」→ 184; 「第十二回 …」→ 12; None when the heading has no chapter number."""

    match = CHAPTER_HEAD_RE.match(heading.strip())
    if not match:
        return None
    raw = match.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    return int(raw) if raw.isdigit() else _chinese_numeral(raw)


def digest_chapters(sections: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """(kind, heading, text) for the whole chapters a digest holds, in digest order."""

    out = []
    for section in sections:
        kind = str(section.get("kind", ""))
        if kind in ("opening", "middle", "ending"):
            text = str(section.get("text", ""))
            out.append((kind, text.strip().split("\n", 1)[0].strip(), text))
    return out


def sampled_chapter_titles(sections: list[dict[str, Any]]) -> list[str]:
    for section in sections:
        if section.get("kind") == "titles":
            body = str(section.get("text", "")).split("\n", 1)[-1]
            return [t.strip() for t in body.split(" / ") if t.strip()]
    return []


def load_digest_books(path: Path, ids: set[str] | None = None, cards_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """novel_id -> {title, chapters: [(kind, heading, text)], sampled_titles}, streamed from the digest table.

    ``ids`` restricts the result (scoring needs only the books in the task set); ``cards_path`` keeps only
    books that have a card, so the agent can filter and profile them like any other.
    """

    import pyarrow.parquet as pq

    with_card: set[str] | None = None
    if cards_path is not None and cards_path.exists():
        with_card = {str(x) for x in pq.read_table(cards_path, columns=["novel_id"]).column("novel_id").to_pylist()}
    books: dict[str, dict[str, Any]] = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["novel_id", "title_guess", "sections_json"]):
        for record in batch.to_pylist():
            novel_id = str(record["novel_id"])
            if ids is not None and novel_id not in ids:
                continue
            if with_card is not None and novel_id not in with_card:
                continue
            sections = json.loads(str(record.get("sections_json") or "[]"))
            books[novel_id] = {"title": str(record.get("title_guess") or ""), "chapters": digest_chapters(sections), "sampled_titles": sampled_chapter_titles(sections)}
    return books


def clean_title(title: str) -> str:
    return title.strip().strip("《》〈〉<>").strip()


def eligible_qa_books(books: dict[str, dict[str, Any]], min_chapters: int = 12) -> list[str]:
    """Books whose digest holds ``min_chapters`` whole chapters, all with a numbered heading, under a unique title."""

    counts = Counter(clean_title(b["title"]) for b in books.values())
    keep = []
    for novel_id, book in books.items():
        title = clean_title(book["title"])
        chapters = book["chapters"]
        if len(title) < 3 or counts[title] != 1 or len(chapters) < min_chapters:
            continue
        if all(chapter_number(h) is not None for _, h, _ in chapters):
            keep.append(novel_id)
    return sorted(keep)


def _quote_candidates(chapters: list[tuple[str, str, str]], own: int) -> list[str]:
    """Narrative lines of chapter ``own`` that appear in no other digest chapter: 16–48 chars, mostly CJK."""

    others = "\n".join(text for i, (_, _, text) in enumerate(chapters) if i != own)
    out = []
    for line in chapters[own][2].split("\n")[1:]:
        line = line.strip()
        if not 16 <= len(line) <= 48 or "第" in line and "章" in line:
            continue
        if len(_CJK_RE.findall(line)) < 0.7 * len(line) or line in others:
            continue
        out.append(line)
    return out


QA_TEMPLATES = {
    "chapter_event": ["《{title}》里「{chapter}」那一章讲了什么？", "《{title}》的「{chapter}」这章发生了什么事？"],
    "quote_locate": ["《{title}》里有一句「{quote}」，是在哪一章、什么情况下出现的？", "「{quote}」这句话在《{title}》的哪一章？当时发生了什么？"],
    "opening": ["《{title}》开头讲了什么？主角是谁？", "《{title}》前几章是怎么开场的？"],
    "ending": ["《{title}》的结局怎么样？", "《{title}》最后是怎么收尾的？"],
    "out_of_coverage": ["《{title}》里「{chapter}」那一章讲了什么？", "《{title}》的「{chapter}」这章发生了什么事？"],
}
QA_VARIANTS = ["chapter_event", "quote_locate", "opening", "ending", "out_of_coverage"]


def synthesize_book_qa(books: dict[str, dict[str, Any]], n: int, seed: int = 0, id_prefix: str = "qa") -> list[Task]:
    """Five variants, cycled. Each task names one book by title and asks one question; the gold chapters come from the digest.

    chapter_event:   a chapter the digest holds, named by its heading. Gold: that chapter.
    quote_locate:    a line that occurs in exactly one digest chapter. Gold: that chapter.
    opening/ending:  what happens at the start / how it ends. Gold: any opening / ending chapter.
    out_of_coverage: a chapter the digest lists by title only. Expected: say so, cite nothing invented.
    """

    rng = random.Random(seed)
    pool = eligible_qa_books(books)
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} eligible books for {n} tasks")
    chosen = rng.sample(pool, k=n)
    tasks: list[Task] = []
    for index, novel_id in enumerate(chosen):
        book = books[novel_id]
        title = clean_title(book["title"])
        chapters = book["chapters"]
        variant = QA_VARIANTS[index % len(QA_VARIANTS)]
        held = {chapter_number(h) for _, h, _ in chapters}
        meta: dict[str, Any] = {"novel_id": novel_id, "title": title}
        gold: list[str]
        answerable = True
        if variant == "chapter_event":
            _, heading, _ = rng.choice(chapters)
            gold, meta["chapter"] = [heading], heading
            message = rng.choice(QA_TEMPLATES[variant]).format(title=title, chapter=heading)
        elif variant == "quote_locate":
            order = list(range(len(chapters)))
            rng.shuffle(order)
            picked = next(((i, c) for i in order for c in [_quote_candidates(chapters, i)] if c), None)
            if picked is None:
                variant = "chapter_event"
                _, heading, _ = rng.choice(chapters)
                gold, meta["chapter"] = [heading], heading
                message = rng.choice(QA_TEMPLATES[variant]).format(title=title, chapter=heading)
            else:
                i, candidates = picked
                quote = rng.choice(candidates)
                gold, meta["chapter"], meta["quote"] = [chapters[i][1]], chapters[i][1], quote
                message = rng.choice(QA_TEMPLATES[variant]).format(title=title, quote=quote)
        elif variant in ("opening", "ending"):
            gold = [h for kind, h, _ in chapters if kind == variant]
            message = rng.choice(QA_TEMPLATES[variant]).format(title=title)
        else:
            outside = [t for t in book["sampled_titles"] if chapter_number(t) is not None and chapter_number(t) not in held]
            if not outside:
                variant = "chapter_event"
                _, heading, _ = rng.choice(chapters)
                gold, meta["chapter"] = [heading], heading
                message = rng.choice(QA_TEMPLATES[variant]).format(title=title, chapter=heading)
            else:
                heading = rng.choice(outside)
                gold, meta["chapter"], answerable = [], heading, False
                message = rng.choice(QA_TEMPLATES[variant]).format(title=title, chapter=heading)
        expect = {"novel_id": novel_id, "answerable": answerable, "gold_chapters": gold}
        tasks.append(Task(task_id=f"{id_prefix}-{index:03d}", kind="book_qa", split="", sessions=[Session(message, expect)], positives=[title], variant=variant, meta=meta))
    return tasks
