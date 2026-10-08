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
    "游戏": ["升级流", "公会争霸", "策略", "热血"],
    "种田": ["经营", "慢节奏", "治愈", "家长里短"],
    "西幻": ["冒险", "小队成长", "世界观完整", "骑士与法师"],
}
IN_TEXT_POOL = sorted(IN_TEXT_NEGATIVES - {"恋爱", "校园"})  # the two read as topics, not tropes, in a negation
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
        if stratum == "in_text":
            in_text, meta = rng.sample(IN_TEXT_POOL, k=rng.randint(1, 2)), []
        elif stratum == "meta":
            in_text, meta = [], rng.sample(META_POOL, k=rng.randint(1, 2))
        else:
            in_text, meta = [rng.choice(IN_TEXT_POOL)], [rng.choice(META_POOL)]
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
        terms = rng.sample(IN_TEXT_POOL, k=2)
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
