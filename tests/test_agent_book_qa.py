"""book_qa: task synthesis from digests, scoring of citations and answers, title matching in search_books."""

from typing import Any

from src.agent.context import ContextBudget
from src.agent.tools import ToolRegistry, build_search_books, title_index, title_matches
from src.agent_eval.metrics import aggregate, answer_support, format_summary, match_chapter, score_qa_session, score_task
from src.agent_eval.tasks import chapter_number, digest_chapters, eligible_qa_books, sampled_chapter_titles, synthesize_book_qa


def chapter(n: int, name: str, body: str) -> str:
    return f"第{n}章 {name}\n{body}"


BODY = "韩立背着竹篓走进青牛镇的药铺，掌柜抬头看了他一眼。\n" * 3
OPENING = [chapter(i, f"开场{i}", BODY + f"少年在第{i}天离开村子，父亲站在门口没有说话。") for i in (1, 2, 3, 4)]
MIDDLE = [chapter(n, f"中段{n}", BODY + f"墨大夫在山洞里点起油灯，第{n}次翻开那本残卷。") for n in (120, 240, 360, 480, 600, 720)]
ENDING = [chapter(n, f"结尾{n}", BODY + "飞升之门在云海尽头缓缓打开，他回头看了一眼人间。") for n in (900, 901)]


def sections_for(opening=OPENING, middle=MIDDLE, ending=ENDING) -> list[dict[str, str]]:
    titles = " / ".join([f"第{n}章 目录章{n}" for n in range(1, 1000, 37)])
    return (
        [{"kind": "blurb", "text": "《测试书》\n作者：某人"}]
        + [{"kind": "opening", "text": t} for t in opening]
        + [{"kind": "titles", "text": "章节目录（抽样）：\n" + titles}]
        + [{"kind": "middle", "text": t} for t in middle]
        + [{"kind": "ending", "text": t} for t in ending]
    )


def fake_books(n: int = 12) -> dict[str, dict[str, Any]]:
    books = {}
    for i in range(n):
        secs = sections_for()
        books[f"id{i:02d}"] = {"title": f"《测试书{i:02d}》", "chapters": digest_chapters(secs), "sampled_titles": sampled_chapter_titles(secs)}
    return books


def test_chapter_numbers_parse_arabic_and_chinese_numerals() -> None:
    assert chapter_number("第184章 龙济光的计划") == 184
    assert chapter_number("第十二回 林教头风雪山神庙") == 12
    assert chapter_number("第一百零三章") == 103
    assert chapter_number("楔子") is None
    assert match_chapter("第184章", "第184章 龙济光的计划") and not match_chapter("第185章", "第184章 龙济光的计划")
    assert match_chapter("龙济光的计划", "第184章 龙济光的计划")


def test_eligibility_requires_twelve_numbered_chapters_and_a_unique_title() -> None:
    books = fake_books(3)
    books["short"] = {"title": "《太短》", "chapters": digest_chapters(sections_for(middle=MIDDLE[:2])), "sampled_titles": []}
    books["dup1"] = {"title": "《重名书》", "chapters": books["id00"]["chapters"], "sampled_titles": []}
    books["dup2"] = {"title": "《重名书》", "chapters": books["id00"]["chapters"], "sampled_titles": []}
    assert eligible_qa_books(books) == ["id00", "id01", "id02"]


def test_book_qa_tasks_cycle_variants_with_gold_from_the_digest() -> None:
    tasks = synthesize_book_qa(fake_books(), 10, seed=3)
    assert [t.variant for t in tasks] == ["chapter_event", "quote_locate", "opening", "ending", "out_of_coverage"] * 2
    assert len({t.meta["novel_id"] for t in tasks}) == 10
    for task in tasks:
        expect = task.sessions[0].expect
        assert task.kind == "book_qa" and expect["novel_id"] == task.meta["novel_id"]
        assert task.positives == [task.meta["title"]] and "《" + task.meta["title"] + "》" in task.sessions[0].user_message
        if task.variant == "out_of_coverage":
            assert expect["answerable"] is False and expect["gold_chapters"] == []
            assert chapter_number(task.meta["chapter"]) not in {chapter_number(h) for _, h, _ in fake_books()["id00"]["chapters"]}
        else:
            assert expect["answerable"] is True and expect["gold_chapters"]
        if task.variant == "quote_locate":
            assert task.meta["quote"] in task.sessions[0].user_message
            assert sum(task.meta["quote"] in text for _, _, text in fake_books()["id00"]["chapters"]) == 1
        if task.variant == "ending":
            assert all(chapter_number(h) in (900, 901) for h in expect["gold_chapters"])
    assert synthesize_book_qa(fake_books(), 10, seed=3)[4].sessions[0].user_message == tasks[4].sessions[0].user_message


def trajectory(answer: str, citations: list[dict[str, str]], asked: str = "id00", termination: str = "finish", returned: tuple[str, ...] = ("第240章 中段240", "第120章 中段120")) -> dict[str, Any]:
    observation = {"tool": "ask_book", "arguments": {"novel_id": asked, "question": "q"}, "result": {"passages": [{"chapter": c, "kind": "middle", "text": "…"} for c in returned]}}
    steps = [{"tool_calls": [{"name": "ask_book", "arguments": {"novel_id": asked, "question": "q"}}], "observations": [observation], "prompt_tokens": 10, "completion_tokens": 5}]
    return {"termination": termination, "final_answer": answer, "structured": {"recommendations": [], "citations": citations}, "steps": steps}


def test_scoring_an_answerable_question() -> None:
    chapters = fake_books(1)["id00"]["chapters"]
    expect = {"novel_id": "id00", "answerable": True, "gold_chapters": ["第240章 中段240"]}
    question = "《测试书00》里「第240章 中段240」那一章讲了什么？"
    good = trajectory("这一章里墨大夫在山洞里点起油灯，翻开那本残卷；韩立背着竹篓走进青牛镇的药铺。", [{"novel_id": "id00", "chapter": "第240章 中段240"}])
    row = score_qa_session(good, expect, chapters, question)
    assert row["labels"] == [] and row["resolved"] and row["citations_grounded"] and row["gold_cited"] and row["answer_support"] > 0.6

    wrong_chapter = trajectory("墨大夫在山洞里点起油灯。", [{"novel_id": "id00", "chapter": "第120章"}])
    assert score_qa_session(wrong_chapter, expect, chapters, question)["labels"] == ["gold_not_cited"]

    invented = trajectory("墨大夫在山洞里点起油灯。", [{"novel_id": "id00", "chapter": "第999章 不存在"}])
    assert score_qa_session(invented, expect, chapters, question)["labels"] == ["ungrounded_citation", "gold_not_cited"]

    unsupported = trajectory("主角在皇城大婚，封侯拜相，与公主白头偕老，举国欢庆三日三夜不息。", [{"novel_id": "id00", "chapter": "第240章"}])
    row = score_qa_session(unsupported, expect, chapters, question)
    assert "answer_unsupported" in row["labels"] and row["answer_support"] < 0.25

    silent = trajectory("讲了一些事。", [])
    assert "no_citation" in score_qa_session(silent, expect, chapters, question)["labels"]

    other_book = trajectory("墨大夫在山洞里点起油灯。", [{"novel_id": "id00", "chapter": "第240章"}], asked="id07")
    assert "book_unresolved" in score_qa_session(other_book, expect, chapters, question)["labels"]


def test_scoring_an_out_of_coverage_question() -> None:
    chapters = fake_books(1)["id00"]["chapters"]
    expect = {"novel_id": "id00", "answerable": False, "gold_chapters": []}
    honest = trajectory("这一章摘要未收录，我只能看到开头四章、六个中段章节和结尾两章。", [])
    row = score_qa_session(honest, expect, chapters)
    assert row["labels"] == [] and row["abstained"] is True and row["gold_cited"] is None
    made_up = trajectory("这一章里主角大战魔尊，最终险胜。", [{"novel_id": "id00", "chapter": "第38章 目录章38"}])
    assert score_qa_session(made_up, expect, chapters)["labels"] == ["ungrounded_citation", "fabricated_answer"]


def test_score_task_dispatches_book_qa_and_the_summary_gains_qa_columns() -> None:
    books = fake_books(1)
    task = {"task_id": "qa-000", "kind": "book_qa", "variant": "chapter_event", "split": "dev",
            "sessions": [{"user_message": "《测试书00》里「第240章 中段240」那一章讲了什么？", "expect": {"novel_id": "id00", "answerable": True, "gold_chapters": ["第240章 中段240"]}}]}
    traj = trajectory("墨大夫在山洞里点起油灯，翻开那本残卷，韩立背着竹篓走进药铺。", [{"novel_id": "id00", "chapter": "第240章 中段240"}])
    row = score_task(task, [traj], [{}], {}, digests=books)
    assert row["pass"] and row["sessions"][0]["gold_cited"]
    summary = aggregate([row])
    assert summary["overall"]["citation_grounded_rate"] == 1.0 and summary["overall"]["book_resolved_rate"] == 1.0
    assert "gold_cited_rate" in format_summary(summary)


def test_answer_support_ignores_bigrams_already_in_the_question() -> None:
    assert answer_support("太短", ["x"]) is None
    assert answer_support("韩立背着竹篓走进青牛镇的药铺，掌柜抬头看了他一眼。", ["韩立背着竹篓走进青牛镇的药铺，掌柜抬头看了他一眼。"], exclude="") == 1.0
    assert answer_support("韩立背着竹篓走进青牛镇的药铺，掌柜抬头看了他一眼。", ["别的内容和这句话没有任何重叠之处可言"], exclude="《测试书》") < 0.2


class Profiles:
    rows = {"a": {"title": "《凡人修仙传》"}, "b": {"title": "《仙逆》"}, "c": {"title": "《重名》"}, "d": {"title": "《重名》"}, "e": {"title": "《凡人》"}}

    def get(self, novel_id: str) -> dict[str, str] | None:
        return self.rows.get(novel_id)


class Searcher:
    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[dict[str, Any]]:
        return [{"novel_id": "b", "title_guess": "《仙逆》", "profile_text_preview": "p", "score": 0.7}, {"novel_id": "a", "title_guess": "《凡人修仙传》", "profile_text_preview": "p", "score": 0.6}]


def test_title_index_and_matching() -> None:
    index = title_index(Profiles())
    assert index == {"凡人修仙传": "a", "仙逆": "b", "凡人": "e"}  # the shared title is left out
    assert title_matches("《凡人修仙传》", index) == ["a"]
    assert title_matches("凡人修仙传的结局怎么样", index) == ["a"]  # 「凡人」 is only two characters, too short to match inside a sentence
    assert title_matches("仙逆", index) == ["b"]
    assert title_matches("热血玄幻", index) == []


def test_search_books_puts_the_named_book_first() -> None:
    registry = ToolRegistry()
    registry.register(build_search_books(Searcher(), ContextBudget(), Profiles()))
    result = registry.call("search_books", {"query": "《凡人修仙传》结局", "k": 2})
    assert [r["novel_id"] for r in result["results"]] == ["a", "b"]
    assert result["results"][0]["title_match"] is True and result["results"][0]["title"] == "《凡人修仙传》"
    plain = registry.call("search_books", {"query": "热血玄幻", "k": 2})
    assert [r["novel_id"] for r in plain["results"]] == ["b", "a"] and "title_match" not in plain["results"][0]


def test_a_gold_chapter_the_tool_never_returned_is_attributed_to_retrieval_not_the_model() -> None:
    chapters = fake_books(1)["id00"]["chapters"]
    expect = {"novel_id": "id00", "answerable": True, "gold_chapters": ["第240章 中段240"]}
    row = score_qa_session(trajectory("只找到别的章节。", [], returned=("第120章 中段120",)), expect, chapters)
    assert row["labels"] == ["chapter_not_retrieved"] and row["gold_retrieved"] is False and row["gold_cited"] is None
    invented = score_qa_session(trajectory("编的。", [{"novel_id": "id00", "chapter": "第999章"}], returned=("第120章 中段120",)), expect, chapters)
    assert invented["labels"] == ["ungrounded_citation", "chapter_not_retrieved"]


def test_quote_question_whose_line_is_not_in_the_returned_window_is_attributed_to_the_tool() -> None:
    from src.agent_eval.metrics import quoted_line, score_qa_session

    chapters = [("middle", "第二章 夜谈", "第二章 夜谈\n他推门进来，说了一句「今夜无人入眠」，然后坐下。" + "闲话" * 50)]
    expect = {"novel_id": "n1", "answerable": True, "gold_chapters": ["第二章 夜谈"]}
    message = "「今夜无人入眠」这句话在《某书》的哪一章？当时发生了什么？"
    assert quoted_line(message) == "今夜无人入眠" and quoted_line("《某书》开头讲了什么？") is None
    window = {"chapter": "第二章 夜谈", "text": "第二章 夜谈\n" + "闲话" * 50}  # the chapter came back, the line did not
    traj = {
        "termination": "finish",
        "steps": [{"tool_calls": [{"name": "ask_book", "arguments": {"novel_id": "n1", "question": "今夜无人入眠"}}], "observations": [{"tool": "ask_book", "arguments": {"novel_id": "n1"}, "result": {"passages": [window]}}]}],
        "structured": {"recommendations": [], "citations": []},
        "final_answer": "库里这本书的摘录里没有这句话，我没法告诉你它在哪一章。",
    }
    row = score_qa_session(traj, expect, chapters, user_message=message)
    assert row["labels"] == ["quote_not_retrieved"] and row["gold_retrieved"] is True and row["quote_retrieved"] is False
    # with the line in the window the model-side labels apply again
    window["text"] = chapters[0][2]
    row = score_qa_session(traj, expect, chapters, user_message=message)
    assert row["quote_retrieved"] is True and "quote_not_retrieved" not in row["labels"] and "no_citation" in row["labels"]
    # the line missing but the chapter cited anyway (inferred from context): a pass, the miss only shows in quote_retrieved
    window["text"] = "第二章 夜谈\n" + "闲话" * 50
    traj["structured"]["citations"] = [{"novel_id": "n1", "chapter": "第二章 夜谈"}]
    traj["final_answer"] = "在第二章 夜谈，他推门进来坐下闲话。"
    row = score_qa_session(traj, expect, chapters, user_message=message)
    assert row["labels"] == [] and row["quote_retrieved"] is False and row["gold_cited"] is True
    # the redacted trajectory has no passage text: no verdict either way
    window.pop("text")
    assert score_qa_session(traj, expect, chapters, user_message=message)["quote_retrieved"] is None
