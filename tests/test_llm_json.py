from src.llm_json import extract_json_object


def test_json_survives_a_reasoning_trace_containing_braces() -> None:
    """Qwen3 thinks by default; a greedy {.*} spans from the trace to the answer."""

    text = (
        "<think>\n先看 {关键点}：题材匹配，但 {慢热} 无法确认。\n</think>\n\n"
        '{"llm_match_score":0.62,"confidence":"medium","matched_preferences":["仙侠"]}'
    )
    assert extract_json_object(text)["llm_match_score"] == 0.62


def test_json_survives_a_truncated_reasoning_trace() -> None:
    text = '思考被截断 {残留\n</think>\n{"llm_match_score":0.4,"confidence":"low"}'
    assert extract_json_object(text)["llm_match_score"] == 0.4


def test_nested_objects_are_matched_by_depth() -> None:
    assert extract_json_object('{"a":{"b":1},"c":2}')["c"] == 2
