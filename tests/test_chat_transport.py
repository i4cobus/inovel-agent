import pytest

from src.chat_transport import HTTPChatTransport, extract_message_content, is_local_url


def test_extract_message_content_requires_choices() -> None:
    with pytest.raises(ValueError):
        extract_message_content({"choices": []})


def test_transport_builds_openai_payload_and_auth_header() -> None:
    transport = HTTPChatTransport(model="qwen", base_url="http://host:8000/v1/", api_key="secret")
    payload = transport.build_payload("hello", max_tokens=64)

    assert transport.base_url == "http://host:8000/v1"
    assert payload["model"] == "qwen"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["max_tokens"] == 64
    assert payload["temperature"] == 0.0


def test_transport_rejects_zero_retries() -> None:
    with pytest.raises(ValueError):
        HTTPChatTransport(model="qwen", max_retries=0)


def test_transport_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INOVELREC_LLM_API_KEY", "from-env")
    assert HTTPChatTransport(model="qwen").api_key == "from-env"


def test_local_urls_bypass_the_proxy_but_gateways_do_not() -> None:
    """This host exports http_proxy; the proxy closes connections to 127.0.0.1."""

    assert is_local_url("http://127.0.0.1:8000/v1")
    assert is_local_url("http://localhost:8000/v1")
    assert not is_local_url("https://gateway.example.com/v1")

    local = HTTPChatTransport(model="m", base_url="http://127.0.0.1:8000/v1")
    remote = HTTPChatTransport(model="m", base_url="https://gateway.example.com/v1")
    assert local.bypass_proxy is True
    assert remote.bypass_proxy is False


def test_proxy_bypass_can_be_forced() -> None:
    transport = HTTPChatTransport(model="m", base_url="https://gateway.example.com/v1", bypass_proxy=True)
    assert transport.bypass_proxy is True


def test_chat_payload_carries_tools_and_parses_tool_calls() -> None:
    from src.chat_transport import parse_chat_response

    transport = HTTPChatTransport(model="qwen", base_url="http://127.0.0.1:11434/v1")
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
    payload = transport.build_chat_payload([{"role": "user", "content": "hi"}], tools, max_tokens=8)
    assert payload["tools"] == tools and "tool_choice" not in payload and "reasoning_effort" not in payload
    assert "tools" not in transport.build_chat_payload([], None, 8)
    quiet = HTTPChatTransport(model="qwen", base_url="http://127.0.0.1:11434/v1", reasoning_effort="none")
    assert quiet.build_chat_payload([], None, 8)["reasoning_effort"] == "none"
    assert quiet.build_payload("p", 8)["reasoning_effort"] == "none"

    data = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "a", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}},
                        {"id": "b", "type": "function", "function": {"name": "g", "arguments": "{broken"}},
                        {"type": "function", "function": {"name": "h", "arguments": {"y": 2}}},
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }
    response = parse_chat_response(data)
    assert response.content == "" and response.finish_reason == "tool_calls"
    assert [c.name for c in response.tool_calls] == ["f", "g", "h"]
    assert response.tool_calls[0].arguments == {"x": 1}
    assert response.tool_calls[1].arguments is None and response.tool_calls[1].raw_arguments == "{broken"
    assert response.tool_calls[2].arguments == {"y": 2} and response.tool_calls[2].id == "call_2"
    assert response.usage.prompt_tokens == 3
