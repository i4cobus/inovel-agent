"""OpenAI-compatible ``/chat/completions`` transport.

One client drives a locally served model (Ollama, vLLM) and a hosted gateway, so
the judge and the agent loop never import an inference engine directly.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_BASE_URL = os.environ.get("INOVELREC_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_API_KEY_ENV = "INOVELREC_LLM_API_KEY"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_MAX_WORKERS = 16
RETRY_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1", "0.0.0.0")


@dataclass(frozen=True)
class TokenUsage:
    """Token counts as reported by the endpoint, for real (not estimated) costing."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


def extract_usage(data: dict[str, Any]) -> TokenUsage:
    """Read the ``usage`` block, tolerating endpoints that omit it."""

    usage = data.get("usage") or {}
    return TokenUsage(
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
    )


class ChatTransport(Protocol):
    """Minimal chat-completion transport, so tests can stub out the network."""

    def complete(self, prompt: str, max_tokens: int) -> str:
        """Return the assistant message text for a single-turn prompt."""


@dataclass(frozen=True)
class ToolCall:
    """One function call the model asked for. ``arguments`` is None when its JSON was invalid."""

    id: str
    name: str
    arguments: dict[str, Any] | None
    raw_arguments: str = ""


@dataclass(frozen=True)
class ChatResponse:
    """An assistant turn: text, the tool calls it requested, and endpoint-reported usage."""

    content: str
    tool_calls: tuple[ToolCall, ...]
    usage: TokenUsage
    finish_reason: str = ""


class ChatModel(Protocol):
    """Multi-turn chat with tool calling, so the agent loop can run on a stub."""

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int) -> ChatResponse:
        """Return the assistant turn for a message list, with optional tool definitions."""


@dataclass
class HTTPChatTransport:
    """POSTs to an OpenAI-compatible ``/chat/completions`` endpoint."""

    model: str
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS
    temperature: float = 0.0
    # None means "decide from the URL". A locally served model must not be routed
    # through an outbound proxy, but a hosted gateway usually has to be.
    bypass_proxy: bool | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Ollama's /v1 endpoint controls thinking through reasoning_effort ("none" disables it on
    # models that allow it; values are per model, see /api/show). None sends nothing.
    reasoning_effort: str | None = None
    # Sized to the caller's concurrency, not to a module constant. The pool used to
    # be a fixed 32 connections however many workers the caller asked for, so a run
    # driving 80 threads still got 32 in flight and the rest queued: SFT assembly
    # measured 4.6 calls/s against a server benchmarked at 16.
    max_connections: int = DEFAULT_MAX_WORKERS * 2
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.bypass_proxy is None:
            self.bypass_proxy = is_local_url(self.base_url)
        if self.api_key is None:
            self.api_key = os.environ.get(DEFAULT_API_KEY_ENV)
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")

    @property
    def client(self) -> Any:
        """Lazily create a pooled client so imports stay cheap."""

        if self._client is None:
            import httpx

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            # Pool generously: score_many drives many concurrent requests.
            limits = httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=max(self.max_connections // 2, 1),
            )
            # trust_env=False also drops proxy settings, which is exactly what a
            # local vLLM needs: this host exports http_proxy, and the proxy answers
            # a request for 127.0.0.1 by closing the connection. A hosted gateway
            # keeps trust_env so it still reaches the outside world.
            self._client = httpx.Client(
                timeout=self.timeout,
                headers=headers,
                limits=limits,
                trust_env=not self.bypass_proxy,
            )
        return self._client

    def close(self) -> None:
        """Close the underlying connection pool."""

        if self._client is not None:
            self._client.close()
            self._client = None

    def build_payload(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        """Build the chat-completions request body.

        ``extra_body`` is where a reasoning model gets switched off. Qwen3 thinks
        by default and its ``<think>`` block consumed the whole token budget
        before any JSON appeared, so every verdict parsed as a failure.
        """

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            **self.extra_body,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def complete(self, prompt: str, max_tokens: int) -> str:
        """Send one prompt, retrying transient failures with linear backoff."""

        return self.complete_with_usage(prompt, max_tokens)[0]

    def complete_with_usage(self, prompt: str, max_tokens: int) -> tuple[str, TokenUsage]:
        """Like ``complete``, but also returns the endpoint-reported token usage."""

        data = self.post_chat(self.build_payload(prompt, max_tokens))
        return extract_message_content(data), extract_usage(data)

    def build_chat_payload(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int
    ) -> dict[str, Any]:
        """Request body for a multi-turn call with OpenAI-style function tools."""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            **self.extra_body,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if tools:
            # No tool_choice: "auto" is the default everywhere and Ollama rejects the field.
            payload["tools"] = tools
        return payload

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int) -> ChatResponse:
        """One assistant turn over a message list; tool calls come back parsed."""

        data = self.post_chat(self.build_chat_payload(messages, tools, max_tokens))
        return parse_chat_response(data)

    def post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST one chat-completions payload, retrying transient failures with linear backoff."""

        import httpx

        url = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.post(url, json=payload)
                if response.status_code in RETRY_STATUS_CODES:
                    last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
                else:
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPError as exc:
                last_error = exc
            if attempt < self.max_retries - 1 and self.backoff_seconds:
                time.sleep(self.backoff_seconds * (attempt + 1))

        raise RuntimeError(f"Chat completion failed after {self.max_retries} attempts: {last_error}")


def is_local_url(url: str) -> bool:
    """Whether a URL points at this machine.

    Hosts commonly export http_proxy for outbound access, and the proxy has no
    route to 127.0.0.1 — it accepts the connection and closes it, which surfaces
    as "Server disconnected without sending a response" rather than anything
    mentioning proxies.
    """

    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    return host in LOCAL_HOSTS


def extract_message_content(data: dict[str, Any]) -> str:
    """Pull the assistant text out of a chat-completions response."""

    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"No choices in chat completion response: {str(data)[:200]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise ValueError("Chat completion response is missing message content")
    return str(content)


def parse_chat_response(data: dict[str, Any]) -> ChatResponse:
    """Parse an assistant turn, keeping an unparseable tool argument string as evidence.

    A model that emits broken JSON for a tool call has made a distinct, scorable
    mistake (参数错误). Raising here would turn it into a transport failure.
    """

    import json

    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"No choices in chat completion response: {str(data)[:200]}")
    choice = choices[0]
    message = choice.get("message") or {}
    calls: list[ToolCall] = []
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        raw = function.get("arguments")
        if isinstance(raw, dict):
            arguments: dict[str, Any] | None = raw
            raw_text = json.dumps(raw, ensure_ascii=False)
        else:
            raw_text = str(raw or "")
            try:
                parsed = json.loads(raw_text) if raw_text.strip() else {}
                arguments = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                arguments = None
        calls.append(
            ToolCall(
                id=str(call.get("id") or f"call_{index}"),
                name=str(function.get("name") or ""),
                arguments=arguments,
                raw_arguments=raw_text,
            )
        )
    return ChatResponse(
        content=str(message.get("content") or ""),
        tool_calls=tuple(calls),
        usage=extract_usage(data),
        finish_reason=str(choice.get("finish_reason") or ""),
    )
