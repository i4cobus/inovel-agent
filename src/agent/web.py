"""The web UI's server: a Starlette app over a ``SessionStore``.

Routes (all JSON unless noted)::

    GET    /                              the single-page UI (static/index.html)
    GET    /api/info                      the agent bundle's info (model, index, tools)
    GET    /api/books/{novel_id}          one book: title, synopsis and its card (the page's book panel)
    GET    /api/sessions                  session summaries, newest first
    POST   /api/sessions   {title?}       create a session
    GET    /api/sessions/{id}             turns, memory, session state
    PATCH  /api/sessions/{id} {title}     rename
    DELETE /api/sessions/{id}
    POST   /api/sessions/{id}/messages {text}
                                          server-sent events: ``step`` per agent step as it completes,
                                          then ``turn`` with the answer, the memory and the session state;
                                          ``error`` on failure. 409 when the session is already running a turn.

The agent runs synchronously (one HTTP call per model step), so a turn is executed in a worker
thread; step summaries cross back to the event loop through a queue. Nothing here touches the
corpus beyond what the agent's tools return.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from src.agent.sessions import SessionBusy, SessionNotFound, SessionStore

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(store: SessionStore, info: dict[str, Any] | None = None) -> Starlette:
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    bundle_info = dict(info if info is not None else getattr(store.bundle, "info", {}) or {})

    async def index(_: Request) -> Response:
        return HTMLResponse(page)

    async def get_info(_: Request) -> Response:
        return JSONResponse({**bundle_info, "sessions": len(store.sessions)})

    async def get_book(request: Request) -> Response:
        novel_id = request.path_params["nid"]
        backends = getattr(store.bundle, "backends", {}) or {}
        profiles, catalog = backends.get("profiles"), backends.get("catalog")
        row = profiles.get(novel_id) if profiles is not None else None
        card = catalog.cards.get(novel_id) if catalog is not None else None
        if row is None and card is None:
            return JSONResponse({"error": "no such book"}, status_code=404)
        row = row or {}
        return JSONResponse({
            "novel_id": novel_id,
            "title": row.get("title", ""),
            "blurb": str(row.get("blurb", ""))[:1500],
            "card": card.to_dict() if hasattr(card, "to_dict") else None,
        })

    async def list_sessions(_: Request) -> Response:
        return JSONResponse({"sessions": store.list()})

    async def create_session(request: Request) -> Response:
        body = await _json_body(request)
        session = store.create(title=str(body.get("title") or ""))
        return JSONResponse(session.to_dict(), status_code=201)

    async def get_session(request: Request) -> Response:
        try:
            session = store.get(request.path_params["sid"])
        except SessionNotFound:
            return JSONResponse({"error": "no such session"}, status_code=404)
        return JSONResponse(session.to_dict())

    async def rename_session(request: Request) -> Response:
        body = await _json_body(request)
        try:
            session = store.rename(request.path_params["sid"], str(body.get("title") or ""))
        except SessionNotFound:
            return JSONResponse({"error": "no such session"}, status_code=404)
        return JSONResponse(session.summary())

    async def delete_session(request: Request) -> Response:
        try:
            store.delete(request.path_params["sid"])
        except SessionNotFound:
            return JSONResponse({"error": "no such session"}, status_code=404)
        except SessionBusy:
            return JSONResponse({"error": "session is busy"}, status_code=409)
        return Response(status_code=204)

    async def send_message(request: Request) -> Response:
        sid = request.path_params["sid"]
        body = await _json_body(request)
        text = str(body.get("text") or "").strip()
        if not text:
            return JSONResponse({"error": "text is required"}, status_code=400)
        try:
            session = store.get(sid)
        except SessionNotFound:
            return JSONResponse({"error": "no such session"}, status_code=404)
        if session.lock.locked():
            return JSONResponse({"error": "session is busy"}, status_code=409)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()

        def on_step(step: dict[str, Any]) -> None:  # runs in the worker thread
            loop.call_soon_threadsafe(queue.put_nowait, ("step", step))

        async def worker() -> None:
            try:
                turn = await run_in_threadpool(store.send, sid, text, on_step)
                queue.put_nowait(("turn", {"turn": turn.to_dict(), "session": session.to_dict()}))
            except SessionBusy:
                queue.put_nowait(("error", {"message": "这个会话正在处理上一条消息"}))
            except Exception as exc:  # noqa: BLE001 - reported to the page, not hidden
                queue.put_nowait(("error", {"message": f"{type(exc).__name__}: {exc}"}))
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(worker())

        async def events():
            yield _sse("accepted", {"session_id": sid})
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield _sse(item[0], item[1])
            await task

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    routes = [
        Route("/", index),
        Route("/api/info", get_info),
        Route("/api/books/{nid}", get_book),
        Route("/api/sessions", list_sessions, methods=["GET"]),
        Route("/api/sessions", create_session, methods=["POST"]),
        Route("/api/sessions/{sid}", get_session, methods=["GET"]),
        Route("/api/sessions/{sid}", rename_session, methods=["PATCH"]),
        Route("/api/sessions/{sid}", delete_session, methods=["DELETE"]),
        Route("/api/sessions/{sid}/messages", send_message, methods=["POST"]),
    ]
    app = Starlette(routes=routes)
    app.state.store = store
    return app


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}
