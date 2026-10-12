"""The web UI's server: a Starlette app over a ``SessionStore``.

Routes (all JSON unless noted)::

    GET    /                              the single-page UI (static/index.html)
    GET    /login, POST /login            the password page when the app was given a token
    GET    /api/info                      the agent bundle's info (model, index, tools)
    GET    /api/books/{novel_id}          one book: title, synopsis and its card (the page's book panel)
    GET    /api/sessions                  this browser's sessions, newest first
    POST   /api/sessions   {title?}       create a session
    GET    /api/sessions/{id}             turns, memory, session state
    PATCH  /api/sessions/{id} {title}     rename
    DELETE /api/sessions/{id}
    POST   /api/sessions/{id}/messages {text}
                                          server-sent events: ``step`` per agent step as it completes,
                                          then ``turn`` with the answer, the memory and the session state;
                                          ``error`` on failure. 409 when the session is already running a turn.

Access: with a ``token`` the whole app sits behind one shared password (a cookie after ``/login``), so a
handful of people can be given the address without giving away the API budget. Every browser gets an
opaque owner cookie and sees only the sessions it created (sessions from before the cookie existed are
visible to all). The agent runs synchronously, so a turn is executed in a worker thread; step summaries
cross back to the event loop through a queue.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import secrets
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Route

from src.agent.sessions import SessionBusy, SessionNotFound, SessionStore

STATIC_DIR = Path(__file__).resolve().parent / "static"
AUTH_COOKIE = "inovel_auth"
OWNER_COOKIE = "inovel_owner"
COOKIE_DAYS = 90

LOGIN_PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>网文 agent</title>
<style>
  :root { --paper: #f3eee4; --panel: #faf7f1; --line: #ddd4c4; --ink: #2a2521; --faint: #a69c8f; --accent: #9c3f1e; }
  @media (prefers-color-scheme: dark) { :root { --paper: #161411; --panel: #1d1a16; --line: #35302a; --ink: #e9e2d5; --faint: #6f675d; --accent: #d9956c; } }
  body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: var(--paper); color: var(--ink); font-family: "Songti SC", "Noto Serif CJK SC", "Source Han Serif SC", Georgia, serif; }
  form { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 28px 32px; width: 300px; }
  h1 { font-size: 20px; font-weight: 500; margin: 0 0 16px; letter-spacing: .04em; }
  input { width: 100%; box-sizing: border-box; padding: 9px 12px; border: 1px solid var(--line); border-radius: 6px; font: inherit; background: var(--paper); color: var(--ink); }
  input:focus { outline: none; border-color: var(--accent); }
  button { margin-top: 12px; width: 100%; padding: 9px; border: 0; border-radius: 6px; background: var(--accent); color: #fbf7f0; font: inherit; cursor: pointer; }
  .err { color: var(--accent); font-size: 14px; margin: 8px 0 0; }
</style></head><body>
<form method="post" action="/login"><h1>网文 agent</h1><input type="password" name="password" placeholder="口令" autofocus autocomplete="current-password">%ERR%<button type="submit">进入</button></form>
</body></html>"""


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _secure(request: Request) -> bool:
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"


def _owner(request: Request) -> str:
    return request.cookies.get(OWNER_COOKIE) or getattr(request.state, "owner", "") or ""


class AccessMiddleware(BaseHTTPMiddleware):
    """Password gate (when a token is set) and the per-browser owner cookie."""

    def __init__(self, app: Any, token: str | None) -> None:
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if self.token and request.url.path != "/login":
            presented = request.cookies.get(AUTH_COOKIE) or ""
            if not hmac.compare_digest(presented, self.token):
                if request.url.path.startswith("/api/"):
                    return JSONResponse({"error": "login required"}, status_code=401)
                return RedirectResponse("/login", status_code=303)
        new_owner = None
        if not request.cookies.get(OWNER_COOKIE):
            new_owner = secrets.token_hex(16)
            request.state.owner = new_owner
        response = await call_next(request)
        if new_owner is not None:
            response.set_cookie(OWNER_COOKIE, new_owner, max_age=COOKIE_DAYS * 86400, httponly=True, samesite="lax", secure=_secure(request))
        return response


def create_app(store: SessionStore, info: dict[str, Any] | None = None, token: str | None = None) -> Starlette:
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    bundle_info = dict(info if info is not None else getattr(store.bundle, "info", {}) or {})
    token = token.strip() if token else None

    async def index(_: Request) -> Response:
        return HTMLResponse(page)

    async def login_page(_: Request) -> Response:
        return HTMLResponse(LOGIN_PAGE.replace("%ERR%", ""))

    async def login(request: Request) -> Response:
        from urllib.parse import parse_qs

        form = parse_qs((await request.body()).decode("utf-8", "replace"))
        presented = (form.get("password") or [""])[0]
        if not token or not hmac.compare_digest(presented, token):
            return HTMLResponse(LOGIN_PAGE.replace("%ERR%", '<p class="err">口令不对</p>'), status_code=401)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(AUTH_COOKIE, token, max_age=COOKIE_DAYS * 86400, httponly=True, samesite="lax", secure=_secure(request))
        return response

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

    async def list_sessions(request: Request) -> Response:
        return JSONResponse({"sessions": store.list(_owner(request))})

    async def create_session(request: Request) -> Response:
        body = await _json_body(request)
        session = store.create(title=str(body.get("title") or ""), owner=_owner(request))
        return JSONResponse(session.to_dict(), status_code=201)

    async def get_session(request: Request) -> Response:
        try:
            session = store.get(request.path_params["sid"], _owner(request))
        except SessionNotFound:
            return JSONResponse({"error": "no such session"}, status_code=404)
        return JSONResponse(session.to_dict())

    async def rename_session(request: Request) -> Response:
        body = await _json_body(request)
        try:
            session = store.rename(request.path_params["sid"], str(body.get("title") or ""), _owner(request))
        except SessionNotFound:
            return JSONResponse({"error": "no such session"}, status_code=404)
        return JSONResponse(session.summary())

    async def delete_session(request: Request) -> Response:
        try:
            store.delete(request.path_params["sid"], _owner(request))
        except SessionNotFound:
            return JSONResponse({"error": "no such session"}, status_code=404)
        except SessionBusy:
            return JSONResponse({"error": "session is busy"}, status_code=409)
        return Response(status_code=204)

    async def send_message(request: Request) -> Response:
        sid = request.path_params["sid"]
        owner = _owner(request)
        body = await _json_body(request)
        text = str(body.get("text") or "").strip()
        if not text:
            return JSONResponse({"error": "text is required"}, status_code=400)
        try:
            session = store.get(sid, owner)
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
                turn = await run_in_threadpool(store.send, sid, text, on_step, owner)
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
        Route("/login", login_page, methods=["GET"]),
        Route("/login", login, methods=["POST"]),
        Route("/api/info", get_info),
        Route("/api/books/{nid}", get_book),
        Route("/api/sessions", list_sessions, methods=["GET"]),
        Route("/api/sessions", create_session, methods=["POST"]),
        Route("/api/sessions/{sid}", get_session, methods=["GET"]),
        Route("/api/sessions/{sid}", rename_session, methods=["PATCH"]),
        Route("/api/sessions/{sid}", delete_session, methods=["DELETE"]),
        Route("/api/sessions/{sid}/messages", send_message, methods=["POST"]),
    ]
    app = Starlette(routes=routes, middleware=[Middleware(AccessMiddleware, token=token)])
    app.state.store = store
    return app


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}
