"""Local VS Code chat bridge for the shared Жвуша chat pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import structlog

from src.agent_runtime.digital_scenarios import (
    REQUIRED_DIGITAL_SCENARIO_IDS,
    REQUIRED_EVAL_VARIANTS,
)
from src.bot.middleware.chat_logger import log_interface_message
from src.skills.base import AgentContext

VSCODE_CHAT_LOG_ID = "vscode"
VSCODE_CHAT_CONTEXT_CHAT_ID = -7331
VSCODE_CHAT_SOURCE = "vscode"
VSCODE_CHAT_DISPLAY_TZ = ZoneInfo("Europe/Moscow")
_CHAT_LOG_TAIL_CHUNK_BYTES = 64 * 1024
VSCODE_INTERFACE_CONTEXT = (
    "Текущий канал контакта: VS Code chat через локальный bridge. "
    "Это часть технического тела Жвуши: его можно использовать как материал "
    "для наблюдения, настроения, ограничения, технического ответа или действия, "
    "но сам факт bridge не делает человеческий вопрос запросом статуса готовности."
)
_OPERATOR_MESSAGE_KINDS = {
    "goal_loop_handoff",
    "goal_loop_proof_replay",
    "codex_skill_approval",
    "codex_tier3_approval",
}

logger = structlog.get_logger(__name__)

ChatProcessor = Callable[[str, AgentContext], Awaitable[str | None]]


class VscodeChatBridge:
    """Turns VS Code/Codex messages into normal personal Жвуша chat turns."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        admin_user_id: int,
        processor: ChatProcessor,
        chat_log_id: str = VSCODE_CHAT_LOG_ID,
        context_chat_id: int = VSCODE_CHAT_CONTEXT_CHAT_ID,
    ) -> None:
        self._workspace_root = workspace_root
        self._admin_user_id = admin_user_id
        self._processor = processor
        self._chat_log_id = chat_log_id
        self._context_chat_id = context_chat_id
        self._pending_tasks: set[asyncio.Task[str]] = set()
        self._message_seq = 0

    async def send_message(
        self,
        *,
        text: str,
        sender: str = "user",
        workspace_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append an input message, run the shared pipeline, append the reply."""
        prepared = self._prepare_input_message(
            text=text,
            sender=sender,
            workspace_context=workspace_context,
        )
        if prepared is None:
            return {"text": "", "warnings": ["empty_message"], "messages": []}

        clean_text, context = prepared
        reply = await self._process_and_log_reply(clean_text, context)
        return {
            "text": reply,
            "warnings": [],
            "messages": self.list_messages(limit=80),
        }

    async def enqueue_message(
        self,
        *,
        text: str,
        sender: str = "user",
        workspace_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append an input message and process Жвуша's reply in background."""
        response, start_processing = self.prepare_enqueued_message(
            text=text,
            sender=sender,
            workspace_context=workspace_context,
        )
        if start_processing is not None:
            start_processing()
        return response

    def prepare_enqueued_message(
        self,
        *,
        text: str,
        sender: str = "user",
        workspace_context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Callable[[], None] | None]:
        """Append an input message and return a callback that starts processing."""
        prepared = self._prepare_input_message(
            text=text,
            sender=sender,
            workspace_context=workspace_context,
        )
        if prepared is None:
            return {"text": "", "warnings": ["empty_message"], "messages": []}, None
        clean_text, context = prepared

        def start_processing() -> None:
            task: asyncio.Task[str] = asyncio.create_task(
                self._process_and_log_reply(clean_text, context)
            )
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
            task.add_done_callback(_log_background_task_result)

        return {
            "text": "",
            "warnings": ["response_pending"],
            "messages": self.list_messages(limit=80),
        }, start_processing

    async def drain_pending(self) -> None:
        """Wait for currently queued background replies. Used by tests/shutdown."""
        if not self._pending_tasks:
            return
        await asyncio.gather(*tuple(self._pending_tasks), return_exceptions=True)

    def _prepare_input_message(
        self,
        *,
        text: str,
        sender: str,
        workspace_context: dict[str, Any] | None,
    ) -> tuple[str, AgentContext] | None:
        clean_text = text.strip()
        if not clean_text:
            return None

        source_actor = _normalize_sender(sender)
        message_id = self._next_message_id()
        project_metadata = _project_metadata_from_workspace_context(workspace_context)
        operator_metadata = _operator_metadata_from_workspace_context(workspace_context)
        log_extra: dict[str, Any] = {"interface": "vscode"}
        log_extra.update(operator_metadata)
        log_interface_message(
            log_dir=self._workspace_root / "logs",
            text=clean_text,
            chat_id=self._chat_log_id,
            role="user",
            source=VSCODE_CHAT_SOURCE,
            mode="personal",
            source_actor=source_actor,
            user_id=self._admin_user_id,
            message_id=message_id,
            extra=log_extra,
        )

        interface_context = VSCODE_INTERFACE_CONTEXT
        project_context = project_metadata.get("project_context")
        if isinstance(project_context, str) and project_context:
            interface_context = f"{interface_context} {project_context}"

        context = AgentContext(
            user_id=self._admin_user_id,
            chat_id=self._context_chat_id,
            mode="personal",
            message_id=message_id,
            metadata={
                "source": VSCODE_CHAT_SOURCE,
                "interface": "vscode",
                "source_actor": source_actor,
                "chat_log_id": self._chat_log_id,
                "interface_context": interface_context,
                "return_response_text": True,
                "skip_response_log": True,
                **project_metadata,
                **operator_metadata,
            },
        )
        return clean_text, context

    def _next_message_id(self) -> int:
        self._message_seq += 1
        return self._message_seq

    async def _process_and_log_reply(
        self,
        clean_text: str,
        context: AgentContext,
    ) -> str:
        try:
            reply = (await self._processor(clean_text, context)) or ""
        except Exception as exc:
            logger.warning(
                "vscode_chat_processing_failed", error=str(exc), exc_info=True
            )
            return ""
        if reply:
            log_interface_message(
                log_dir=self._workspace_root / "logs",
                text=reply,
                chat_id=self._chat_log_id,
                role="assistant",
                source=VSCODE_CHAT_SOURCE,
                mode="personal",
                source_actor="zhvusha",
                extra={
                    "interface": "vscode",
                    "reply_to_source_actor": context.metadata.get("source_actor", ""),
                    "reply_to_message_id": context.message_id,
                },
            )
        return reply

    def list_messages(self, *, limit: int = 80) -> list[dict[str, Any]]:
        """Return recent VS Code chat JSONL entries in chronological order."""
        chat_dir = self._workspace_root / "logs" / self._chat_log_id
        if not chat_dir.is_dir():
            return []
        requested = max(1, min(limit, 200))
        raw_reversed: list[str] = []
        for log_file in sorted(chat_dir.glob("chat_*.jsonl"), reverse=True):
            remaining = requested - len(raw_reversed)
            for line in _tail_nonempty_lines_reversed(log_file, limit=remaining):
                raw_reversed.append(line)
                if len(raw_reversed) >= requested:
                    break
            if len(raw_reversed) >= requested:
                break

        messages: list[dict[str, Any]] = []
        for index, line in enumerate(reversed(raw_reversed)):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            raw_ts = str(entry.get("ts", ""))
            entry = _with_display_timestamp(entry)
            entry.setdefault(
                "id",
                f"{raw_ts}:{entry.get('role', '')}:{index}",
            )
            messages.append(entry)
        return messages


def _tail_nonempty_lines_reversed(path: Path, *, limit: int) -> list[str]:
    """Return up to limit non-empty lines from the end of a log file, newest first."""
    if limit <= 0:
        return []
    lines: list[str] = []
    pending = b""
    try:
        with path.open("rb") as file:
            file.seek(0, 2)
            position = file.tell()
            while position > 0 and len(lines) < limit:
                chunk_size = min(_CHAT_LOG_TAIL_CHUNK_BYTES, position)
                position -= chunk_size
                file.seek(position)
                chunk = file.read(chunk_size)
                parts = (chunk + pending).split(b"\n")
                pending = parts[0]
                for raw_line in reversed(parts[1:]):
                    stripped = raw_line.rstrip(b"\r")
                    if not stripped.strip():
                        continue
                    lines.append(stripped.decode("utf-8", errors="replace"))
                    if len(lines) >= limit:
                        break
            if position == 0 and len(lines) < limit:
                stripped = pending.rstrip(b"\r")
                if stripped.strip():
                    lines.append(stripped.decode("utf-8", errors="replace"))
    except OSError:
        return []
    return lines


class VscodeChatHttpServer:
    """Tiny local JSON HTTP server for VS Code and Codex clients."""

    def __init__(
        self,
        *,
        bridge: VscodeChatBridge,
        host: str = "127.0.0.1",
        port: int = 7331,
        max_body_bytes: int = 256 * 1024,
        auth_token: str = "",
    ) -> None:
        self._bridge = bridge
        self._host = host
        self._port = port
        self._max_body_bytes = max_body_bytes
        self._auth_token = auth_token.strip()
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_connection,
            host=self._host,
            port=self._port,
        )
        logger.info("vscode_chat_server_started", host=self._host, port=self._port)

    async def stop(self) -> None:
        if self._server is None:
            return
        await self._bridge.drain_pending()
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        logger.info("vscode_chat_server_stopped", host=self._host, port=self._port)

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        after_response: Callable[[], None] | None = None
        try:
            method, target, headers = await _read_http_request(reader)
            body = await _read_body(reader, headers, max_bytes=self._max_body_bytes)
            status, payload, after_response = await self._handle_request_parts(
                method,
                target,
                body,
                headers,
                defer_background_processing=True,
            )
            _write_json_response(writer, status, payload)
        except Exception as exc:
            logger.warning("vscode_chat_request_failed", error=str(exc))
            _write_json_response(writer, 500, {"error": "internal_error"})
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            if after_response is not None:
                after_response()

    async def _handle_request(
        self,
        method: str,
        target: str,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        status, payload, after_response = await self._handle_request_parts(
            method,
            target,
            body,
            headers,
            defer_background_processing=False,
        )
        if after_response is not None:
            after_response()
        return status, payload

    async def _handle_request_parts(
        self,
        method: str,
        target: str,
        body: bytes,
        headers: dict[str, str] | None = None,
        *,
        defer_background_processing: bool,
    ) -> tuple[int, dict[str, Any], Callable[[], None] | None]:
        request_headers = headers or {}
        if not _origin_allowed(request_headers.get("origin", "")):
            return 403, {"error": "forbidden_origin"}, None
        parsed = urlparse(target)
        if method == "OPTIONS":
            return 204, {}, None
        if method == "GET" and parsed.path == "/health":
            return 200, {"ok": True}, None
        if method == "GET" and parsed.path == "/messages":
            query = parse_qs(parsed.query)
            limit = _parse_int(query.get("limit", ["80"])[0], default=80)
            return 200, {"messages": self._bridge.list_messages(limit=limit)}, None
        if not self._auth_token:
            return 503, {"error": "auth_not_configured"}, None
        if not _authorized(request_headers, self._auth_token):
            return 401, {"error": "unauthorized"}, None
        if method == "POST" and parsed.path == "/message":
            return await self._handle_message_request(
                body,
                defer_background_processing=defer_background_processing,
            )
        return 404, {"error": "not_found"}, None

    async def _handle_message_request(
        self,
        body: bytes,
        *,
        defer_background_processing: bool,
    ) -> tuple[int, dict[str, Any], Callable[[], None] | None]:
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return 400, {"error": "invalid_json"}, None
        if not isinstance(payload, dict):
            return 400, {"error": "invalid_payload"}, None
        text = str(payload.get("text", ""))
        sender = str(payload.get("sender", "user"))
        workspace_context = payload.get("context")
        response, start_processing = self._bridge.prepare_enqueued_message(
            text=text,
            sender=sender,
            workspace_context=workspace_context
            if isinstance(workspace_context, dict)
            else None,
        )
        if defer_background_processing:
            return 202, response, start_processing
        if start_processing is not None:
            start_processing()
        return 202, response, None


async def _read_http_request(
    reader: asyncio.StreamReader,
) -> tuple[str, str, dict[str, str]]:
    request_line = (await reader.readline()).decode("ascii", errors="replace").strip()
    parts = request_line.split()
    if len(parts) != 3:
        raise ValueError("bad request line")
    method, target, _version = parts
    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if line in {b"\r\n", b"\n", b""}:
            break
        decoded = line.decode("ascii", errors="replace")
        name, _, value = decoded.partition(":")
        if name:
            headers[name.strip().lower()] = value.strip()
    return method.upper(), target, headers


async def _read_body(
    reader: asyncio.StreamReader,
    headers: dict[str, str],
    *,
    max_bytes: int,
) -> bytes:
    length = _parse_int(headers.get("content-length", "0"), default=0)
    if length < 0 or length > max_bytes:
        raise ValueError("request body too large")
    if length == 0:
        return b""
    return await reader.readexactly(length)


def _write_json_response(
    writer: asyncio.StreamWriter,
    status: int,
    payload: dict[str, Any],
) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    reason = {
        200: "OK",
        202: "Accepted",
        204: "No Content",
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        500: "Internal Server Error",
        503: "Service Unavailable",
    }.get(status, "OK")
    headers = (
        f"HTTP/1.1 {status} {reason}\r\n"
        "content-type: application/json; charset=utf-8\r\n"
        f"content-length: {len(body)}\r\n"
        "access-control-allow-origin: *\r\n"
        "access-control-allow-methods: GET, POST, OPTIONS\r\n"
        "access-control-allow-headers: authorization, content-type, x-vscode-chat-token\r\n"
        "connection: close\r\n"
        "\r\n"
    ).encode("ascii")
    writer.write(headers + body)


def _normalize_sender(sender: str) -> str:
    normalized = sender.strip().lower()
    return "codex" if normalized == "codex" else "user"


def _project_metadata_from_workspace_context(
    workspace_context: dict[str, Any] | None,
) -> dict[str, str]:
    if not workspace_context:
        return {}
    raw_path = (
        workspace_context.get("workspacePath")
        or workspace_context.get("workspaceFolder")
        or workspace_context.get("rootPath")
    )
    if not isinstance(raw_path, str) or not raw_path.strip():
        return {}
    try:
        project_root = Path(raw_path).expanduser().resolve()
    except OSError:
        return {}
    if not project_root.is_dir():
        return {}
    return {
        "project_root": str(project_root),
        "project_context": (
            f"Текущий VS Code workspace/project root: {project_root}. "
            "Можно использовать project read-only tools для чтения и поиска по "
            "этому проекту; это не даёт права писать файлы, запускать shell, "
            "делать commit, менять env или выполнять внешние действия."
        ),
    }


def _operator_metadata_from_workspace_context(
    workspace_context: dict[str, Any] | None,
) -> dict[str, str]:
    """Return whitelisted operator-only metadata for eval/audit runs."""
    if not workspace_context:
        return {}
    metadata: dict[str, str] = {}
    variant = _workspace_context_text(
        workspace_context,
        "digitalScenarioEvalVariant",
        "digital_scenario_eval_variant",
    )
    if variant in REQUIRED_EVAL_VARIANTS:
        metadata["digital_scenario_eval_variant"] = variant
    scenario_id = _workspace_context_text(
        workspace_context,
        "digitalScenarioId",
        "digital_scenario_id",
    ).removeprefix("digital_scenario.")
    if scenario_id in REQUIRED_DIGITAL_SCENARIO_IDS:
        metadata["digital_scenario_id"] = scenario_id
    run_id = _workspace_context_text(
        workspace_context,
        "digitalScenarioEvalRunId",
        "digital_scenario_eval_run_id",
        limit=120,
    )
    if run_id:
        metadata["digital_scenario_eval_run_id"] = run_id
    message_kind = _workspace_context_text(
        workspace_context,
        "operatorMessageKind",
        "operator_message_kind",
    )
    if message_kind in _OPERATOR_MESSAGE_KINDS:
        metadata["operator_message_kind"] = message_kind
    return metadata


def _workspace_context_text(
    workspace_context: dict[str, Any],
    *keys: str,
    limit: int = 80,
) -> str:
    for key in keys:
        raw = workspace_context.get(key)
        if isinstance(raw, str | int | float):
            value = str(raw).strip()
            if value:
                return value[:limit]
    return ""


def _authorized(headers: dict[str, str], expected_token: str) -> bool:
    header_token = headers.get("x-vscode-chat-token", "").strip()
    auth = headers.get("authorization", "").strip()
    if auth.lower().startswith("bearer "):
        header_token = auth[7:].strip()
    return bool(header_token) and secrets.compare_digest(header_token, expected_token)


def _with_display_timestamp(entry: dict[str, Any]) -> dict[str, Any]:
    raw_ts = entry.get("ts")
    if not isinstance(raw_ts, str) or not raw_ts.strip():
        return entry
    display_ts = _to_display_timestamp(raw_ts)
    if not display_ts or display_ts == raw_ts:
        return entry
    return {**entry, "ts": display_ts, "ts_utc": raw_ts}


def _to_display_timestamp(raw_ts: str) -> str:
    try:
        parsed = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=VSCODE_CHAT_DISPLAY_TZ)
    return parsed.astimezone(VSCODE_CHAT_DISPLAY_TZ).isoformat()


def _origin_allowed(origin: str) -> bool:
    if not origin:
        return True
    if origin.startswith("vscode-webview://"):
        return True
    parsed = urlparse(origin)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def _log_background_task_result(task: asyncio.Task[str]) -> None:
    with contextlib.suppress(asyncio.CancelledError):
        exc = task.exception()
        if exc is not None:
            logger.warning("vscode_chat_background_task_failed", error=str(exc))


def _parse_int(value: object, *, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default
