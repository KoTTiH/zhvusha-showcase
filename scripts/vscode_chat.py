#!/usr/bin/env python3
"""Small CLI for Codex/operator access to the local VS Code Жвуша chat."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from src.agent_runtime.digital_scenarios import (
    BUILTIN_DIGITAL_SCENARIOS,
    REQUIRED_EVAL_VARIANTS,
    DigitalScenarioDefinition,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read/write Жвуша VS Code chat.")
    parser.add_argument("--api-url", default="http://127.0.0.1:7331")
    parser.add_argument("--token", default=_default_token())
    subparsers = parser.add_subparsers(dest="command", required=True)

    send_parser = subparsers.add_parser("send", help="send a Codex-marked message")
    send_parser.add_argument("text", nargs="+")
    send_parser.add_argument("--sender", default="codex", choices=("codex", "user"))
    send_parser.add_argument(
        "--workspace-path",
        default=str(Path.cwd()),
        help="workspace/project root passed to Жвуша read-only project tools",
    )
    send_parser.add_argument(
        "--no-workspace-context",
        action="store_true",
        help="send without workspace/project root context",
    )
    send_parser.add_argument(
        "--eval-variant",
        choices=REQUIRED_EVAL_VARIANTS,
        help="mark this operator run as a digital scenario matrix variant",
    )
    send_parser.add_argument(
        "--eval-run-id",
        default="",
        help="stable id for grouping digital scenario matrix runs",
    )
    send_parser.add_argument(
        "--allow-user-impersonation",
        action="store_true",
        help="allow sender=user for explicit user-path smoke tests",
    )

    eval_parser = subparsers.add_parser(
        "eval-matrix",
        help="queue digital scenario matrix prompts as Codex/operator messages",
    )
    eval_parser.add_argument(
        "scenario_id",
        help="scenario id to queue, or 'all'",
    )
    eval_parser.add_argument(
        "--variant",
        choices=REQUIRED_EVAL_VARIANTS,
        action="append",
        help="queue only selected variant; may be repeated",
    )
    eval_parser.add_argument(
        "--workspace-path",
        default=str(Path.cwd()),
        help="workspace/project root passed to Жвуша read-only project tools",
    )
    eval_parser.add_argument(
        "--no-workspace-context",
        action="store_true",
        help="send without workspace/project root context",
    )
    eval_parser.add_argument(
        "--eval-run-id",
        default="",
        help="stable id for grouping this matrix run",
    )
    eval_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print payloads without sending them to the bridge",
    )
    eval_parser.add_argument(
        "--parallel",
        action="store_true",
        help="queue all prompts immediately; default sends sequentially",
    )
    eval_parser.add_argument(
        "--reply-timeout",
        type=float,
        default=180.0,
        help="seconds to wait for each Жвуша reply in sequential mode",
    )
    eval_parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="history polling interval in sequential mode",
    )

    tail_parser = subparsers.add_parser("tail", help="print recent messages")
    tail_parser.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()
    if args.command == "send":
        if args.sender == "user" and not args.allow_user_impersonation:
            raise SystemExit(
                "Refusing to send as user from Codex CLI. "
                "Use --allow-user-impersonation only for explicit user-path smoke tests."
            )
        text = " ".join(args.text)
        context = _message_context(
            workspace_path=args.workspace_path,
            no_workspace_context=args.no_workspace_context,
            eval_variant=args.eval_variant or "",
            eval_run_id=args.eval_run_id,
        )
        payload = _request_json(
            args.api_url,
            "POST",
            "/message",
            {
                "sender": args.sender,
                "text": text,
                **({"context": context} if context is not None else {}),
            },
            token=args.token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "eval-matrix":
        run_id = args.eval_run_id.strip() or f"digital-scenario-{uuid4().hex[:12]}"
        payloads = _eval_matrix_payloads(
            scenario_id=args.scenario_id,
            variants=tuple(args.variant or REQUIRED_EVAL_VARIANTS),
            workspace_path=args.workspace_path,
            no_workspace_context=args.no_workspace_context,
            eval_run_id=run_id,
        )
        if args.dry_run:
            print(
                json.dumps(
                    {"evalRunId": run_id, "payloads": payloads},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        observed_replies = 0
        for payload in payloads:
            _request_json(
                args.api_url,
                "POST",
                "/message",
                payload,
                token=args.token,
            )
            if not args.parallel:
                text = str(payload.get("text", ""))
                if _wait_for_assistant_reply(
                    args.api_url,
                    token=args.token,
                    text=text,
                    eval_run_id=run_id,
                    timeout_seconds=max(1.0, args.reply_timeout),
                    poll_seconds=max(0.2, args.poll_seconds),
                ):
                    observed_replies += 1
                    continue
                raise SystemExit(
                    "Timed out waiting for Жвуша reply after matrix prompt: "
                    f"{text[:120]}"
                )
        print(
            json.dumps(
                {
                    "evalRunId": run_id,
                    "queued": len(payloads),
                    "replyObserved": observed_replies,
                    "mode": "parallel" if args.parallel else "sequential",
                    "scenarioIds": _scenario_ids_from_payloads(payloads),
                    "variants": list(
                        dict.fromkeys(args.variant or REQUIRED_EVAL_VARIANTS)
                    ),
                    "sender": "codex",
                    "impersonation": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "tail":
        payload = _request_json(
            args.api_url,
            "GET",
            f"/messages?limit={max(1, args.limit)}",
            None,
            token=args.token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    parser.error("unknown command")
    return 2


def _message_context(
    *,
    workspace_path: str,
    no_workspace_context: bool,
    eval_variant: str = "",
    scenario_id: str = "",
    eval_run_id: str = "",
) -> dict[str, object] | None:
    context: dict[str, object] = {}
    if not no_workspace_context:
        context["workspacePath"] = str(Path(workspace_path).resolve())
    if eval_variant:
        context["digitalScenarioEvalVariant"] = eval_variant
    if scenario_id:
        context["digitalScenarioId"] = scenario_id
    if eval_run_id:
        context["digitalScenarioEvalRunId"] = eval_run_id
    return context or None


def _eval_matrix_payloads(
    *,
    scenario_id: str,
    variants: tuple[str, ...],
    workspace_path: str,
    no_workspace_context: bool,
    eval_run_id: str,
) -> list[dict[str, object]]:
    selected = _select_scenarios(scenario_id)
    if not selected:
        known = ", ".join(scenario.id for scenario in BUILTIN_DIGITAL_SCENARIOS)
        raise SystemExit(f"Unknown scenario_id '{scenario_id}'. Known: all, {known}")
    allowed_variants = set(REQUIRED_EVAL_VARIANTS)
    payloads: list[dict[str, object]] = []
    for scenario in selected:
        for case in scenario.eval_cases:
            if case.variant not in variants:
                continue
            if case.variant not in allowed_variants:
                continue
            context = _message_context(
                workspace_path=workspace_path,
                no_workspace_context=no_workspace_context,
                eval_variant=case.variant,
                scenario_id=scenario.id,
                eval_run_id=eval_run_id,
            )
            payload: dict[str, object] = {
                "sender": "codex",
                "text": case.prompt,
            }
            if context is not None:
                payload["context"] = context
            payloads.append(payload)
    return payloads


def _select_scenarios(scenario_id: str) -> tuple[DigitalScenarioDefinition, ...]:
    normalized = scenario_id.strip().removeprefix("digital_scenario.")
    if normalized == "all":
        return tuple(BUILTIN_DIGITAL_SCENARIOS)
    return tuple(
        scenario for scenario in BUILTIN_DIGITAL_SCENARIOS if scenario.id == normalized
    )


def _scenario_ids_from_payloads(payloads: list[dict[str, object]]) -> list[str]:
    ids: set[str] = set()
    for payload in payloads:
        context = payload.get("context")
        if isinstance(context, dict):
            scenario_id = str(context.get("digitalScenarioId", "") or "").strip()
            if scenario_id:
                ids.add(scenario_id)
    return sorted(ids)


def _wait_for_assistant_reply(
    base_url: str,
    *,
    token: str,
    text: str,
    eval_run_id: str = "",
    timeout_seconds: float,
    poll_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        payload = _request_json(
            base_url,
            "GET",
            "/messages?limit=200",
            None,
            token=token,
        )
        messages = payload.get("messages")
        if isinstance(messages, list) and _has_assistant_reply_after_user(
            messages,
            text,
            eval_run_id=eval_run_id,
        ):
            return True
        time.sleep(poll_seconds)
    return False


def _has_assistant_reply_after_user(
    messages: list[object],
    text: str,
    *,
    eval_run_id: str = "",
) -> bool:
    seen_user = False
    target_message_id = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "") or "")
        message_text = str(message.get("text", "") or "")
        if role == "user" and message_text == text:
            if eval_run_id and not _message_matches_eval_run_id(message, eval_run_id):
                continue
            seen_user = True
            target_message_id = str(message.get("message_id", "") or "")
            continue
        if not seen_user or role != "assistant":
            continue
        reply_to_message_id = str(message.get("reply_to_message_id", "") or "")
        if target_message_id:
            if reply_to_message_id == target_message_id:
                return True
            continue
        return True
    return False


def _message_matches_eval_run_id(message: dict[str, object], eval_run_id: str) -> bool:
    expected = eval_run_id.strip()
    if not expected:
        return True
    for key in (
        "digital_scenario_eval_run_id",
        "digitalScenarioEvalRunId",
        "eval_run_id",
    ):
        value = str(message.get(key, "") or "").strip()
        if value == expected:
            return True
    return False


def _request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, object] | None,
    *,
    token: str = "",
) -> dict[str, object]:
    target = urlparse(base_url.rstrip("/"))
    if target.scheme not in {"http", "https"}:
        raise SystemExit(f"Unsupported scheme: {target.scheme}")
    if target.scheme == "https":
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            target.netloc,
            timeout=10,
        )
    else:
        connection = http.client.HTTPConnection(target.netloc, timeout=10)
    body = (
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if payload is not None
        else None
    )
    headers = {"content-type": "application/json"} if body is not None else {}
    if token:
        headers["authorization"] = f"Bearer {token}"
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read().decode("utf-8")
    connection.close()
    if response.status < 200 or response.status >= 300:
        raise SystemExit(f"HTTP {response.status}: {raw}")
    parsed = json.loads(raw or "{}")
    if not isinstance(parsed, dict):
        raise SystemExit("Unexpected JSON response")
    return parsed


def _default_token() -> str:
    token = os.environ.get("VSCODE_CHAT_TOKEN", "").strip()
    if token:
        return token
    env_path = Path(".env")
    if not env_path.is_file():
        return ""
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        key, sep, value = line.partition("=")
        if sep and key.strip() == "VSCODE_CHAT_TOKEN":
            return value.strip().strip('"').strip("'")
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
