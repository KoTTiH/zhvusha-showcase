from __future__ import annotations

from pathlib import Path


def test_vscode_chat_context_keeps_eval_metadata_without_workspace() -> None:
    from scripts.vscode_chat import _message_context

    context = _message_context(
        workspace_path=str(Path.cwd()),
        no_workspace_context=True,
        eval_variant="paraphrase",
        scenario_id="ai_cto_projects",
        eval_run_id="run-42",
    )

    assert context == {
        "digitalScenarioEvalVariant": "paraphrase",
        "digitalScenarioId": "ai_cto_projects",
        "digitalScenarioEvalRunId": "run-42",
    }


def test_vscode_chat_eval_matrix_payloads_use_codex_sender_and_variants(
    tmp_path: Path,
) -> None:
    from scripts.vscode_chat import _eval_matrix_payloads

    payloads = _eval_matrix_payloads(
        scenario_id="ai_cto_projects",
        variants=("happy_path", "paraphrase"),
        workspace_path=str(tmp_path),
        no_workspace_context=False,
        eval_run_id="run-42",
    )

    assert len(payloads) == 2
    assert {payload["sender"] for payload in payloads} == {"codex"}
    assert all(not str(payload["text"]).startswith("/") for payload in payloads)
    contexts = [payload["context"] for payload in payloads]
    assert all(isinstance(context, dict) for context in contexts)
    assert {
        str(context["digitalScenarioEvalVariant"])
        for context in contexts
        if isinstance(context, dict)
    } == {"happy_path", "paraphrase"}
    assert {
        str(context["digitalScenarioId"])
        for context in contexts
        if isinstance(context, dict)
    } == {"ai_cto_projects"}
    assert {
        str(context["digitalScenarioEvalRunId"])
        for context in contexts
        if isinstance(context, dict)
    } == {"run-42"}


def test_vscode_chat_wait_detector_requires_assistant_after_matching_user() -> None:
    from scripts.vscode_chat import _has_assistant_reply_after_user

    assert _has_assistant_reply_after_user(
        [
            {"role": "assistant", "text": "старый ответ"},
            {"role": "user", "text": "matrix prompt", "message_id": 17},
            {
                "role": "assistant",
                "text": "новый ответ",
                "reply_to_message_id": 17,
            },
        ],
        "matrix prompt",
    )


def test_vscode_chat_wait_detector_can_scope_to_eval_run_id() -> None:
    from scripts.vscode_chat import _has_assistant_reply_after_user

    messages = [
        {
            "role": "user",
            "text": "matrix prompt",
            "message_id": 11,
            "digital_scenario_eval_run_id": "old-run",
        },
        {
            "role": "assistant",
            "text": "старый ответ",
            "reply_to_message_id": 11,
        },
        {
            "role": "user",
            "text": "matrix prompt",
            "message_id": 12,
            "digital_scenario_eval_run_id": "new-run",
        },
        {
            "role": "assistant",
            "text": "новый ответ",
            "reply_to_message_id": 12,
        },
    ]

    assert _has_assistant_reply_after_user(
        messages,
        "matrix prompt",
        eval_run_id="new-run",
    )
    assert not _has_assistant_reply_after_user(
        messages[:2],
        "matrix prompt",
        eval_run_id="new-run",
    )
    assert not _has_assistant_reply_after_user(
        [
            {"role": "user", "text": "matrix prompt", "message_id": 17},
            {
                "role": "assistant",
                "text": "чужой ответ",
                "reply_to_message_id": 18,
            },
        ],
        "matrix prompt",
    )
    assert _has_assistant_reply_after_user(
        [
            {"role": "user", "text": "matrix prompt", "message_id": 17},
            {
                "role": "assistant",
                "text": "чужой ответ",
                "reply_to_message_id": 18,
            },
            {
                "role": "assistant",
                "text": "нужный ответ",
                "reply_to_message_id": 17,
            },
        ],
        "matrix prompt",
    )
    assert not _has_assistant_reply_after_user(
        [
            {"role": "assistant", "text": "старый ответ"},
            {"role": "user", "text": "matrix prompt"},
        ],
        "matrix prompt",
    )
    assert not _has_assistant_reply_after_user(
        [
            {"role": "user", "text": "другой prompt"},
            {"role": "assistant", "text": "ответ"},
        ],
        "matrix prompt",
    )
