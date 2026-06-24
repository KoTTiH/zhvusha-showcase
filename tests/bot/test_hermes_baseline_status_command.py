"""Bot command surface for Hermes Stage L baseline intake status."""

from __future__ import annotations

from pathlib import Path


def test_hermes_baseline_status_command_is_admin_only(tmp_path: Path) -> None:
    from src.bot.main import _hermes_baseline_status_reply
    from src.skills.base import AgentContext

    reply = _hermes_baseline_status_reply(
        "/hermes_baseline_status",
        AgentContext(user_id=2, chat_id=1, mode="personal"),
        admin_user_id=1,
        project_root=tmp_path,
    )

    assert reply == "Эта команда доступна только Никите."


def test_hermes_baseline_status_command_renders_progress_and_next_scorecard(
    tmp_path: Path,
) -> None:
    from src.bot.main import _hermes_baseline_status_reply
    from src.skills.base import AgentContext
    from src.skills.external_skill_loader.evaluation import (
        FileHermesParityBaselineStore,
        HermesParityBenchmarkGroup,
        build_default_hermes_parity_tasks,
    )

    task = build_default_hermes_parity_tasks()[0]
    store = FileHermesParityBaselineStore(tmp_path / "reports" / "hermes-baselines")
    store.write_manifest((task,))
    store.write_scorecard_template(
        task_id=task.task_id,
        group=HermesParityBenchmarkGroup.DIRECT_CODEX,
    )
    store.append_result(
        task_id=task.task_id,
        group=HermesParityBenchmarkGroup.DIRECT_HERMES,
        solved=True,
        notes="operator-secret-token",
    )

    reply = _hermes_baseline_status_reply(
        "/hermes_baseline_status",
        AgentContext(user_id=1, chat_id=1, mode="personal"),
        admin_user_id=1,
        project_root=tmp_path,
    )

    assert reply is not None
    assert "Hermes Stage L baselines: NOT READY" in reply
    assert "Progress: 1/5 (20%)" in reply
    assert "Matrix: tasks=1, groups=5, scorecards=5" in reply
    assert (
        "Meaning: NOT READY means the Stage L parity claim is not proven yet; "
        "it does not mean the bot failed to start."
    ) in reply
    assert (
        "Next scorecard: "
        "reports/hermes-baselines/scorecards/"
        "codebase-investigation--direct_codex.json"
    ) in reply
    assert "direct_hermes: 1/1" in reply
    assert "direct_codex: 0/1" in reply
    assert "direct_hermes: real Hermes baseline runs" in reply
    assert (
        "zhvusha_with_execution_adapters: ZHVUSHA with approved execution adapters"
        in reply
    )
    assert "operator-secret-token" not in reply
    assert "reports/hermes-baseline-runbook.md" in reply


def test_hermes_baseline_status_command_reports_missing_manifest(
    tmp_path: Path,
) -> None:
    from src.bot.main import _hermes_baseline_status_reply
    from src.skills.base import AgentContext

    reply = _hermes_baseline_status_reply(
        "/hermes_baseline_status",
        AgentContext(user_id=1, chat_id=1, mode="personal"),
        admin_user_id=1,
        project_root=tmp_path,
    )

    assert reply is not None
    assert "Hermes Stage L baselines: manifest не найден" in reply
    assert "reports/hermes-baselines/tasks.json" in reply


def test_hermes_baseline_import_command_is_admin_only(tmp_path: Path) -> None:
    from src.bot.main import _hermes_baseline_import_reply
    from src.skills.base import AgentContext

    reply = _hermes_baseline_import_reply(
        "/hermes_baseline_import scorecards/demo--direct_codex.json",
        AgentContext(user_id=2, chat_id=1, mode="personal"),
        admin_user_id=1,
        project_root=tmp_path,
    )

    assert reply == "Эта команда доступна только Никите."


def test_hermes_baseline_import_command_ingests_scorecard_and_refreshes_reports(
    tmp_path: Path,
) -> None:
    from src.bot.main import _hermes_baseline_import_reply
    from src.skills.base import AgentContext
    from src.skills.external_skill_loader.evaluation import (
        FileHermesParityBaselineStore,
        HermesBaselineScorecard,
        HermesParityBenchmarkGroup,
        build_default_hermes_parity_tasks,
    )

    task = build_default_hermes_parity_tasks()[0]
    store = FileHermesParityBaselineStore(tmp_path / "reports" / "hermes-baselines")
    store.write_manifest((task,))
    scorecard_path = store.write_scorecard_template(
        task_id=task.task_id,
        group=HermesParityBenchmarkGroup.DIRECT_CODEX,
    )
    scorecard_file = tmp_path / "reports" / "hermes-baselines" / scorecard_path
    scorecard = HermesBaselineScorecard.model_validate_json(
        scorecard_file.read_text(encoding="utf-8")
    ).model_copy(
        update={
            "solved": True,
            "operator": "nikita",
            "evidence_quality": 0.8,
            "final_answer_usefulness": 0.85,
            "verification_quality": 0.9,
            "recovery_quality": 0.7,
            "evidence_artifacts": ("reports/codex-run.md",),
            "notes": "secret run notes should stay out of Telegram status",
        }
    )
    scorecard_file.write_text(
        scorecard.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    reply = _hermes_baseline_import_reply(
        f"/hermes_baseline_import reports/hermes-baselines/{scorecard_path}",
        AgentContext(user_id=1, chat_id=1, mode="personal"),
        admin_user_id=1,
        project_root=tmp_path,
    )

    assert reply is not None
    assert "Hermes baseline imported: codebase-investigation / direct_codex" in reply
    assert "Progress: 1/5 (20%)" in reply
    assert "secret run notes" not in reply
    assert len(store.load_results()) == 1
    assert (tmp_path / "reports" / "hermes-baseline-intake.md").exists()
    assert (tmp_path / "reports" / "hermes-baseline-runbook.md").exists()
    assert (tmp_path / "reports" / "hermes-parity.md").exists()
    assert (tmp_path / "reports" / "hermes-completion-audit.md").exists()


def test_hermes_baseline_import_command_rejects_unfilled_template(
    tmp_path: Path,
) -> None:
    from src.bot.main import _hermes_baseline_import_reply
    from src.skills.base import AgentContext
    from src.skills.external_skill_loader.evaluation import (
        FileHermesParityBaselineStore,
        HermesParityBenchmarkGroup,
        build_default_hermes_parity_tasks,
    )

    task = build_default_hermes_parity_tasks()[0]
    store = FileHermesParityBaselineStore(tmp_path / "reports" / "hermes-baselines")
    store.write_manifest((task,))
    scorecard_path = store.write_scorecard_template(
        task_id=task.task_id,
        group=HermesParityBenchmarkGroup.DIRECT_CODEX,
    )

    reply = _hermes_baseline_import_reply(
        f"/hermes_baseline_import reports/hermes-baselines/{scorecard_path}",
        AgentContext(user_id=1, chat_id=1, mode="personal"),
        admin_user_id=1,
        project_root=tmp_path,
    )

    assert reply is not None
    assert "Hermes baseline import rejected" in reply
    assert "operator is required" in reply
    assert "evidence_artifacts is required" in reply
    assert store.load_results() == ()


def test_hermes_baseline_import_command_requires_scorecard_path(
    tmp_path: Path,
) -> None:
    from src.bot.main import _hermes_baseline_import_reply
    from src.skills.base import AgentContext

    reply = _hermes_baseline_import_reply(
        "/hermes_baseline_import",
        AgentContext(user_id=1, chat_id=1, mode="personal"),
        admin_user_id=1,
        project_root=tmp_path,
    )

    assert reply is not None
    assert "Используй: /hermes_baseline_import" in reply
