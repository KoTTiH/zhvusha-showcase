"""Contract tests for protected live ``.env`` keys in self-coding."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.contract


class TestProtectedEnvKeys:
    def test_exact_and_pattern_denylist(self) -> None:
        from src.skills.implement_spec.env_guard import is_protected_env_key

        assert is_protected_env_key("BOT_TOKEN")
        assert is_protected_env_key("DATABASE_URL")
        assert is_protected_env_key("OPENROUTER_API_KEY")
        assert is_protected_env_key("CUSTOM_SECRET")
        assert is_protected_env_key("SOME_PASSWORD")
        assert is_protected_env_key("TELEGRAM_API_HASH")

        assert not is_protected_env_key("BOT_RESTART_ENABLED")
        assert not is_protected_env_key("SELF_CODING_ENABLED")

    def test_prompt_lists_forbidden_keys_without_values(self) -> None:
        from src.skills.implement_spec.env_guard import format_protected_env_prompt

        prompt = format_protected_env_prompt()

        assert "BOT_TOKEN" in prompt
        assert "DATABASE_URL" in prompt
        assert "*_API_KEY" in prompt
        assert "*_PASSWORD" in prompt
        assert "BOT_RESTART_ENABLED" not in prompt
        assert "sk-" not in prompt


class TestEnvGuard:
    def test_restores_changed_protected_key_and_keeps_allowed_key(
        self, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.env_guard import EnvGuard

        env_path = tmp_path / ".env"
        env_path.write_text(
            "BOT_TOKEN=fake_original\nBOT_RESTART_ENABLED=false\n",
            encoding="utf-8",
        )
        guard = EnvGuard.from_env_file(env_path)

        env_path.write_text(
            "BOT_TOKEN=fake_changed\nBOT_RESTART_ENABLED=true\n",
            encoding="utf-8",
        )
        result = guard.enforce(tmp_path)

        assert result.triggered
        assert result.changed_keys == ("BOT_TOKEN",)
        assert env_path.read_text(encoding="utf-8") == (
            "BOT_TOKEN=fake_original\nBOT_RESTART_ENABLED=true\n"
        )
        assert "BOT_TOKEN" in result.message
        assert "вернула" in result.message.lower()

    def test_removes_new_protected_key_without_baseline(self, tmp_path: Path) -> None:
        from src.skills.implement_spec.env_guard import EnvGuard

        env_path = tmp_path / ".env"
        env_path.write_text("BOT_RESTART_ENABLED=false\n", encoding="utf-8")
        guard = EnvGuard.from_env_file(env_path)

        env_path.write_text(
            "BOT_RESTART_ENABLED=true\nANTHROPIC_API_KEY=fake_secret_new\n",
            encoding="utf-8",
        )
        result = guard.enforce(tmp_path)

        assert result.triggered
        assert result.removed_keys == ("ANTHROPIC_API_KEY",)
        assert env_path.read_text(encoding="utf-8") == "BOT_RESTART_ENABLED=true\n"

    def test_readds_deleted_protected_key(self, tmp_path: Path) -> None:
        from src.skills.implement_spec.env_guard import EnvGuard

        env_path = tmp_path / ".env"
        env_path.write_text(
            "DATABASE_URL=postgres://local\nBOT_RESTART_ENABLED=false\n",
            encoding="utf-8",
        )
        guard = EnvGuard.from_env_file(env_path)

        env_path.write_text("BOT_RESTART_ENABLED=true\n", encoding="utf-8")
        result = guard.enforce(tmp_path)

        assert result.triggered
        assert result.restored_keys == ("DATABASE_URL",)
        assert env_path.read_text(encoding="utf-8") == (
            "BOT_RESTART_ENABLED=true\nDATABASE_URL=postgres://local\n"
        )

    def test_preserves_new_non_protected_live_keys_when_repairing(
        self, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.env_guard import EnvGuard

        env_path = tmp_path / ".env"
        env_path.write_text(
            "BOT_TOKEN=fake_original\nBOT_RESTART_ENABLED=false\n",
            encoding="utf-8",
        )
        guard = EnvGuard.from_env_file(env_path)

        env_path.write_text(
            "\n".join(
                [
                    "# live local settings",
                    "BOT_TOKEN=fake_changed",
                    "BOT_RESTART_ENABLED=true",
                    "CHANNEL_VISUALS_ENABLED=true",
                    "IMAGE_GENERATION_PROVIDER=openai",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        result = guard.enforce(tmp_path)

        assert result.triggered
        assert result.changed_keys == ("BOT_TOKEN",)
        assert env_path.read_text(encoding="utf-8") == (
            "# live local settings\n"
            "BOT_TOKEN=fake_original\n"
            "BOT_RESTART_ENABLED=true\n"
            "CHANNEL_VISUALS_ENABLED=true\n"
            "IMAGE_GENERATION_PROVIDER=openai\n"
        )

    def test_deletes_env_file_when_only_new_protected_keys_were_created(
        self, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.env_guard import EnvGuard

        env_path = tmp_path / ".env"
        guard = EnvGuard.from_env_file(env_path)

        env_path.write_text("OPENROUTER_API_KEY=fake_secret_new\n", encoding="utf-8")
        result = guard.enforce(tmp_path)

        assert result.triggered
        assert result.removed_keys == ("OPENROUTER_API_KEY",)
        assert not env_path.exists()

    def test_ignores_missing_env_file_in_isolated_workspace(
        self, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.env_guard import EnvGuard

        live = tmp_path / "live"
        workspace = tmp_path / "workspace"
        live.mkdir()
        workspace.mkdir()
        (live / ".env").write_text(
            "BOT_TOKEN=fake_original\nBOT_RESTART_ENABLED=false\n",
            encoding="utf-8",
        )
        guard = EnvGuard.from_env_file(live / ".env")

        result = guard.enforce(workspace)

        assert not result.triggered
        assert not (workspace / ".env").exists()

    def test_keeps_allowed_workspace_env_without_restoring_live_secrets(
        self, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.env_guard import EnvGuard

        live = tmp_path / "live"
        workspace = tmp_path / "workspace"
        live.mkdir()
        workspace.mkdir()
        (live / ".env").write_text(
            "BOT_TOKEN=fake_original\nBOT_RESTART_ENABLED=false\n",
            encoding="utf-8",
        )
        (workspace / ".env").write_text(
            "BOT_RESTART_ENABLED=true\n",
            encoding="utf-8",
        )
        guard = EnvGuard.from_env_file(live / ".env")

        result = guard.enforce(workspace)

        assert not result.triggered
        assert (workspace / ".env").read_text(encoding="utf-8") == (
            "BOT_RESTART_ENABLED=true\n"
        )

    def test_preserves_new_non_protected_workspace_keys_when_blocking(
        self, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.env_guard import EnvGuard

        live = tmp_path / "live"
        workspace = tmp_path / "workspace"
        live.mkdir()
        workspace.mkdir()
        (live / ".env").write_text("BOT_TOKEN=fake_original\n", encoding="utf-8")
        (workspace / ".env").write_text(
            "\n".join(
                [
                    "# safe repo-side defaults",
                    "BOT_TOKEN=fake_changed",
                    "CHANNEL_VISUALS_ENABLED=true",
                    "IMAGE_GENERATION_PROVIDER=openai",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        guard = EnvGuard.from_env_file(live / ".env")

        result = guard.enforce(workspace)

        assert result.triggered
        assert result.removed_keys == ("BOT_TOKEN",)
        assert (workspace / ".env").read_text(encoding="utf-8") == (
            "# safe repo-side defaults\n"
            "CHANNEL_VISUALS_ENABLED=true\n"
            "IMAGE_GENERATION_PROVIDER=openai\n"
        )

    def test_removes_protected_workspace_key_without_leaking_live_secret(
        self, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.env_guard import EnvGuard

        live = tmp_path / "live"
        workspace = tmp_path / "workspace"
        live.mkdir()
        workspace.mkdir()
        (live / ".env").write_text("BOT_TOKEN=fake_original\n", encoding="utf-8")
        (workspace / ".env").write_text(
            "BOT_TOKEN=fake_changed\nBOT_RESTART_ENABLED=true\n",
            encoding="utf-8",
        )
        guard = EnvGuard.from_env_file(live / ".env")

        result = guard.enforce(workspace)

        assert result.triggered
        assert result.removed_keys == ("BOT_TOKEN",)
        assert (workspace / ".env").read_text(encoding="utf-8") == (
            "BOT_RESTART_ENABLED=true\n"
        )


class TestLiveEnvActivation:
    def test_applies_allowed_non_protected_key_and_writes_audit(
        self, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.env_guard import LiveEnvActivator

        live = tmp_path / "live"
        workspace = tmp_path / "workspace"
        audit = tmp_path / "audit"
        live.mkdir()
        workspace.mkdir()
        (live / ".env").write_text("BOT_RESTART_ENABLED=false\n", encoding="utf-8")
        (workspace / ".env").write_text(
            "BOT_RESTART_ENABLED=true\n",
            encoding="utf-8",
        )

        result = LiveEnvActivator(
            live_env_path=live / ".env",
            audit_root=audit,
        ).apply_from_workspace(
            workspace_root=workspace,
            allowed_keys=("BOT_RESTART_ENABLED",),
            spec_slug="safe-restart",
        )

        assert result.applied
        assert result.changed_keys == ("BOT_RESTART_ENABLED",)
        assert (live / ".env").read_text(encoding="utf-8") == (
            "BOT_RESTART_ENABLED=true\n"
        )
        assert result.audit_path is not None
        audit_text = result.audit_path.read_text(encoding="utf-8")
        assert "BOT_RESTART_ENABLED" in audit_text
        assert "true" not in audit_text

    def test_refuses_protected_key_even_when_allowed(self, tmp_path: Path) -> None:
        from src.skills.implement_spec.env_guard import LiveEnvActivator

        live = tmp_path / "live"
        workspace = tmp_path / "workspace"
        live.mkdir()
        workspace.mkdir()
        (live / ".env").write_text("BOT_TOKEN=old\n", encoding="utf-8")
        (workspace / ".env").write_text("BOT_TOKEN=new\n", encoding="utf-8")

        with pytest.raises(ValueError, match="protected"):
            LiveEnvActivator(live_env_path=live / ".env").apply_from_workspace(
                workspace_root=workspace,
                allowed_keys=("BOT_TOKEN",),
                spec_slug="bad-env",
            )
