"""Fixed denylist guard for live ``.env`` keys touched by self-coding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_ENV_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")

PROTECTED_ENV_EXACT_KEYS: tuple[str, ...] = (
    "ADMIN_USER_ID",
    "BOT_TOKEN",
    "DATABASE_URL",
    "REDIS_URL",
    "TELEGRAM_API_ID",
    "TELEGRAM_API_HASH",
    "KWORK_LOGIN",
    "KWORK_PASSWORD",
    "KWORK_PHONE_LAST",
    "CODEX_CLI_PATH",
    "CODE_AGENT_BACKEND",
    "CLAUDE_CLI_PATH",
    "PROJECT_PATH",
    "WORKSPACE_PATH",
    "WORKER_PROVIDER",
    "ANALYST_PROVIDER",
    "STRATEGIST_PROVIDER",
    "VISION_PROVIDER",
    "WORKER_MODEL",
    "ANALYST_MODEL",
    "STRATEGIST_MODEL",
    "VISION_MODEL",
    "DEFAULT_LLM_TIER",
    "COMPARE_PROVIDER",
    "COMPARE_MODEL",
    "COMPARE_MAIN_TIER",
)

PROTECTED_ENV_SUFFIXES: tuple[str, ...] = (
    "_API_KEY",
    "_SECRET",
    "_TOKEN",
    "_PASSWORD",
    "_HASH",
)


@dataclass(frozen=True)
class EnvGuardResult:
    """Outcome of enforcing the protected live ``.env`` key policy."""

    triggered: bool
    changed_keys: tuple[str, ...] = ()
    restored_keys: tuple[str, ...] = ()
    removed_keys: tuple[str, ...] = ()
    env_path: Path | None = None
    message: str = ""


@dataclass(frozen=True)
class LiveEnvActivationResult:
    """Audited result of applying allowed runtime env keys to live `.env`."""

    applied: bool
    changed_keys: tuple[str, ...] = ()
    added_keys: tuple[str, ...] = ()
    audit_path: Path | None = None
    message: str = ""


@dataclass(frozen=True)
class _ProtectedLine:
    key: str
    line: str


@dataclass(frozen=True)
class _RewriteResult:
    lines: tuple[str, ...]
    changed: tuple[str, ...]
    restored: tuple[str, ...]
    removed: tuple[str, ...]


@dataclass(frozen=True)
class _LiveEnvRewrite:
    lines: tuple[str, ...]
    changed: tuple[str, ...]
    added: tuple[str, ...]


class EnvGuard:
    """Restore protected live ``.env`` assignments after a code-agent run.

    The policy is intentionally fixed and denylist-based. The guard stores only
    protected assignment lines that existed when the bot process constructed it;
    at enforcement time it restores those lines and removes newly introduced
    protected assignments.
    """

    instruction_text: str = ""

    def __init__(
        self,
        *,
        protected_lines: tuple[_ProtectedLine, ...],
        had_env_file: bool,
        baseline_env_path: Path | None = None,
    ) -> None:
        self._protected_lines = protected_lines
        self._baseline_by_key = {line.key: line.line for line in protected_lines}
        self._had_env_file = had_env_file
        self._baseline_env_path = baseline_env_path
        self.instruction_text = format_protected_env_prompt()

    @classmethod
    def from_env_file(cls, env_path: Path) -> EnvGuard:
        baseline_env_path = env_path.expanduser().resolve()
        if not env_path.exists():
            return cls(
                protected_lines=(),
                had_env_file=False,
                baseline_env_path=baseline_env_path,
            )
        protected_lines: list[_ProtectedLine] = []
        seen: set[str] = set()
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            key = _extract_env_key(raw_line)
            if key is None or key in seen or not is_protected_env_key(key):
                continue
            protected_lines.append(_ProtectedLine(key=key, line=raw_line))
            seen.add(key)
        return cls(
            protected_lines=tuple(protected_lines),
            had_env_file=True,
            baseline_env_path=baseline_env_path,
        )

    def enforce(self, project_root: Path) -> EnvGuardResult:
        env_path = project_root / ".env"
        is_baseline_env = (
            self._baseline_env_path is not None
            and env_path.expanduser().resolve() == self._baseline_env_path
        )
        if not env_path.exists() and not self._baseline_by_key:
            return EnvGuardResult(triggered=False, env_path=env_path)
        if not env_path.exists() and not is_baseline_env:
            return EnvGuardResult(triggered=False, env_path=env_path)

        current_lines = (
            env_path.read_text(encoding="utf-8").splitlines()
            if env_path.exists()
            else []
        )
        rewrite = self._rewrite_lines(
            current_lines,
            restore_baseline=is_baseline_env,
        )
        triggered = bool(rewrite.changed or rewrite.restored or rewrite.removed)
        if not triggered:
            return EnvGuardResult(triggered=False, env_path=env_path)

        self._write_repaired_env(env_path, rewrite.lines)
        all_keys = tuple(
            sorted(set(rewrite.changed) | set(rewrite.restored) | set(rewrite.removed))
        )
        message = _format_guard_message(
            all_keys,
            rewrite.changed,
            rewrite.restored,
            rewrite.removed,
        )
        return EnvGuardResult(
            triggered=True,
            changed_keys=rewrite.changed,
            restored_keys=rewrite.restored,
            removed_keys=rewrite.removed,
            env_path=env_path,
            message=message,
        )

    def _rewrite_lines(
        self,
        current_lines: list[str],
        *,
        restore_baseline: bool,
    ) -> _RewriteResult:
        output_lines: list[str] = []
        seen_output_keys: set[str] = set()
        changed: set[str] = set()
        restored: set[str] = set()
        removed: set[str] = set()

        for raw_line in current_lines:
            key = _extract_env_key(raw_line)
            if key is None or not is_protected_env_key(key):
                output_lines.append(raw_line)
                continue

            baseline_line = self._baseline_by_key.get(key)
            if baseline_line is None or not restore_baseline:
                removed.add(key)
                continue
            if key not in seen_output_keys:
                output_lines.append(baseline_line)
                seen_output_keys.add(key)
            if raw_line != baseline_line:
                changed.add(key)

        if restore_baseline:
            for protected_line in self._protected_lines:
                if protected_line.key in seen_output_keys:
                    continue
                output_lines.append(protected_line.line)
                restored.add(protected_line.key)

        return _RewriteResult(
            lines=tuple(output_lines),
            changed=tuple(sorted(changed)),
            restored=tuple(sorted(restored)),
            removed=tuple(sorted(removed)),
        )

    def _write_repaired_env(
        self, env_path: Path, output_lines: tuple[str, ...]
    ) -> None:
        if output_lines or self._had_env_file:
            env_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
        elif env_path.exists():
            env_path.unlink()


class LiveEnvActivator:
    """Apply explicitly allowed non-protected `.env` keys to the live host file."""

    def __init__(
        self,
        *,
        live_env_path: Path,
        audit_root: Path | None = None,
    ) -> None:
        self._live_env_path = live_env_path.expanduser().resolve()
        self._audit_root = (
            audit_root.expanduser().resolve()
            if audit_root is not None
            else (self._live_env_path.parent / ".agent_runtime" / "host_ops").resolve()
        )

    def apply_from_workspace(
        self,
        *,
        workspace_root: Path,
        allowed_keys: tuple[str, ...],
        spec_slug: str,
    ) -> LiveEnvActivationResult:
        """Copy allowed non-protected env assignments from worktree to live `.env`.

        Values are never written to the audit artifact. Protected keys remain a
        hard block even if a malformed spec tries to list them as allowed.
        """
        normalized_keys = tuple(sorted({key.strip().upper() for key in allowed_keys}))
        if not normalized_keys:
            return LiveEnvActivationResult(
                applied=False,
                message="live env activation skipped: no allowed keys declared",
            )
        protected = tuple(key for key in normalized_keys if is_protected_env_key(key))
        if protected:
            raise ValueError(
                "protected live env keys cannot be activated: " + ", ".join(protected)
            )

        workspace_env_path = workspace_root / ".env"
        if not workspace_env_path.exists():
            return LiveEnvActivationResult(
                applied=False,
                message="live env activation skipped: workspace .env not found",
            )

        source_by_key = _assignment_lines_by_key(
            workspace_env_path.read_text(encoding="utf-8").splitlines()
        )
        live_lines = (
            self._live_env_path.read_text(encoding="utf-8").splitlines()
            if self._live_env_path.exists()
            else []
        )

        rewrite = _rewrite_live_env_lines(
            live_lines=live_lines,
            source_by_key=source_by_key,
            allowed_keys=normalized_keys,
        )

        if not rewrite.changed and not rewrite.added:
            return LiveEnvActivationResult(
                applied=False,
                message="live env activation skipped: no declared key changed",
            )

        self._live_env_path.write_text(
            "\n".join(rewrite.lines) + "\n",
            encoding="utf-8",
        )
        audit_path = self._write_audit(
            spec_slug=spec_slug,
            changed_keys=rewrite.changed,
            added_keys=rewrite.added,
        )
        keys = tuple(sorted(set(rewrite.changed) | set(rewrite.added)))
        return LiveEnvActivationResult(
            applied=True,
            changed_keys=rewrite.changed,
            added_keys=rewrite.added,
            audit_path=audit_path,
            message=(
                "Live `.env` activation applied for: "
                + ", ".join(keys)
                + ". Values are redacted in audit."
            ),
        )

    def _write_audit(
        self,
        *,
        spec_slug: str,
        changed_keys: tuple[str, ...],
        added_keys: tuple[str, ...],
    ) -> Path:
        self._audit_root.mkdir(parents=True, exist_ok=True)
        audit_path = self._audit_root / f"env-activation-{spec_slug}.md"
        lines = [
            "# Live .env Activation",
            "",
            f"spec_slug: {spec_slug}",
            "values: redacted",
            f"changed_keys: {', '.join(changed_keys) or '-'}",
            f"added_keys: {', '.join(added_keys) or '-'}",
        ]
        audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return audit_path


def is_protected_env_key(key: str) -> bool:
    normalized = key.strip().upper()
    return normalized in PROTECTED_ENV_EXACT_KEYS or any(
        normalized.endswith(suffix) for suffix in PROTECTED_ENV_SUFFIXES
    )


def format_protected_env_prompt() -> str:
    exact = ", ".join(f"`{key}`" for key in PROTECTED_ENV_EXACT_KEYS)
    suffixes = ", ".join(f"`*{suffix}`" for suffix in PROTECTED_ENV_SUFFIXES)
    return (
        "Protected .env denylist for self-coding:\n"
        f"- Forbidden exact keys: {exact}.\n"
        f"- Forbidden key patterns: {suffixes}.\n"
        "- Do not edit, add, remove, rename, or activate these live `.env` "
        "keys. If the spec appears to require one of them, STOP and report the "
        "blocker to Никита. Repo-side safe defaults and `.env.example` changes "
        "are allowed only when they stay inside the approved whitelist."
    )


def _extract_env_key(line: str) -> str | None:
    match = _ENV_ASSIGN_RE.match(line)
    if match is None:
        return None
    return match.group(1).upper()


def _assignment_lines_by_key(lines: list[str]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for line in lines:
        key = _extract_env_key(line)
        if key is None:
            continue
        assignments[key] = line
    return assignments


def _rewrite_live_env_lines(
    *,
    live_lines: list[str],
    source_by_key: dict[str, str],
    allowed_keys: tuple[str, ...],
) -> _LiveEnvRewrite:
    output_lines: list[str] = []
    seen_live_keys: set[str] = set()
    changed: set[str] = set()
    added: set[str] = set()

    for raw_line in live_lines:
        key = _extract_env_key(raw_line)
        if key is None or key not in allowed_keys:
            output_lines.append(raw_line)
            continue
        seen_live_keys.add(key)
        replacement = source_by_key.get(key)
        if replacement is None:
            output_lines.append(raw_line)
            continue
        output_lines.append(replacement)
        if replacement != raw_line:
            changed.add(key)

    for key in allowed_keys:
        if key in seen_live_keys or key not in source_by_key:
            continue
        output_lines.append(source_by_key[key])
        added.add(key)

    return _LiveEnvRewrite(
        lines=tuple(output_lines),
        changed=tuple(sorted(changed)),
        added=tuple(sorted(added)),
    )


def _format_guard_message(
    keys: tuple[str, ...],
    changed_keys: tuple[str, ...],
    restored_keys: tuple[str, ...],
    removed_keys: tuple[str, ...],
) -> str:
    key_list = ", ".join(keys) or "protected .env key"
    parts = [
        f"Сработала защита live `.env`: нельзя трогать {key_list}.",
        "Я автоматически вернула запрещённые значения обратно и остановила цикл.",
    ]
    if changed_keys:
        parts.append("Изменённые ключи: " + ", ".join(changed_keys) + ".")
    if restored_keys:
        parts.append("Восстановленные ключи: " + ", ".join(restored_keys) + ".")
    if removed_keys:
        parts.append(
            "Удалённые новые запрещённые ключи: " + ", ".join(removed_keys) + "."
        )
    parts.append(
        "В этой self-coding сессии такие ключи запрещены; нужно убрать это "
        "из задачи или отдельно попросить Никиту о host-ops решении."
    )
    return " ".join(parts)
