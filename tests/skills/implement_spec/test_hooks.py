"""Contract tests for PreToolUse hooks (Phase 13).

Hooks are pure ``(tool_name, tool_input) -> HookDecision`` callables built
from the active spec's ``whitelist_paths``. They define the shared Editor
safe-list: the same logic is unit-tested here and carried into the backend
prompt/commit gates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.contract


PROJECT_ROOT = Path("/workspace/zhvusha-showcase")
WHITELIST = [
    "src/skills/weather/__init__.py",
    "src/skills/weather/skill.py",
    "tests/skills/weather/test_contract.py",
]


def _edit_hook():  # type: ignore[no-untyped-def]
    from src.skills.implement_spec.hooks import make_edit_write_hook

    return make_edit_write_hook(
        whitelist_paths=WHITELIST,
        project_root=PROJECT_ROOT,
    )


def _bash_hook():  # type: ignore[no-untyped-def]
    from src.skills.implement_spec.hooks import make_bash_hook

    return make_bash_hook()


# =====================================================================
# Edit / Write / MultiEdit / NotebookEdit
# =====================================================================


class TestEditWriteWhitelist:
    def test_edit_inside_whitelist_relative(self) -> None:
        decision = _edit_hook()("Edit", {"file_path": "src/skills/weather/skill.py"})
        assert decision.allowed is True

    def test_edit_inside_whitelist_absolute(self) -> None:
        path = str(PROJECT_ROOT / "src/skills/weather/skill.py")
        decision = _edit_hook()("Edit", {"file_path": path})
        assert decision.allowed is True

    def test_edit_outside_whitelist_blocks(self) -> None:
        decision = _edit_hook()("Edit", {"file_path": "src/llm/router.py"})
        assert decision.allowed is False
        assert decision.reason is not None
        assert "whitelist" in decision.reason.lower()

    def test_edit_outside_project_root_blocks(self) -> None:
        decision = _edit_hook()("Edit", {"file_path": "/etc/passwd"})
        assert decision.allowed is False

    def test_write_outside_whitelist_blocks(self) -> None:
        decision = _edit_hook()("Write", {"file_path": "src/llm/secret.py"})
        assert decision.allowed is False

    def test_multi_edit_outside_whitelist_blocks(self) -> None:
        decision = _edit_hook()("MultiEdit", {"file_path": "src/llm/router.py"})
        assert decision.allowed is False

    def test_notebook_edit_outside_whitelist_blocks(self) -> None:
        decision = _edit_hook()(
            "NotebookEdit", {"notebook_path": "experiments/foo.ipynb"}
        )
        assert decision.allowed is False

    def test_unrelated_tool_passes_through(self) -> None:
        # Read / Grep / Glob etc. are NOT this hook's concern.
        decision = _edit_hook()("Read", {"file_path": "/etc/passwd"})
        assert decision.allowed is True

    def test_missing_file_path_blocks(self) -> None:
        decision = _edit_hook()("Edit", {})
        assert decision.allowed is False


# =====================================================================
# Phase 16: existing_tests_to_update legitimate-mutation channel
# =====================================================================


def _edit_hook_with_test_updates(  # type: ignore[no-untyped-def]
    *,
    update_paths: list[str],
):
    from src.skills.implement_spec.hooks import make_edit_write_hook

    return make_edit_write_hook(
        whitelist_paths=WHITELIST,
        project_root=PROJECT_ROOT,
        existing_tests_to_update_paths=update_paths,
    )


class TestEditWriteWithExistingTestsToUpdate:
    """The hook must accept paths from
    ``spec.existing_tests_to_update`` as legitimate edit targets.

    Phase 16: Architect declares specific existing tests that the spec
    legitimately needs to mutate (e.g. extending a finite collection
    behind a fixed-set assertion). These paths are NOT in
    ``whitelist_paths`` (the whitelist is the surgical change surface
    of *new* code), but the hook must allow Edit/Write/MultiEdit on
    them — otherwise even the prompt-level "you may edit listed tests"
    affordance is overruled by the runtime gate.

    The path must still live under the project root, and any path
    that's neither in the whitelist nor in the test-update list must
    still be denied.
    """

    LEGIT_TEST = "tests/research/test_research_service.py"

    def test_listed_test_path_is_allowed_for_edit(self) -> None:
        hook = _edit_hook_with_test_updates(update_paths=[self.LEGIT_TEST])
        decision = hook("Edit", {"file_path": self.LEGIT_TEST})
        assert decision.allowed is True

    def test_listed_test_path_is_allowed_for_write(self) -> None:
        hook = _edit_hook_with_test_updates(update_paths=[self.LEGIT_TEST])
        decision = hook("Write", {"file_path": self.LEGIT_TEST})
        assert decision.allowed is True

    def test_listed_test_path_is_allowed_for_multiedit(self) -> None:
        hook = _edit_hook_with_test_updates(update_paths=[self.LEGIT_TEST])
        decision = hook("MultiEdit", {"file_path": self.LEGIT_TEST})
        assert decision.allowed is True

    def test_listed_test_path_via_absolute_path_is_allowed(self) -> None:
        hook = _edit_hook_with_test_updates(update_paths=[self.LEGIT_TEST])
        absolute = str(PROJECT_ROOT / self.LEGIT_TEST)
        decision = hook("Edit", {"file_path": absolute})
        assert decision.allowed is True

    def test_unlisted_existing_test_remains_blocked(self) -> None:
        # A different existing test, not in either list, must still be denied.
        hook = _edit_hook_with_test_updates(update_paths=[self.LEGIT_TEST])
        decision = hook("Edit", {"file_path": "tests/skills/research/test_other.py"})
        assert decision.allowed is False

    def test_default_empty_list_is_backwards_compatible(self) -> None:
        # No `existing_tests_to_update_paths=` kwarg ⇒ identical to
        # pre-Phase-16 behaviour.
        from src.skills.implement_spec.hooks import make_edit_write_hook

        hook = make_edit_write_hook(
            whitelist_paths=WHITELIST,
            project_root=PROJECT_ROOT,
        )
        # An existing-test path NOT in WHITELIST is denied.
        decision = hook(
            "Edit", {"file_path": "tests/research/test_research_service.py"}
        )
        assert decision.allowed is False

    def test_explicit_empty_list_keeps_old_behaviour(self) -> None:
        hook = _edit_hook_with_test_updates(update_paths=[])
        decision = hook(
            "Edit", {"file_path": "tests/research/test_research_service.py"}
        )
        assert decision.allowed is False


# =====================================================================
# Bash safe-list
# =====================================================================


class TestBashSafeList:
    def test_pytest_allowed(self) -> None:
        assert (
            _bash_hook()("Bash", {"command": "pytest tests/skills/weather"}).allowed
            is True
        )

    def test_uv_run_pytest_allowed(self) -> None:
        assert (
            _bash_hook()("Bash", {"command": "uv run pytest -x --no-cov"}).allowed
            is True
        )

    def test_ruff_allowed(self) -> None:
        assert (
            _bash_hook()("Bash", {"command": "ruff check src/skills/weather/"}).allowed
            is True
        )

    def test_mypy_allowed(self) -> None:
        assert (
            _bash_hook()(
                "Bash", {"command": "uv run mypy src/skills/weather/ --strict"}
            ).allowed
            is True
        )

    def test_lint_imports_allowed(self) -> None:
        assert (
            _bash_hook()(
                "Bash", {"command": "uv run lint-imports --config .importlinter"}
            ).allowed
            is True
        )

    def test_git_status_allowed(self) -> None:
        assert _bash_hook()("Bash", {"command": "git status"}).allowed is True

    def test_git_diff_allowed(self) -> None:
        assert _bash_hook()("Bash", {"command": "git diff src/skills"}).allowed is True

    def test_git_log_allowed(self) -> None:
        assert _bash_hook()("Bash", {"command": "git log -1"}).allowed is True

    def test_python_smoke_import_allowed(self) -> None:
        assert (
            _bash_hook()(
                "Bash", {"command": 'python -c "from src.skills.weather import x"'}
            ).allowed
            is True
        )


class TestBashBlocks:
    def test_git_push_blocked(self) -> None:
        decision = _bash_hook()("Bash", {"command": "git push origin main"})
        assert decision.allowed is False

    def test_git_commit_blocked(self) -> None:
        decision = _bash_hook()("Bash", {"command": "git commit -m 'x'"})
        assert decision.allowed is False

    def test_rm_blocked(self) -> None:
        decision = _bash_hook()("Bash", {"command": "rm -rf src/"})
        assert decision.allowed is False

    def test_curl_blocked(self) -> None:
        decision = _bash_hook()("Bash", {"command": "curl https://example.com"})
        assert decision.allowed is False

    def test_pipe_metacharacter_blocked(self) -> None:
        decision = _bash_hook()("Bash", {"command": "pytest | tee /tmp/log.txt"})
        assert decision.allowed is False

    def test_semicolon_chain_blocked(self) -> None:
        decision = _bash_hook()("Bash", {"command": "pytest; rm -rf src/"})
        assert decision.allowed is False

    def test_subshell_blocked(self) -> None:
        decision = _bash_hook()("Bash", {"command": "pytest $(echo args)"})
        assert decision.allowed is False

    def test_redirect_blocked(self) -> None:
        decision = _bash_hook()("Bash", {"command": "pytest > /tmp/out.txt"})
        assert decision.allowed is False

    def test_chain_with_and_blocked(self) -> None:
        decision = _bash_hook()("Bash", {"command": "pytest && ruff check"})
        assert decision.allowed is False

    def test_python_with_metacharacter_blocked(self) -> None:
        # python -c with a semicolon to chain shell-like statements: the
        # blanket metacharacter ban is the safety guarantee.
        decision = _bash_hook()(
            "Bash", {"command": 'python -c "import shutil; shutil.rmtree(\\"/x\\")"'}
        )
        assert decision.allowed is False

    def test_empty_command_blocked(self) -> None:
        decision = _bash_hook()("Bash", {"command": ""})
        assert decision.allowed is False


class TestBashPassThrough:
    def test_unrelated_tool_passes_through(self) -> None:
        # Hook only opines on Bash.
        decision = _bash_hook()("Read", {"file_path": "/anything"})
        assert decision.allowed is True
