"""Side-effect-free preview of an ImplementSpec cycle (Phase 13).

Two callers:

* ``ImplementSpecSkill.dry_run`` — invoked by the v4 framework when
  ``SELF_CODING_ENABLED=False`` so Никита can sanity-check the planned
  temporary worktree + commit + change surface without flipping the master
  switch.
* CLI smoke (`/spec_run --dry-run <slug>`) — same code path.

Pure function: takes a validated :class:`SpecModel`, returns a
    ``SimulatedResult`` with a markdown plan in ``would_produce``. Does not
    import ``git``, the code-agent backend, or anything that touches the filesystem.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from src.skills.base import SimulatedResult

if TYPE_CHECKING:
    from src.skills.spec_command.parser import SpecModel


def simulate(*, spec: SpecModel) -> SimulatedResult:
    """Return a markdown preview of what the Editor cycle would do."""
    whitelist_block = "\n".join(f"  • `{p}`" for p in spec.whitelist_paths)
    preserve_block = "\n".join(f"  • {p}" for p in spec.preserve_behavior) or (
        "  • Existing behaviour must remain intact unless the spec explicitly "
        "allows simplification."
    )
    simplification_block = (
        "\n".join(f"  • {s}" for s in spec.allowed_simplifications) or "  • none"
    )
    chat_context_block = "\n".join(f"  • {line}" for line in spec.chat_context) or (
        "  • none"
    )
    previous_attempts_block = (
        "\n".join(
            f"  • `{attempt.archive_slug}` · {attempt.status} · tier "
            f"{attempt.tier}: {attempt.insight}"
            for attempt in spec.previous_attempts
        )
        or "  • none"
    )

    update_block = ""
    if spec.existing_tests_to_update:
        update_lines = []
        for entry in spec.existing_tests_to_update:
            update_lines.append(
                f"  • `{entry.path}::{entry.test_name}` — "
                f"{entry.allowed_changes} (reason: {entry.reason})"
            )
        update_block = (
            "  Legitimate existing-test mutations declared by spec:\n"
            + "\n".join(update_lines)
            + "\n"
        )

    plan = (
        f"**Dry-run plan for `{spec.slug}` (Tier {spec.tier})**\n\n"
        f"1. Verify live repo is clean and not on `zhvusha/*`.\n"
        f"2. Create a temporary detached git worktree from current HEAD.\n"
        f"3. Spawn Codex Editor backend inside that worktree with shared "
        f"whitelist rules:\n"
        f"{whitelist_block}\n"
        f"{update_block}"
        f"4. Preserve behaviour contract:\n{preserve_block}\n"
        f"5. Allowed simplifications:\n{simplification_block}\n"
        f"6. /код dialogue context:\n{chat_context_block}\n"
        f"7. Archive previous attempts:\n{previous_attempts_block}\n"
        f"8. Failing test to make green: "
        f"`{spec.failing_test.file}::{spec.failing_test.name}`\n"
        f"9. Run test gate (ruff / mypy / lint-imports / pytest contract+chain).\n"
        f"10. Commit inside the temporary worktree as `zhvusha-coder` with footer "
        f"`Spec: tasks/<date>-{spec.slug}.yaml`.\n"
        f"11. If all gates pass, cherry-pick the commit into the live branch, "
        f"then remove the temporary worktree.\n"
        f"12. Update spec status → `done` (or `failed` on red).\n\n"
        f"_No git/backend/filesystem mutations performed in dry-run._"
    )
    return SimulatedResult(
        would_succeed=True,
        would_produce=plan,
        dependencies_available=True,
        estimated_actual_cost=Decimal("0"),
    )
