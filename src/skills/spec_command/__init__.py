"""spec_command skill — InlineSkill for /spec list/show/approve/reject.

The package exposes the Pydantic schema (`SpecModel`, `SpecStatus`,
`FailingTest`, `ResearchFinding`, `PreviousAttempt`) used by every link in
the spec-first chain:
ideation_to_spec writes through it, implement_spec validates through it before
execution, and `scripts/check_whitelist.sh` parses YAML by the same field
names.

Phase 10 ships only the schema. The skill itself (Phase 11) and Architect /
Editor delegated skills (Phase 12-13) layer on top.
"""

from src.skills.spec_command.parser import (
    FailingTest,
    PreviousAttempt,
    ResearchFinding,
    SpecModel,
    SpecStatus,
)

__all__ = [
    "FailingTest",
    "PreviousAttempt",
    "ResearchFinding",
    "SpecModel",
    "SpecStatus",
]
