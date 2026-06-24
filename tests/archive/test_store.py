"""Archive SQL store parameter tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from src.archive.models import ArchiveNode, ArchiveStatus
from src.archive.store import _params, _row_to_node


def _archive_node() -> ArchiveNode:
    return ArchiveNode(
        slug="morning-recovery",
        spec_slug="morning-recovery",
        tier=2,
        status=ArchiveStatus.COMMITTED,
        created_at=datetime(2026, 5, 24, 16, 0, tzinfo=UTC),
        diff_summary="fixed recovery window",
        tests_summary="targeted tests passed",
        insight="collectors must receive the consolidation window",
        source_evidence=[{"path": "src/collectors/youtube.py", "reason": "since"}],
        model_config={"backend": "codex_cli"},
        tags=["self-coding", "morning"],
        metadata={"actor": "zhvusha"},
    )


def test_archive_params_encode_jsonb_fields_as_db_safe_strings() -> None:
    params = _params(_archive_node())

    for key in ("source_evidence", "model_config", "tags", "metadata"):
        assert isinstance(params[key], str)

    assert json.loads(params["source_evidence"]) == [
        {"path": "src/collectors/youtube.py", "reason": "since"}
    ]
    assert json.loads(params["model_config"]) == {"backend": "codex_cli"}
    assert json.loads(params["tags"]) == ["self-coding", "morning"]
    assert json.loads(params["metadata"]) == {"actor": "zhvusha"}


def test_archive_row_to_node_decodes_jsonb_strings() -> None:
    params = _params(_archive_node())

    node = _row_to_node(params)

    assert node.source_evidence == [
        {"path": "src/collectors/youtube.py", "reason": "since"}
    ]
    assert node.runtime_config == {"backend": "codex_cli"}
    assert node.tags == ["self-coding", "morning"]
    assert node.metadata == {"actor": "zhvusha"}
