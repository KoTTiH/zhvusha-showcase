"""Contract tests for the migrated channel_writer v4 skill.

These tests verify the public contract of ``ChannelWriterSkill`` against the
v4 ``InlineSkill`` base class and its sibling ``skill.yaml`` manifest:

* inheritance (``InlineSkill`` → ``BaseSkill``);
* ``skill_type`` matches the manifest;
* the manifest loads and all class attributes align with YAML declarations;
* declared ``side_effects`` cover the observable behaviour (channel post,
  Telegram send, workspace write);
* ``approval_policy`` is ``required`` — publishing to the channel must
  always go through approval.

Run: ``pytest tests/skills/channel_writer/test_contract.py``
"""

from __future__ import annotations

import pytest
from src.skills.base import BaseSkill, InlineSkill, SideEffect
from src.skills.channel_writer.skill import ChannelWriterSkill
from src.skills.manifest import (
    load_manifest_for_skill_class,
    validate_manifest_matches_class,
)

pytestmark = pytest.mark.contract


class TestContract:
    def test_inherits_inline_skill(self) -> None:
        assert issubclass(ChannelWriterSkill, InlineSkill)
        assert issubclass(ChannelWriterSkill, BaseSkill)

    def test_skill_type_is_inline(self) -> None:
        assert ChannelWriterSkill.skill_type == "inline"

    def test_manifest_loads(self) -> None:
        manifest = load_manifest_for_skill_class(ChannelWriterSkill)
        assert manifest.name == "channel_writer"
        assert manifest.type == "inline"
        assert manifest.llm_tier == "strategist"
        assert manifest.version == "1.0.0"

    def test_manifest_matches_class(self) -> None:
        manifest = load_manifest_for_skill_class(ChannelWriterSkill)
        # Should not raise — attributes align with YAML.
        validate_manifest_matches_class(manifest, ChannelWriterSkill)

    def test_required_side_effects_declared(self) -> None:
        assert SideEffect.POSTS_TO_CHANNEL in ChannelWriterSkill.side_effects
        assert SideEffect.SENDS_TELEGRAM_MESSAGE in ChannelWriterSkill.side_effects
        assert SideEffect.WRITES_WORKSPACE in ChannelWriterSkill.side_effects

    def test_approval_policy_is_required(self) -> None:
        assert ChannelWriterSkill.approval_policy == "required"

    def test_modifies_no_core_capabilities(self) -> None:
        assert ChannelWriterSkill.modifies == []

    def test_triggers_contain_post_prefix(self) -> None:
        assert any(
            trigger.strip() == "/post" or trigger.startswith("/post")
            for trigger in ChannelWriterSkill.triggers
        )

    def test_mode_tags_personal_only(self) -> None:
        assert ChannelWriterSkill.mode_tags == ["personal"]
