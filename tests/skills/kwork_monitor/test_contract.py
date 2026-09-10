"""Contract tests for the migrated kwork_monitor v4 skill."""

from __future__ import annotations

import pytest
from src.skills.base import AgentContext, BackgroundSkill, BaseSkill, SideEffect
from src.skills.kwork_monitor.skill import KworkMonitorSkill
from src.skills.manifest import (
    load_manifest_for_skill_class,
    validate_manifest_matches_class,
)

pytestmark = pytest.mark.contract


class TestContract:
    def test_inherits_background_skill(self) -> None:
        assert issubclass(KworkMonitorSkill, BackgroundSkill)
        assert issubclass(KworkMonitorSkill, BaseSkill)

    def test_skill_type_is_background(self) -> None:
        assert KworkMonitorSkill.skill_type == "background"

    def test_manifest_loads(self) -> None:
        manifest = load_manifest_for_skill_class(KworkMonitorSkill)
        assert manifest.name == "kwork_monitor"
        assert manifest.type == "background"
        assert manifest.llm_tier == "analyst"
        assert manifest.version == "1.0.0"

    def test_manifest_matches_class(self) -> None:
        manifest = load_manifest_for_skill_class(KworkMonitorSkill)
        validate_manifest_matches_class(manifest, KworkMonitorSkill)

    def test_trigger_type_is_interval(self) -> None:
        assert KworkMonitorSkill.trigger_type == "interval"

    def test_trigger_config_has_interval_seconds(self) -> None:
        assert "interval_seconds" in KworkMonitorSkill.trigger_config

    def test_side_effects_cover_polling(self) -> None:
        assert SideEffect.READS_FROM_KB in KworkMonitorSkill.side_effects
        assert SideEffect.CALLS_LLM in KworkMonitorSkill.side_effects
        assert SideEffect.SENDS_TELEGRAM_MESSAGE in KworkMonitorSkill.side_effects
        assert SideEffect.NETWORK_IO_EXTERNAL in KworkMonitorSkill.side_effects

    def test_modifies_no_core_capabilities(self) -> None:
        assert KworkMonitorSkill.modifies == []

    def test_mode_tags_personal_only(self) -> None:
        assert KworkMonitorSkill.mode_tags == ["personal"]

    async def test_can_handle_claims_explicit_chat_first_controls(self) -> None:
        skill = KworkMonitorSkill()
        context = AgentContext(user_id=1, chat_id=1, mode="personal")
        assert await skill.can_handle("/kwork", context) >= 0.9
        assert await skill.can_handle("покажи свежие kwork", context) >= 0.9
        assert await skill.can_handle("усыпи мониторинг", context) >= 0.9
        assert await skill.can_handle("разбуди мониторинг", context) >= 0.9
        assert await skill.can_handle("обсудим мониторинг фриланса", context) == 0.0

    async def test_execute_natural_sleep_and_wake_controls(self) -> None:
        skill = KworkMonitorSkill()
        context = AgentContext(user_id=1, chat_id=1, mode="personal")

        sleep_result = await skill.execute("усыпи мониторинг", context)
        wake_result = await skill.execute("разбуди мониторинг", context)

        assert sleep_result.success is True
        assert "приостановлен" in sleep_result.response
        assert wake_result.success is True
        assert "возобновл" in wake_result.response

    async def test_natural_sleep_uses_skill_approval_gate(self) -> None:
        from src.skills.invocation import (
            InMemorySkillApprovalStore,
            SkillInvocationService,
        )

        async def _approval_classifier(text: str) -> str:
            del text
            return "yes"

        skill = KworkMonitorSkill()
        await skill.wake()
        context = AgentContext(user_id=1, chat_id=1, mode="personal")
        service = SkillInvocationService(
            approval_store=InMemorySkillApprovalStore(),
            approval_classifier=_approval_classifier,
            is_skill_allowed=lambda _name, _mode: True,
        )

        pending = await service.dispatch("усыпи мониторинг", context, [skill])

        assert pending.result is not None
        assert pending.result.metadata["approval_pending"] is True
        assert skill.is_sleeping is False

        approved = await service.dispatch("да", context, [skill])

        assert approved.result is not None
        assert approved.result.success is True
        assert skill.is_sleeping is True


class TestManifestParameters:
    def test_parameters_include_polling_knobs(self) -> None:
        manifest = load_manifest_for_skill_class(KworkMonitorSkill)
        assert "poll_interval_seconds" in manifest.parameters
        assert "seen_ttl_seconds" in manifest.parameters
        assert "default_sleep_hours" in manifest.parameters

    def test_redis_key_parameter_is_tier_2(self) -> None:
        manifest = load_manifest_for_skill_class(KworkMonitorSkill)
        assert manifest.parameters["redis_seen_key"].tier == 2
