"""Skill manifest loading and validation.

Reads ``skill.yaml`` files next to ``skill.py``, validates them against a
Pydantic schema, and verifies that the class attributes of a v4 skill match
its manifest declarations.

This is the minimal subset needed for phase 3 of the v3 → v4 migration:
the loader exists so the migrated ``channel_writer`` skill (and any future
v4 skills) can be validated at startup. Full parameter loading (KB #85) is
deferred to a later phase — only the ``parameters`` schema is declared here
to accept existing manifests without failing.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from src.skills.base import CoreCapability, SideEffect


class SkillParameter(BaseModel):
    """Single parameter entry in a skill.yaml ``parameters`` section."""

    default: Any
    description: str
    type: str  # int | float | str | bool | list | dict
    min: float | None = None
    max: float | None = None
    choices: list[Any] | None = None
    env_override: str | None = None
    tier: int = 1


class SkillManifest(BaseModel):
    """Top-level schema for skill.yaml manifests (KB #80 + #85)."""

    name: str
    description: str
    version: str
    type: str  # inline | delegated | background
    llm_tier: str  # worker | analyst | strategist
    triggers: list[str] = Field(default_factory=list)
    cost_estimate: str = "low"
    approval_policy: str = "auto"
    side_effects: list[str] = Field(default_factory=list)
    mode_tags: list[str] = Field(default_factory=lambda: ["personal"])
    modifies: list[str] = Field(default_factory=list)
    parameters: dict[str, SkillParameter] = Field(default_factory=dict)
    dependencies: dict[str, list[str]] = Field(default_factory=dict)
    executor: str | None = None
    max_duration_seconds: float | None = None
    trigger_type: str | None = None
    trigger_config: dict[str, Any] = Field(default_factory=dict)
    source: str = "manual"
    status: str = "enabled"
    disabled_reason: str = ""
    created_at: str | None = None
    tests: dict[str, str] = Field(default_factory=dict)

    @field_validator("side_effects")
    @classmethod
    def _validate_side_effects(cls, v: list[str]) -> list[str]:
        valid = {e.value for e in SideEffect}
        for item in v:
            if item not in valid:
                raise ValueError(
                    f"Unknown side effect: {item!r}. Valid values: {sorted(valid)}"
                )
        return v

    @field_validator("modifies")
    @classmethod
    def _validate_modifies(cls, v: list[str]) -> list[str]:
        valid = {e.value for e in CoreCapability}
        for item in v:
            if item not in valid:
                raise ValueError(
                    f"Unknown core capability: {item!r}. Valid values: {sorted(valid)}"
                )
        return v

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in {"inline", "delegated", "background"}:
            raise ValueError(
                f"Unknown skill type: {v!r}. Valid: 'inline', 'delegated', 'background'"
            )
        return v

    @field_validator("llm_tier")
    @classmethod
    def _validate_llm_tier(cls, v: str) -> str:
        if v not in {"worker", "analyst", "strategist"}:
            raise ValueError(
                f"Unknown llm_tier: {v!r}. Valid: 'worker', 'analyst', 'strategist'"
            )
        return v

    @field_validator("cost_estimate")
    @classmethod
    def _validate_cost_estimate(cls, v: str) -> str:
        if v not in {"low", "medium", "high"}:
            raise ValueError(
                f"Unknown cost_estimate: {v!r}. Valid: 'low', 'medium', 'high'"
            )
        return v

    @field_validator("approval_policy")
    @classmethod
    def _validate_approval_policy(cls, v: str) -> str:
        if v not in {"auto", "required", "mode_dependent"}:
            raise ValueError(
                f"Unknown approval_policy: {v!r}. "
                f"Valid: 'auto', 'required', 'mode_dependent'"
            )
        return v

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        if v not in {"enabled", "disabled", "experimental"}:
            raise ValueError(
                f"Unknown status: {v!r}. Valid: 'enabled', 'disabled', 'experimental'"
            )
        return v


def load_manifest_for_skill_class(skill_class: type) -> SkillManifest:
    """Load ``skill.yaml`` from the directory containing the skill class module.

    Raises:
        FileNotFoundError: if ``skill.yaml`` is missing next to ``skill.py``.
        pydantic.ValidationError: if the YAML does not match the schema.
    """
    module_file = inspect.getfile(skill_class)
    manifest_path = Path(module_file).parent / "skill.yaml"

    if not manifest_path.exists():
        raise FileNotFoundError(f"skill.yaml not found: {manifest_path}")

    with manifest_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return SkillManifest(**data)


def validate_manifest_matches_class(manifest: SkillManifest, skill_class: type) -> None:
    """Verify class attributes match manifest declarations.

    Checks ``name``, ``description``, ``llm_tier``, and ``skill_type``
    (vs. manifest ``type``).

    Raises:
        ValueError: if any class attribute disagrees with the manifest.
    """
    checks: list[tuple[str, Any, Any]] = [
        ("name", manifest.name, getattr(skill_class, "name", None)),
        (
            "description",
            manifest.description,
            getattr(skill_class, "description", None),
        ),
        ("llm_tier", manifest.llm_tier, getattr(skill_class, "llm_tier", None)),
        ("type", manifest.type, getattr(skill_class, "skill_type", None)),
    ]
    mismatches = [
        (field_name, yaml_val, class_val)
        for field_name, yaml_val, class_val in checks
        if yaml_val != class_val
    ]
    if mismatches:
        raise ValueError(
            f"Manifest/class mismatch for {skill_class.__name__}: {mismatches}"
        )
