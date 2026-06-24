"""Architect prompt content tests.

The Architect's system prompt is the only place where the SDK learns the
spec schema and writing conventions. When the SpecModel grows a new
field, the system prompt must mention it — otherwise the SDK omits the
field from drafts and the downstream validator rejects them (or, worse,
fills a default that masks the omission).

These tests pin the field-mention contract for ``spec.kind`` (the
feature/refactor/fix/docs classification that lets the Editor decide
whether existing tests can be touched).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract


class TestArchitectSystemPromptMentionsKind:
    """``IDEATION_SYSTEM_PROMPT`` must teach the SDK to set ``kind``.

    Without this, the SDK omits ``kind`` from drafts; the SpecModel
    default (FEATURE) silently kicks in even on a refactor request, and
    the Editor stays in the strictest test-immutable mode by accident.
    """

    def test_prompt_mentions_kind_field(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        assert "kind" in IDEATION_SYSTEM_PROMPT.lower()

    def test_prompt_lists_all_kind_values(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT
        from src.skills.spec_command.parser import SpecKind

        lower = IDEATION_SYSTEM_PROMPT.lower()
        for kind in SpecKind:
            assert kind.value in lower, (
                f"system prompt missing kind value '{kind.value}'"
            )


class TestArchitectSystemPromptPreservesZhvushaIdentity:
    """Self-coding must not collapse into a detached coding service.

    The Architect still returns strict YAML, but the minimal identity
    anchor matters: specs are Жвуша planning work for Никита, not a
    generic assistant emitting tickets.
    """

    def test_prompt_names_zhvusha_and_nikita(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        assert "жвуш" in IDEATION_SYSTEM_PROMPT.lower()
        assert "никит" in IDEATION_SYSTEM_PROMPT.lower()

    def test_prompt_rejects_generic_coding_service_voice(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        assert "detached coding service" in lower
        assert "generic assistant" in lower
        assert "strict yaml" in lower


class TestArchitectNoDowngradePrinciple:
    def test_prompt_requires_preserve_behavior_field(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        assert "preserve_behavior" in lower
        assert "non-empty" in lower or "≥1" in IDEATION_SYSTEM_PROMPT

    def test_prompt_defaults_allowed_simplifications_to_empty(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        assert "allowed_simplifications" in lower
        assert "[]" in IDEATION_SYSTEM_PROMPT
        assert "explicitly approved" in lower or "никита explicitly" in lower

    def test_prompt_treats_growth_as_enrichment_not_flattening(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        assert "enrichment" in lower
        assert "no-downgrade" in lower
        assert "flatten" in lower

    def test_prompt_allows_clarification_when_simplification_is_unclear(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        assert "CLARIFICATION_NEEDED" in IDEATION_SYSTEM_PROMPT
        lower = IDEATION_SYSTEM_PROMPT.lower()
        assert "clarifying question" in lower or "ask никита" in lower

    def test_prompt_applies_to_future_self_growth_and_internet_topics(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        assert "future self-growth" in lower
        assert "internet" in lower

    def test_prompt_mentions_chat_context_field(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        assert "chat_context" in lower
        assert "/код" in IDEATION_SYSTEM_PROMPT


class TestArchitectSystemPromptDirectsRepoExploration:
    """``IDEATION_SYSTEM_PROMPT`` must direct the SDK to verify paths
    via Grep/Glob/Read BEFORE writing whitelist_paths or
    failing_test.file.

    Without this, Architect guesses typical layouts (``tests/test_X.py``)
    instead of the project's actual structure
    (``tests/skills/<module>/test_X.py``), producing specs whose whitelist
    omits the real callsites and whose failing_test points at
    non-existent files. Editor then either fails to commit (path outside
    whitelist) or fails to find the test it's supposed to make green.

    The fix is a prompt rule, not a hook — Architect already has Read /
    Grep / Glob in its allowed_tools; we just have to tell it to use
    them.
    """

    def test_prompt_requires_grep_or_glob_before_writing_paths(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # Must explicitly tell the SDK to use the search tools.
        assert "grep" in lower or "glob" in lower, (
            "Architect prompt must mention Grep/Glob to direct path verification"
        )

    def test_prompt_warns_against_guessing_paths(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # Must explicitly forbid guessing — otherwise the model defaults
        # to its prior on typical project layouts.
        assert any(
            phrase in lower
            for phrase in (
                "do not guess",
                "don't guess",
                "never guess",
                "verify",
                "must exist",
            )
        )

    def test_prompt_mentions_paths_must_exist(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # Must connect the rule to whitelist_paths / failing_test
        # specifically, not just say "verify things".
        assert "whitelist" in lower or "path" in lower

    def test_prompt_requires_opening_urls_before_spec(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        assert "url" in lower
        assert "open that url" in lower or "open the url" in lower
        assert "research_findings" in lower


class TestArchitectSystemPromptClassifiesGrepMatches:
    """``IDEATION_SYSTEM_PROMPT`` must direct the SDK to classify each
    Grep match by its surrounding syntactic context BEFORE adding the
    file to ``whitelist_paths``.

    Background — the refactor-presets-test-subset cycle exposed this:
    Architect Grep'd ``test_four_presets_defined`` and ``TestResearchPresets``
    across the repo, found them in 6 test files, and put all 6 into
    ``whitelist_paths``. Five of those matches were *string literals*
    inside Phase 16 contract tests — examples of an
    ``ExistingTestUpdate`` test_name field, or docstring backstory
    referring to the symbol abstractly. None of them were real
    callsites; renaming the actual test does not require editing those
    test data lines or docstrings.

    Without this rule, every rename spec where the symbol's name also
    appears in unrelated test data / docstrings produces a bloated
    whitelist. Editor then has license to mutate files that the spec
    has no business touching, and the surgical-change discipline
    breaks down.

    The classification is structural, not semantic: a callsite is an
    ``import``, a function/method call, an attribute access, a
    decorator, or a class base name. A *mention* is a string literal,
    a docstring, a comment, or test fixture data. Both look identical
    in raw Grep output, so the prompt must direct the SDK to inspect
    the line + surrounding context before deciding.

    Tests pin the prompt's vocabulary, not the SDK's runtime classifier
    (which lives behind a live model call). The rule must:
    * direct the SDK to classify each match,
    * distinguish callsite shapes from mention shapes,
    * connect the classification to whitelist_paths inclusion.
    """

    def test_prompt_directs_classifying_grep_matches(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # Must explicitly direct the SDK to classify / inspect each
        # match's surrounding context, not just count hits.
        assert any(
            phrase in lower
            for phrase in (
                "classify",
                "classification",
                "inspect each match",
                "inspect the line",
                "examine each match",
                "context around",
                "surrounding context",
                "syntactic context",
            )
        ), "prompt must direct match classification, not raw Grep counting"

    def test_prompt_names_callsite_shapes(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # Must literally name at least one callsite shape so the SDK
        # has a concrete checklist.
        assert "callsite" in lower or "call site" in lower, (
            "prompt must use the term 'callsite' to name the relevant category"
        )
        assert any(
            shape in lower
            for shape in (
                "import",
                "function call",
                "method call",
                "attribute access",
                "decorator",
            )
        ), "prompt must name at least one concrete callsite shape"

    def test_prompt_names_mention_shapes(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # Must name the non-callsite category so the SDK can recognise
        # it as exclusion-worthy.
        assert any(
            term in lower
            for term in (
                "string literal",
                "string-literal",
                "docstring",
                "doc-string",
                "comment",
                "fixture data",
                "test data",
                "mention",
            )
        ), "prompt must name at least one mention shape (string/docstring/comment)"

    def test_prompt_directs_only_callsites_into_whitelist(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # The classification must be tied to whitelist_paths inclusion,
        # not left as abstract advice. The prompt must say something
        # like "only callsites belong in whitelist_paths" or "mentions
        # do not warrant whitelist entry".
        assert any(
            phrase in lower
            for phrase in (
                "only callsites",
                "only call sites",
                "callsites belong",
                "callsites go in",
                "mentions do not",
                "mentions are excluded",
                "string literal does not",
                "string literals do not",
                "docstrings do not",
                "do not add",
                "exclude mentions",
            )
        ), (
            "prompt must connect classification to whitelist_paths "
            "(only callsites go in, mentions stay out)"
        )

    def test_prompt_directs_read_context_around_matches(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # The mechanism for classification is Read'ing context around
        # the matched line — Grep alone gives no surrounding tokens.
        # Either the prompt names "Read" + a small N of context lines,
        # or it explicitly directs context-aware -B/-A flags on Grep.
        assert "read" in lower
        # Must mention reading or extracting *context* / *surrounding*
        # lines, not just the match line.
        assert any(
            term in lower
            for term in (
                "context line",
                "context lines",
                "surrounding line",
                "lines around",
                "few lines",
                "neighbouring",
                "neighboring",
                "before/after",
                "-b/-a",
                "before and after",
            )
        ), "prompt must direct reading lines around each match"

    def test_prompt_requires_env_runtime_boundary_scan(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        assert "env/config/runtime boundary scan" in lower
        assert ".env.example" in lower
        assert "safe/off" in lower
        assert "systemd" in lower
        assert "clarification_needed" in lower
        assert 'never write "function is enabled"' in lower

    def test_prompt_includes_live_env_denylist(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        assert "BOT_TOKEN" in IDEATION_SYSTEM_PROMPT
        assert "DATABASE_URL" in IDEATION_SYSTEM_PROMPT
        assert "*_API_KEY" in IDEATION_SYSTEM_PROMPT
        assert "*_PASSWORD" in IDEATION_SYSTEM_PROMPT
        assert "Protected .env denylist" in IDEATION_SYSTEM_PROMPT


class TestArchitectSystemPromptDetectsHiddenTestContracts:
    """``IDEATION_SYSTEM_PROMPT`` must direct the SDK to scan read_only
    test files for anti-pattern asserts (fixed-set / fixed-length /
    fixed-list equality) BEFORE finalising the spec.

    Background — the bug_investigation preset cycle exposed this:
    ``tests/research/test_research_service.py::test_four_presets_defined``
    asserted ``names == {"foundational", "current_practices",
    "api_integration", "hot_topic"}``. The spec added a fifth preset to
    a read-only-listed file. Editor (correctly, by ``kind: feature``
    rules) refused to modify the existing test, the test failed, the
    commit gate aborted. Architect did not see this hidden contract at
    spec time because it never inspected the read_only test bodies.

    Without this rule, every feature that extends a finite collection
    behind a count/set assert silently stalls the Editor cycle. The
    detection is mechanical (a Grep against well-known assert shapes)
    so the SDK can do it deterministically — but only if the prompt
    tells it to.

    Tests below pin the prompt's vocabulary, not the SDK's runtime
    behaviour (which lives behind a live model call). The rule must
    name the patterns to look for, the place to look (read_only_paths
    test files), and the action on detection (escalate / mark for
    update / propose a pre-spec).
    """

    def test_prompt_directs_scan_of_readonly_test_files(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # Must explicitly direct the SDK to inspect tests in
        # read_only_paths (or read_only files generally) before
        # finalising the spec.
        assert "read_only" in lower or "read-only" in lower, (
            "Architect prompt must reference read_only test inspection"
        )

    def test_prompt_names_fixed_equality_anti_pattern(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        # The prompt must literally name the assert shapes the SDK is
        # supposed to grep for. "anti-pattern" / "hidden contract" alone
        # is too abstract — model needs concrete syntax.
        text = IDEATION_SYSTEM_PROMPT
        assert "==" in text, (
            "Architect prompt must mention `==` equality asserts as the scan target"
        )

    def test_prompt_lists_set_length_or_collection_examples(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # At least one concrete shape must be named — set, length,
        # list, or "fixed collection" — so the SDK knows what kind of
        # contract is brittle.
        assert any(
            term in lower
            for term in (
                "set equality",
                "fixed set",
                "fixed-set",
                "len(",
                "length",
                "fixed collection",
                "fixed-collection",
                "exact set",
            )
        ), "prompt must name at least one concrete brittle-assert shape"

    def test_prompt_uses_grep_on_assert_patterns(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # The detection mechanism must be Grep — same tool the SDK
        # already has and uses for path resolution.
        assert "grep" in lower, "detection mechanism must be Grep"
        # And the target of the grep must be assert/expression patterns,
        # not just symbols.
        assert "assert" in lower, (
            "prompt must direct the SDK to grep for `assert` lines"
        )

    def test_prompt_specifies_action_on_detection(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # On detection, the SDK must be told what to do: either include
        # the test in a planned-update field, or escalate (separate
        # pre-spec, abort, ask for clarification). Vague "be careful"
        # doesn't qualify.
        assert "existing_tests_to_update" in lower, (
            "prompt must reference the existing_tests_to_update field "
            "as the legitimate channel for planned test mutations"
        )
        assert any(
            phrase in lower
            for phrase in (
                "pre-spec",
                "separate spec",
                "another spec",
                "first spec",
                "fix the test first",
            )
        ), "prompt must offer the alternative of a separate pre-spec"

    def test_prompt_explains_why_addition_to_collection_is_risk(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # Concrete trigger phrasing — the SDK must learn that adding
        # an element to a finite collection (preset, registry entry,
        # enum value) is the canonical case where this rule fires.
        assert any(
            phrase in lower
            for phrase in (
                "extend a collection",
                "extends a collection",
                "extending a collection",
                "add to a collection",
                "adding to a collection",
                "adds to a collection",
                "new entry to",
                "new element to",
                "new item to",
            )
        ), "prompt must concretise the trigger as collection-extension"


class TestExistingFormatProducerScan:
    """Phase 40 follow-up — when a spec changes how some output format
    is produced/parsed/rendered (e.g. adding ``parse_mode``, changing a
    converter), Architect must Grep for existing producers of that
    same format and explicitly decide what to do with each. Without
    this step a universal converter silently breaks producers that
    already emit the canonical form (the «двойная конвертация Phase 40
    welcome» bug).
    """

    def test_prompt_mentions_existing_format_producers_scan(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # Must contain the trigger phrasing and the concept.
        assert "producer" in lower or "produce" in lower, (
            "Architect prompt must reference existing producers of the format"
        )
        assert any(
            term in lower
            for term in (
                "format",
                "render",
                "parse_mode",
                "converter",
                "serializ",
                "escape",
            )
        ), "prompt must name format/render/converter as the trigger surface"

    def test_prompt_grep_for_existing_producers(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        # The detection mechanism is Grep, same as for hidden test
        # contracts. Tool reuse keeps Architect's surface narrow.
        assert "Grep" in IDEATION_SYSTEM_PROMPT, (
            "prompt must direct Grep as the producer-detection mechanism"
        )

    def test_prompt_lists_concrete_format_examples(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        # Architect needs concrete syntactic examples of what to grep
        # for, otherwise it skips the step. HTML and markdown are the
        # two canonical Telegram-bot cases.
        text = IDEATION_SYSTEM_PROMPT
        assert "<b>" in text or "<code>" in text or "parse_mode" in text, (
            "prompt must show at least one concrete HTML producer pattern"
        )
        assert "**" in text or "`" in text, (
            "prompt must show at least one concrete markdown producer pattern"
        )

    def test_prompt_specifies_action_branches(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # Three legitimate dispositions on a found producer.
        assert "whitelist_paths" in lower, (
            "prompt must offer adapting the producer (add to whitelist)"
        )
        assert "read_only_paths" in lower, (
            "prompt must offer pinning producer as read-only"
        )

    def test_prompt_explains_double_conversion_risk(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # The why — Architect must understand the failure mode, not
        # just the rule. «Universal converter breaks existing producers»
        # is the canonical phrasing; we accept several wordings.
        assert any(
            phrase in lower
            for phrase in (
                "double",
                "twice",
                "already emit",
                "existing producer",
                "silently break",
                "broken by",
            )
        ), "prompt must explain the double-conversion / silent-break risk"


class TestStep6CoversAllContractTypes:
    """Step 6 must apply to ANY contract a spec changes, not only the
    output-format case that prompted it. Symmetric to Step 5 (which
    catches brittle asserts in tests) — Step 6 must catch brittle
    dependencies in production code.
    """

    def test_step_6_states_general_contract_principle(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # General principle: ANY contract, not just format.
        assert "contract" in lower, (
            "step 6 must explicitly use the word `contract` to cover the "
            "general case (Pydantic model, Protocol, DB schema, event payload), "
            "not only output format"
        )

    def test_step_6_covers_pydantic_model_or_schema(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # Pydantic model or generic schema is a contract dependents
        # rely on (constructor calls, field access, serialization).
        assert any(
            term in lower for term in ("pydantic", "schema", "model field", "field add")
        ), (
            "step 6 must list Pydantic model / schema changes as a "
            "category that triggers the dependents scan"
        )

    def test_step_6_covers_protocol_or_interface(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        # Protocols are the v4 isolation boundary — changing them
        # affects every implementer and call site.
        text = IDEATION_SYSTEM_PROMPT
        assert (
            "Protocol" in text
            or "protocol" in text.lower()
            or "interface" in text.lower()
        ), "step 6 must list Protocol / interface changes as a category"

    def test_step_6_covers_db_or_event_payloads(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        # DB schema (columns, tables) or event payloads (Redis Streams,
        # Pub/Sub) are common contract surfaces.
        assert any(
            term in lower
            for term in (
                "db schema",
                "database schema",
                "column",
                "migration",
                "event payload",
                "publisher",
                "subscriber",
                "redis stream",
            )
        ), "step 6 must list DB schema OR event-payload changes as a category"

    def test_step_6_lists_grep_patterns_per_category(self) -> None:
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        text = IDEATION_SYSTEM_PROMPT
        # The prompt was concrete enough about format producers
        # (`<b>`, `**`); it must stay equally concrete for other
        # categories. At least one of these production-facing Grep
        # patterns must show up.
        assert any(
            pat in text
            for pat in (
                "model_validate",
                "BaseModel",
                ".publish(",
                ".xadd(",
                "class .* (Protocol)",
                "alembic",
                "ALTER TABLE",
            )
        ), (
            "step 6 must include at least one concrete Grep pattern for a "
            "non-format contract (Pydantic, Protocol, event, DB)"
        )

    def test_step_6_disposition_branches_remain_three(self) -> None:
        """Three named dispositions (adapt / pin read-only / convert at
        boundary) must apply to any contract type, not only format.
        Re-checks they are still present after generalisation."""
        from src.skills.ideation_to_spec.prompts import IDEATION_SYSTEM_PROMPT

        lower = IDEATION_SYSTEM_PROMPT.lower()
        assert "whitelist_paths" in lower
        assert "read_only_paths" in lower
        # «Convert at boundary» phrasing or a synonym.
        assert any(
            phrase in lower
            for phrase in (
                "boundary",
                "convert at the",
                "fork",
                "explicitly route",
            )
        ), "third disposition (boundary / fork) must remain"
