"""Hermes parity evaluation harness contracts."""

from __future__ import annotations


def test_evaluation_harness_scores_groups_and_surfaces_concrete_gaps() -> None:
    from src.skills.external_skill_loader.evaluation import (
        EvaluationHarness,
        EvaluationRunResult,
        EvaluationTask,
        HermesParityBenchmarkGroup,
        HermesParityTaskCategory,
    )

    task = EvaluationTask(
        task_id="kube-ingress-debug",
        title="Debug Kubernetes ingress",
        category=HermesParityTaskCategory.EXTERNAL_SKILL_ACQUISITION,
        required_capabilities=("kubernetes_debug", "browser_read"),
    )
    harness = EvaluationHarness(tasks=(task,))
    report = harness.evaluate(
        results=(
            EvaluationRunResult(
                task_id=task.task_id,
                group=HermesParityBenchmarkGroup.DIRECT_HERMES,
                solved=True,
                nikita_interventions=1,
                unsafe_action_attempts=0,
                evidence_quality=0.8,
                final_answer_usefulness=0.8,
                verification_quality=0.7,
                recovery_quality=0.6,
            ),
            EvaluationRunResult(
                task_id=task.task_id,
                group=HermesParityBenchmarkGroup.ZHVUSHA_WITH_EXECUTION_ADAPTERS,
                solved=False,
                nikita_interventions=3,
                unsafe_action_attempts=0,
                evidence_quality=0.4,
                final_answer_usefulness=0.3,
                verification_quality=0.2,
                recovery_quality=0.2,
                notes="No Kubernetes external skill import route.",
            ),
        )
    )

    assert (
        report.group_summaries[HermesParityBenchmarkGroup.DIRECT_HERMES].completion_rate
        == 1.0
    )
    assert (
        report.group_summaries[
            HermesParityBenchmarkGroup.ZHVUSHA_WITH_EXECUTION_ADAPTERS
        ].completion_rate
        == 0.0
    )
    assert report.gaps[0].task_id == "kube-ingress-debug"
    assert report.gaps[0].winning_group is HermesParityBenchmarkGroup.DIRECT_HERMES
    assert report.gaps[0].lagging_group is (
        HermesParityBenchmarkGroup.ZHVUSHA_WITH_EXECUTION_ADAPTERS
    )
    assert report.gaps[0].capability_backlog_item == (
        "Close Hermes parity gap for kube-ingress-debug: kubernetes_debug, browser_read"
    )
    assert "Unsafe action attempts: 0" in report.render_markdown()


def test_evaluation_harness_requires_registered_tasks_for_results() -> None:
    from src.skills.external_skill_loader.evaluation import (
        EvaluationHarness,
        EvaluationRunResult,
        HermesParityBenchmarkGroup,
    )

    harness = EvaluationHarness(tasks=())

    try:
        harness.evaluate(
            results=(
                EvaluationRunResult(
                    task_id="unknown",
                    group=HermesParityBenchmarkGroup.DIRECT_HERMES,
                    solved=True,
                ),
            )
        )
    except ValueError as exc:
        assert "unknown evaluation task" in str(exc)
    else:
        raise AssertionError("evaluation must reject results for unknown tasks")


def test_parity_gate_blocks_missing_baselines_and_missing_gap_artifacts() -> None:
    from src.skills.external_skill_loader.evaluation import (
        EvaluationHarness,
        EvaluationRunResult,
        EvaluationTask,
        HermesParityBenchmarkGroup,
        HermesParityGate,
        HermesParityTaskCategory,
    )

    task = EvaluationTask(
        task_id="browser-research-draft",
        title="Research and draft browser workflow",
        category=HermesParityTaskCategory.BROWSER_WORKFLOW_DRAFT,
        required_capabilities=("browser_read", "browser_draft_form"),
    )
    report = EvaluationHarness(tasks=(task,)).evaluate(
        results=(
            EvaluationRunResult(
                task_id=task.task_id,
                group=HermesParityBenchmarkGroup.DIRECT_HERMES,
                solved=True,
                evidence_quality=0.9,
                final_answer_usefulness=0.9,
                verification_quality=0.8,
                recovery_quality=0.7,
            ),
            EvaluationRunResult(
                task_id=task.task_id,
                group=HermesParityBenchmarkGroup.ZHVUSHA_WITH_EXECUTION_ADAPTERS,
                solved=False,
                evidence_quality=0.2,
                final_answer_usefulness=0.2,
                verification_quality=0.1,
                recovery_quality=0.1,
            ),
        )
    )

    decision = HermesParityGate(
        required_categories=(HermesParityTaskCategory.BROWSER_WORKFLOW_DRAFT,),
    ).evaluate(report)

    assert decision.ready is False
    assert any("missing direct_codex baseline" in item for item in decision.blockers)
    assert any(
        "missing zhvusha_without_imported_skills baseline" in item
        for item in decision.blockers
    )
    assert any(
        "written gap report artifact is required" in item for item in decision.blockers
    )
    assert decision.next_capability_backlog == (
        "Close Hermes parity gap for browser-research-draft: browser_read, browser_draft_form",
    )
    assert "NOT READY" in decision.render_markdown()


def test_parity_gate_passes_only_with_full_baselines_no_gaps_and_backlog_artifact() -> (
    None
):
    from src.skills.external_skill_loader.evaluation import (
        EvaluationHarness,
        EvaluationRunResult,
        EvaluationTask,
        HermesParityBenchmarkGroup,
        HermesParityGate,
        HermesParityTaskCategory,
    )

    task = EvaluationTask(
        task_id="memory-recall",
        title="Recall a prior decision with source",
        category=HermesParityTaskCategory.MEMORY_RECALL_FOLLOWUP,
        required_capabilities=("source_backed_recall",),
    )
    results = tuple(
        EvaluationRunResult(
            task_id=task.task_id,
            group=group,
            solved=True,
            evidence_quality=0.95,
            final_answer_usefulness=0.95,
            verification_quality=0.9,
            recovery_quality=0.9,
            remembered_lesson_later=group
            is HermesParityBenchmarkGroup.ZHVUSHA_WITH_EXECUTION_ADAPTERS,
        )
        for group in HermesParityBenchmarkGroup
    )
    report = EvaluationHarness(tasks=(task,)).evaluate(results=results)

    decision = HermesParityGate(
        required_categories=(HermesParityTaskCategory.MEMORY_RECALL_FOLLOWUP,),
    ).evaluate(
        report,
        gap_report_artifact="reports/hermes-parity.md",
        next_capability_backlog_artifact="docs/recent-unimplemented-backlog.md",
    )

    assert decision.ready is True
    assert decision.blockers == ()
    assert decision.render_markdown().startswith("# Hermes parity gate: READY")


def test_parity_artifact_writer_persists_gap_report_and_backlog(tmp_path) -> None:
    from src.skills.external_skill_loader.evaluation import (
        EvaluationHarness,
        EvaluationRunResult,
        EvaluationTask,
        HermesParityArtifactWriter,
        HermesParityBenchmarkGroup,
        HermesParityGate,
        HermesParityTaskCategory,
    )

    task = EvaluationTask(
        task_id="external-skill-import",
        title="Import and use an external skill",
        category=HermesParityTaskCategory.EXTERNAL_SKILL_ACQUISITION,
        required_capabilities=("external_skill_import", "external_skill_execute"),
    )
    report = EvaluationHarness(tasks=(task,)).evaluate(
        results=(
            EvaluationRunResult(
                task_id=task.task_id,
                group=HermesParityBenchmarkGroup.DIRECT_HERMES,
                solved=True,
                evidence_quality=0.9,
                final_answer_usefulness=0.9,
                verification_quality=0.8,
                recovery_quality=0.8,
            ),
            EvaluationRunResult(
                task_id=task.task_id,
                group=HermesParityBenchmarkGroup.ZHVUSHA_WITH_EXECUTION_ADAPTERS,
                solved=False,
                evidence_quality=0.4,
                final_answer_usefulness=0.4,
                verification_quality=0.3,
                recovery_quality=0.3,
                notes="execution path missing one approval bridge",
            ),
        )
    )

    artifacts = HermesParityArtifactWriter(root=tmp_path).write(
        report=report,
        gate=HermesParityGate(
            required_groups=(
                HermesParityBenchmarkGroup.DIRECT_HERMES,
                HermesParityBenchmarkGroup.ZHVUSHA_WITH_EXECUTION_ADAPTERS,
            ),
            required_categories=(HermesParityTaskCategory.EXTERNAL_SKILL_ACQUISITION,),
        ),
        gap_report_path="reports/hermes-parity.md",
        next_capability_backlog_path="reports/hermes-parity-backlog.md",
    )

    assert artifacts.decision.ready is False
    assert "written gap report artifact is required" not in artifacts.decision.blockers
    assert (tmp_path / "reports" / "hermes-parity.md").exists()
    assert (tmp_path / "reports" / "hermes-parity-backlog.md").exists()
    assert "Hermes parity evaluation" in (
        tmp_path / "reports" / "hermes-parity.md"
    ).read_text(encoding="utf-8")
    backlog = (tmp_path / "reports" / "hermes-parity-backlog.md").read_text(
        encoding="utf-8"
    )
    assert (
        "Close Hermes parity gap for external-skill-import: "
        "external_skill_import, external_skill_execute"
    ) in backlog


def test_default_hermes_parity_task_suite_covers_every_required_category() -> None:
    from src.skills.external_skill_loader.evaluation import (
        HermesParityTaskCategory,
        build_default_hermes_parity_tasks,
    )

    tasks = build_default_hermes_parity_tasks()
    categories = {task.category for task in tasks}

    assert categories == set(HermesParityTaskCategory)
    assert all(task.required_capabilities for task in tasks)
    assert all(task.success_criteria for task in tasks)


def test_file_parity_baseline_store_persists_manifest_results_and_artifacts(
    tmp_path,
) -> None:
    from src.skills.external_skill_loader.evaluation import (
        FileHermesParityBaselineStore,
        HermesParityArtifactWriter,
        HermesParityBenchmarkGroup,
        HermesParityGate,
        build_default_hermes_parity_tasks,
    )

    task = build_default_hermes_parity_tasks()[0]
    store = FileHermesParityBaselineStore(tmp_path / "parity")
    store.write_manifest((task,))
    store.append_result(
        task_id=task.task_id,
        group=HermesParityBenchmarkGroup.DIRECT_HERMES,
        solved=True,
        evidence_quality=0.9,
        final_answer_usefulness=0.9,
        verification_quality=0.8,
        recovery_quality=0.8,
        notes="direct Hermes baseline",
    )
    store.append_result(
        task_id=task.task_id,
        group=HermesParityBenchmarkGroup.ZHVUSHA_WITH_EXECUTION_ADAPTERS,
        solved=False,
        evidence_quality=0.4,
        final_answer_usefulness=0.4,
        verification_quality=0.3,
        recovery_quality=0.3,
        notes="ZHVUSHA run still lacks one tool",
    )

    report = store.evaluate()
    artifacts = HermesParityArtifactWriter(root=tmp_path).write(
        report=report,
        gate=HermesParityGate(
            required_groups=(
                HermesParityBenchmarkGroup.DIRECT_HERMES,
                HermesParityBenchmarkGroup.ZHVUSHA_WITH_EXECUTION_ADAPTERS,
            ),
            required_categories=(task.category,),
        ),
        gap_report_path="reports/hermes-parity.md",
        next_capability_backlog_path="reports/hermes-parity-backlog.md",
    )

    assert store.load_manifest() == (task,)
    assert len(store.load_results()) == 2
    assert report.gaps
    assert artifacts.gap_report_artifact == "reports/hermes-parity.md"
    assert (tmp_path / "reports" / "hermes-parity.md").exists()


def test_file_parity_baseline_store_loads_latest_result_per_task_group(
    tmp_path,
) -> None:
    from src.skills.external_skill_loader.evaluation import (
        FileHermesParityBaselineStore,
        HermesParityBenchmarkGroup,
        build_default_hermes_parity_tasks,
    )

    task = build_default_hermes_parity_tasks()[0]
    store = FileHermesParityBaselineStore(tmp_path / "parity")
    store.write_manifest((task,))
    store.append_result(
        task_id=task.task_id,
        group=HermesParityBenchmarkGroup.ZHVUSHA_WITHOUT_IMPORTED_SKILLS,
        solved=False,
        unsafe_action_attempts=1,
        notes="superseded unsafe run",
    )
    store.append_result(
        task_id=task.task_id,
        group=HermesParityBenchmarkGroup.ZHVUSHA_WITHOUT_IMPORTED_SKILLS,
        solved=True,
        unsafe_action_attempts=0,
        evidence_quality=0.9,
        final_answer_usefulness=0.9,
        verification_quality=0.9,
        recovery_quality=0.9,
        notes="clean rerun",
    )

    results = store.load_results()

    assert len(results) == 1
    assert results[0].solved is True
    assert results[0].unsafe_action_attempts == 0
    assert results[0].notes == "clean rerun"
    assert len((tmp_path / "parity" / "results.jsonl").read_text().splitlines()) == 2


def test_file_parity_baseline_store_rejects_unknown_task_results(tmp_path) -> None:
    from src.skills.external_skill_loader.evaluation import (
        FileHermesParityBaselineStore,
        HermesParityBenchmarkGroup,
    )

    store = FileHermesParityBaselineStore(tmp_path / "parity")
    store.write_manifest(())

    try:
        store.append_result(
            task_id="unknown",
            group=HermesParityBenchmarkGroup.DIRECT_HERMES,
            solved=True,
        )
    except ValueError as exc:
        assert "unknown evaluation task" in str(exc)
    else:  # pragma: no cover - explicit assertion readability
        raise AssertionError("unknown task result was accepted")


def test_file_parity_baseline_store_reports_missing_baseline_matrix(tmp_path) -> None:
    from src.skills.external_skill_loader.evaluation import (
        FileHermesParityBaselineStore,
        HermesParityBenchmarkGroup,
        build_default_hermes_parity_tasks,
    )

    task = build_default_hermes_parity_tasks()[0]
    store = FileHermesParityBaselineStore(tmp_path / "parity")
    store.write_manifest((task,))
    store.append_result(
        task_id=task.task_id,
        group=HermesParityBenchmarkGroup.DIRECT_HERMES,
        solved=True,
    )

    coverage = store.coverage(
        required_groups=(
            HermesParityBenchmarkGroup.DIRECT_HERMES,
            HermesParityBenchmarkGroup.DIRECT_CODEX,
            HermesParityBenchmarkGroup.ZHVUSHA_WITH_EXECUTION_ADAPTERS,
        )
    )

    assert coverage.ready is False
    assert [
        (missing.task_id, missing.group) for missing in coverage.missing_baselines
    ] == [
        (task.task_id, HermesParityBenchmarkGroup.DIRECT_CODEX),
        (task.task_id, HermesParityBenchmarkGroup.ZHVUSHA_WITH_EXECUTION_ADAPTERS),
    ]
    rendered = coverage.render_markdown()
    assert "Hermes baseline intake: NOT READY" in rendered
    assert f"missing {HermesParityBenchmarkGroup.DIRECT_CODEX.value}" in rendered
    assert task.task_id in rendered


def test_file_parity_baseline_store_creates_and_ingests_operator_scorecard(
    tmp_path,
) -> None:
    from src.skills.external_skill_loader.evaluation import (
        FileHermesParityBaselineStore,
        HermesParityBenchmarkGroup,
        build_default_hermes_parity_tasks,
    )

    task = build_default_hermes_parity_tasks()[0]
    store = FileHermesParityBaselineStore(tmp_path / "parity")
    store.write_manifest((task,))

    template = store.create_scorecard_template(
        task_id=task.task_id,
        group=HermesParityBenchmarkGroup.ZHVUSHA_WITH_EXECUTION_ADAPTERS,
    )
    scorecard = template.model_copy(
        update={
            "solved": True,
            "operator": "nikita",
            "evidence_quality": 0.9,
            "final_answer_usefulness": 0.8,
            "verification_quality": 0.85,
            "recovery_quality": 0.7,
            "evidence_artifacts": ("reports/run.md", "reports/run.json"),
            "notes": "real ZHVUSHA baseline",
        }
    )

    result = store.append_scorecard(scorecard)

    assert template.task_title == task.title
    assert template.success_criteria == task.success_criteria
    assert result.solved is True
    assert result.group is HermesParityBenchmarkGroup.ZHVUSHA_WITH_EXECUTION_ADAPTERS
    assert "real ZHVUSHA baseline" in result.notes
    assert "operator: nikita" in result.notes
    assert "evidence_artifacts: reports/run.md, reports/run.json" in result.notes


def test_file_parity_baseline_store_writes_and_ingests_scorecard_json(
    tmp_path,
) -> None:
    from src.skills.external_skill_loader.evaluation import (
        FileHermesParityBaselineStore,
        HermesBaselineScorecard,
        HermesParityBenchmarkGroup,
        build_default_hermes_parity_tasks,
    )

    task = build_default_hermes_parity_tasks()[0]
    store = FileHermesParityBaselineStore(tmp_path / "parity")
    store.write_manifest((task,))

    template_path = store.write_scorecard_template(
        task_id=task.task_id,
        group=HermesParityBenchmarkGroup.DIRECT_CODEX,
    )
    template_file = tmp_path / "parity" / template_path
    scorecard = HermesBaselineScorecard.model_validate_json(
        template_file.read_text(encoding="utf-8")
    ).model_copy(
        update={
            "solved": True,
            "operator": "nikita",
            "evidence_quality": 0.8,
            "final_answer_usefulness": 0.8,
            "verification_quality": 0.9,
            "recovery_quality": 0.6,
            "evidence_artifacts": ("reports/codex-run.md",),
            "notes": "real Codex baseline",
        }
    )
    template_file.write_text(
        scorecard.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    result = store.append_scorecard_json(template_path)

    assert template_path == f"scorecards/{task.task_id}--direct_codex.json"
    assert result.group is HermesParityBenchmarkGroup.DIRECT_CODEX
    assert result.notes.startswith("real Codex baseline")
    assert store.load_results() == (result,)


def test_file_parity_baseline_store_rejects_unfilled_scorecard_json(
    tmp_path,
) -> None:
    from src.skills.external_skill_loader.evaluation import (
        FileHermesParityBaselineStore,
        HermesParityBenchmarkGroup,
        build_default_hermes_parity_tasks,
    )

    task = build_default_hermes_parity_tasks()[0]
    store = FileHermesParityBaselineStore(tmp_path / "parity")
    store.write_manifest((task,))
    template_path = store.write_scorecard_template(
        task_id=task.task_id,
        group=HermesParityBenchmarkGroup.DIRECT_CODEX,
    )

    try:
        store.append_scorecard_json(template_path)
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - explicit assertion readability
        raise AssertionError("unfilled scorecard template was accepted")

    assert "baseline scorecard is not operator-filled" in message
    assert "operator is required" in message
    assert "notes are required" in message
    assert "evidence_artifacts is required" in message
    assert store.load_results() == ()


def test_file_parity_baseline_store_builds_operator_runbook_for_missing_scorecards(
    tmp_path,
) -> None:
    from src.skills.external_skill_loader.evaluation import (
        FileHermesParityBaselineStore,
        HermesParityBenchmarkGroup,
        build_default_hermes_parity_tasks,
    )

    task = build_default_hermes_parity_tasks()[0]
    store = FileHermesParityBaselineStore(tmp_path / "parity")
    store.write_manifest((task,))
    store.append_result(
        task_id=task.task_id,
        group=HermesParityBenchmarkGroup.DIRECT_HERMES,
        solved=True,
        operator="nikita",
        notes="real direct Hermes run",
        evidence_artifacts=("reports/hermes-baselines/evidence/hermes.md",),
    )

    runbook = store.build_runbook(
        required_groups=(
            HermesParityBenchmarkGroup.DIRECT_HERMES,
            HermesParityBenchmarkGroup.DIRECT_CODEX,
        )
    )

    assert runbook.ready is False
    assert len(runbook.steps) == 1
    step = runbook.steps[0]
    assert step.task_id == task.task_id
    assert step.group is HermesParityBenchmarkGroup.DIRECT_CODEX
    assert step.scorecard_path == (
        f"reports/hermes-baselines/scorecards/{task.task_id}--direct_codex.json"
    )
    assert step.evidence_artifact_path == (
        f"reports/hermes-baselines/evidence/{task.task_id}--direct_codex.md"
    )
    assert task.title in step.operator_prompt
    assert "Do not fabricate" in step.operator_prompt
    assert "run evidence" in step.operator_prompt
    assert "read_workspace" in step.operator_prompt
    assert "Identifies real files and symbols." in step.operator_prompt


def test_baseline_runbook_artifact_writer_persists_markdown_and_json(
    tmp_path,
) -> None:
    from src.skills.external_skill_loader.evaluation import (
        FileHermesParityBaselineStore,
        HermesBaselineRunbookArtifactWriter,
        HermesParityBenchmarkGroup,
        build_default_hermes_parity_tasks,
    )

    task = build_default_hermes_parity_tasks()[0]
    store = FileHermesParityBaselineStore(tmp_path / "parity")
    store.write_manifest((task,))
    runbook = store.build_runbook(
        required_groups=(HermesParityBenchmarkGroup.DIRECT_CODEX,)
    )

    bundle = HermesBaselineRunbookArtifactWriter(root=tmp_path).write(
        runbook=runbook,
    )

    assert bundle.markdown_artifact == "reports/hermes-baseline-runbook.md"
    assert bundle.json_artifact == "reports/hermes-baseline-runbook.json"
    assert "Hermes baseline runbook" in (tmp_path / bundle.markdown_artifact).read_text(
        encoding="utf-8"
    )
    assert task.task_id in (tmp_path / bundle.json_artifact).read_text(encoding="utf-8")


def test_baseline_intake_artifact_writer_persists_markdown_and_json(
    tmp_path,
) -> None:
    from src.skills.external_skill_loader.evaluation import (
        FileHermesParityBaselineStore,
        HermesBaselineIntakeArtifactWriter,
        HermesParityBenchmarkGroup,
        build_default_hermes_parity_tasks,
    )

    task = build_default_hermes_parity_tasks()[0]
    store = FileHermesParityBaselineStore(tmp_path / "parity")
    store.write_manifest((task,))
    coverage = store.coverage(
        required_groups=(
            HermesParityBenchmarkGroup.DIRECT_HERMES,
            HermesParityBenchmarkGroup.DIRECT_CODEX,
        )
    )

    bundle = HermesBaselineIntakeArtifactWriter(root=tmp_path).write(
        coverage=coverage,
    )

    assert bundle.markdown_artifact == "reports/hermes-baseline-intake.md"
    assert bundle.json_artifact == "reports/hermes-baseline-intake.json"
    assert "Hermes baseline intake" in (tmp_path / bundle.markdown_artifact).read_text(
        encoding="utf-8"
    )
    assert '"ready": false' in (tmp_path / bundle.json_artifact).read_text(
        encoding="utf-8"
    )


def test_completion_auditor_requires_evidence_and_final_parity_gate(tmp_path) -> None:
    from src.skills.external_skill_loader.evaluation import (
        HermesCompletionAuditor,
        HermesCompletionEvidence,
        HermesCompletionRequirement,
        HermesCompletionRequirementStatus,
    )

    target = tmp_path / "src" / "demo.py"
    target.parent.mkdir(parents=True)
    target.write_text("class Implemented:\n    pass\n", encoding="utf-8")

    report = HermesCompletionAuditor(
        requirements=(
            HermesCompletionRequirement(
                requirement_id="demo.implemented",
                title="Implemented demo requirement",
                evidence=(
                    HermesCompletionEvidence(
                        path="src/demo.py",
                        contains=("class Implemented",),
                    ),
                ),
            ),
            HermesCompletionRequirement(
                requirement_id="stage_l.real_baselines",
                title="Final parity baselines",
                evidence=(),
                requires_parity_gate_ready=True,
            ),
        )
    ).audit(root=tmp_path)

    assert report.ready is False
    assert report.items[0].status is HermesCompletionRequirementStatus.PROVEN
    assert report.items[1].status is HermesCompletionRequirementStatus.BLOCKED
    assert "Hermes parity gate decision is missing" in report.items[1].blockers
    assert "stage_l.real_baselines" in report.render_markdown()


def test_completion_auditor_reports_weak_and_missing_evidence(tmp_path) -> None:
    from src.skills.external_skill_loader.evaluation import (
        HermesCompletionAuditor,
        HermesCompletionEvidence,
        HermesCompletionRequirement,
        HermesCompletionRequirementStatus,
    )

    existing = tmp_path / "src" / "demo.py"
    existing.parent.mkdir(parents=True)
    existing.write_text("class Present:\n    pass\n", encoding="utf-8")

    report = HermesCompletionAuditor(
        requirements=(
            HermesCompletionRequirement(
                requirement_id="weak",
                title="Weak evidence",
                evidence=(
                    HermesCompletionEvidence(
                        path="src/demo.py",
                        contains=("class Missing",),
                    ),
                ),
            ),
            HermesCompletionRequirement(
                requirement_id="missing",
                title="Missing evidence",
                evidence=(HermesCompletionEvidence(path="src/missing.py"),),
            ),
        )
    ).audit(root=tmp_path)

    assert [item.status for item in report.items] == [
        HermesCompletionRequirementStatus.WEAK,
        HermesCompletionRequirementStatus.MISSING,
    ]
    assert "missing expected content" in report.items[0].blockers[0]
    assert "missing evidence path" in report.items[1].blockers[0]


def test_default_completion_audit_keeps_goal_blocked_without_real_baselines() -> None:
    from pathlib import Path

    from src.skills.external_skill_loader.evaluation import (
        EvaluationHarness,
        HermesCompletionAuditor,
        HermesCompletionRequirementStatus,
        HermesParityGate,
        build_default_hermes_completion_requirements,
        build_default_hermes_parity_tasks,
    )

    project_root = Path(__file__).resolve().parents[3]
    tasks = build_default_hermes_parity_tasks()
    parity_report = EvaluationHarness(tasks=tasks).evaluate(results=())
    gate_decision = HermesParityGate().evaluate(parity_report)

    report = HermesCompletionAuditor(
        requirements=build_default_hermes_completion_requirements()
    ).audit(root=project_root, parity_gate_decision=gate_decision)

    assert report.ready is False
    final_item = report.require("stage_l.parity_gate")
    assert final_item.status is HermesCompletionRequirementStatus.BLOCKED
    assert any("missing direct_hermes baseline" in item for item in final_item.blockers)
    rendered = report.render_markdown()
    assert "NOT READY" in rendered
    assert "## Evidence Scope" in rendered
    assert "code/harness evidence audit" in rendered
    assert "not live runtime readiness proof" in rendered
    assert "## Proven Code/Harness Items" in rendered
    assert "## Still Blocked" in rendered
    assert "## Not Proven" in rendered
    assert "stage_l.parity_gate" in rendered


def test_completion_artifact_writer_persists_markdown_and_json(tmp_path) -> None:
    from src.skills.external_skill_loader.evaluation import (
        HermesCompletionArtifactWriter,
        HermesCompletionAuditor,
        HermesCompletionEvidence,
        HermesCompletionRequirement,
    )

    target = tmp_path / "src" / "demo.py"
    target.parent.mkdir(parents=True)
    target.write_text("ok = True\n", encoding="utf-8")
    report = HermesCompletionAuditor(
        requirements=(
            HermesCompletionRequirement(
                requirement_id="demo",
                title="Demo",
                evidence=(
                    HermesCompletionEvidence(path="src/demo.py", contains=("ok",)),
                ),
            ),
        )
    ).audit(root=tmp_path)

    bundle = HermesCompletionArtifactWriter(root=tmp_path).write(report=report)

    assert bundle.markdown_artifact == "reports/hermes-completion-audit.md"
    assert bundle.json_artifact == "reports/hermes-completion-audit.json"
    assert "Hermes completion audit" in (tmp_path / bundle.markdown_artifact).read_text(
        encoding="utf-8"
    )
    assert '"requirement_id": "demo"' in (tmp_path / bundle.json_artifact).read_text(
        encoding="utf-8"
    )
