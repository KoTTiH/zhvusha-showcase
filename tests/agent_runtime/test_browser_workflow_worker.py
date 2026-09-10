"""Browser workflow draft worker tests."""

from __future__ import annotations

import json


async def test_browser_workflow_worker_reads_form_and_creates_draft(tmp_path) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import AgentJob, ContextPack
    from src.agent_runtime.profiles import BROWSER_WORKFLOW_DRAFT
    from src.agent_runtime.workers.browser_workflow import (
        BrowserWorkflowDraftWorkerBackend,
    )

    async def fetch(url: str) -> str:
        assert url == "https://example.com/forms/post"
        return """
        <form method="post" action="/post">
          <input name="custname">
          <input name="custemail" type="email">
          <textarea name="comments"></textarea>
          <button type="submit">Submit order</button>
        </form>
        """

    gateway = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        web_fetcher=fetch,
    )
    worker = BrowserWorkflowDraftWorkerBackend(tool_gateway=gateway)
    context_pack = ContextPack(
        user_request="prepare draft",
        metadata={
            "browser_workflow_payload": json.dumps(
                {
                    "read_url": "https://example.com/forms/post",
                    "fields": {
                        "custname": "Stage L Draft",
                        "custemail": "stage-l@example.invalid",
                        "comments": "Draft only.",
                    },
                    "purpose": "contract test",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        },
    )
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="vscode:browser",
        fingerprint="browser-workflow",
        kind="browser_workflow_draft",
        profile=BROWSER_WORKFLOW_DRAFT,
        context_pack=context_pack,
    )

    capsule = await worker.run(job=job, context_pack=context_pack)

    assert capsule.sources == ("https://example.com/forms/post",)
    assert len(capsule.artifacts) == 1
    artifact = tmp_path / capsule.artifacts[0]
    draft = json.loads(artifact.read_text(encoding="utf-8"))
    assert draft["action_url"] == "https://example.com/post"
    assert draft["method"] == "POST"
    assert draft["fields"]["custname"] == "Stage L Draft"
    assert draft["submit_blocked"] is True
    assert draft["requires_approval_for_submit"] is True
    assert draft["next_required_capability"] == "browser_submit"
    assert "browser_submit was not used" in capsule.processed_context


async def test_browser_workflow_profile_does_not_expose_submit(tmp_path) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.profiles import BROWSER_WORKFLOW_DRAFT

    gateway = build_builtin_tool_gateway(workspace_root=tmp_path)
    toolset = gateway.build_toolset(BROWSER_WORKFLOW_DRAFT)

    assert "browser_read_url" in toolset
    assert "browser_draft_form" in toolset
    assert "browser_submit_form" not in toolset
