"""Tests for System 2 active memory retrieval and depth classification."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.core.decision import DecisionEngine, RetrievalResult
from src.core.file_access import FileAccessService
from src.llm.protocols import LLMResponse, LLMUsage

if TYPE_CHECKING:
    from pathlib import Path


def _llm_resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="sonnet", usage=LLMUsage())


@pytest.fixture
def mock_llm_router() -> AsyncMock:
    router = AsyncMock()
    router.generate = AsyncMock(return_value=_llm_resp("A"))
    return router


@pytest.fixture
def mock_personality() -> MagicMock:
    p = MagicMock()
    p.get_personality_tree_summary = MagicMock(return_value="I am Zhvusha.")
    return p


@pytest.fixture
def engine(
    mock_episodic: AsyncMock,
    mock_personality: MagicMock,
    mock_llm_router: AsyncMock,
) -> DecisionEngine:
    return DecisionEngine(
        episodic=mock_episodic,
        personality=mock_personality,
        llm_router=mock_llm_router,
    )


async def test_classifies_simple_question_as_quick(engine: DecisionEngine):
    """Simple questions get classified as QUICK."""
    engine.llm.generate = AsyncMock(return_value=_llm_resp("A"))
    result = await engine.classify_depth("как дела?")
    assert result.depth == "QUICK"


async def test_classifies_knowledge_question_as_memory(engine: DecisionEngine):
    """Knowledge questions get classified as MEMORY."""
    engine.llm.generate = AsyncMock(return_value=_llm_resp("B"))
    result = await engine.classify_depth("что ты знаешь про aiogram?")
    assert result.depth == "MEMORY"


async def test_classifies_news_question_as_research(engine: DecisionEngine):
    """News/current questions get classified as RESEARCH."""
    engine.llm.generate = AsyncMock(return_value=_llm_resp("C"))
    result = await engine.classify_depth("какие сейчас актуальные AI фреймворки?")
    assert result.depth == "RESEARCH"


async def test_classifies_ambiguous_as_unclear(engine: DecisionEngine):
    """Ambiguous questions get classified as UNCLEAR."""
    engine.llm.generate = AsyncMock(return_value=_llm_resp("D"))
    result = await engine.classify_depth("расскажи про это")
    assert result.depth == "UNCLEAR"


async def test_active_retrieval_executes_planned_queries(engine: DecisionEngine):
    """Multi-step retrieval runs queries planned by the analyst tier."""
    # Analyst tier returns a plan with two queries.
    plan = json.dumps(
        {
            "queries": [
                {"query": "aiogram features", "source_filter": ["youtube", "channel"]},
                {"query": "aiogram patterns", "source_filter": ["chat"]},
            ]
        }
    )

    engine.llm.generate = AsyncMock(return_value=_llm_resp(plan))

    # Mock episodes returned by retrieve
    ep1 = SimpleNamespace(
        id=1, content="aiogram 3 middleware chaining", metadata_json=None
    )
    ep2 = SimpleNamespace(
        id=2,
        content="aiogram patterns in Zhvusha",
        metadata_json=json.dumps({"file_path": "youtube/aiogram.md"}),
    )
    engine.episodic.retrieve = AsyncMock(side_effect=[[ep1], [ep2]])

    context = await engine._active_retrieval("что знаешь про aiogram?")

    assert "middleware chaining" in context
    assert "Knowledge: youtube/aiogram.md" in context
    assert engine.episodic.retrieve.await_count == 2


async def test_source_filter_passed_to_retrieve(engine: DecisionEngine):
    """source_filter from planned queries is passed to episodic.retrieve()."""
    plan = json.dumps(
        {
            "queries": [
                {"query": "test", "source_filter": ["youtube", "browser"]},
            ]
        }
    )
    engine.llm.generate = AsyncMock(return_value=_llm_resp(plan))
    engine.episodic.retrieve = AsyncMock(return_value=[])

    await engine._active_retrieval("test query")

    call_kwargs = engine.episodic.retrieve.call_args.kwargs
    assert call_kwargs["source_filter"] == ["youtube", "browser"]


# --- retrieve_for_question tests (012) ---


def _make_engine_with_fa(
    *,
    llm_response: str = "",
    episodic: AsyncMock | None = None,
    file_access: FileAccessService | MagicMock | None = None,
) -> DecisionEngine:
    ep = episodic or AsyncMock()
    ep.retrieve = getattr(ep, "retrieve", AsyncMock(return_value=[]))
    ep.retrieve_by_somatic_marker = getattr(
        ep, "retrieve_by_somatic_marker", AsyncMock(return_value=[])
    )
    personality = MagicMock()
    personality.get_personality_tree_summary = MagicMock(return_value="I am Zhvusha.")
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=_llm_resp(llm_response))
    return DecisionEngine(ep, personality, llm, file_access=file_access)


async def test_retrieve_quick_with_response() -> None:
    """QUICK plan returns quick_response directly."""
    plan_json = json.dumps(
        {
            "depth": "QUICK",
            "response": "привет!",
            "memory_queries": [],
            "workspace_files": [],
            "code_files": [],
        }
    )
    fa = MagicMock(spec=FileAccessService)
    fa.get_workspace_index = MagicMock(return_value="diary/")
    fa.get_project_index = MagicMock(return_value="src/")

    engine = _make_engine_with_fa(llm_response=plan_json, file_access=fa)
    result = await engine.retrieve_for_question("привет")

    assert result.depth == "QUICK"
    assert result.quick_response == "привет!"
    assert result.file_context == ""


async def test_retrieve_memory_with_files(tmp_path: Path) -> None:
    """MEMORY plan triggers file reads and returns FILE_CONTENT tags."""
    ws = tmp_path / "ws"
    (ws / "diary").mkdir(parents=True)
    (ws / "diary" / "2026-04-02.md").write_text("Good day", encoding="utf-8")
    proj = tmp_path / "proj"
    proj.mkdir()

    fa = FileAccessService(ws, proj)
    plan_json = json.dumps(
        {
            "depth": "MEMORY",
            "response": "",
            "memory_queries": [],
            "workspace_files": ["diary/2026-04-02.md"],
            "code_files": [],
        }
    )
    engine = _make_engine_with_fa(llm_response=plan_json, file_access=fa)
    result = await engine.retrieve_for_question("что в дневнике?")

    assert result.depth == "MEMORY"
    assert "FILE_CONTENT" in result.file_context
    assert "Good day" in result.file_context


async def test_retrieve_memory_with_queries() -> None:
    """MEMORY plan with memory_queries calls episodic.retrieve."""
    plan_json = json.dumps(
        {
            "depth": "MEMORY",
            "response": "",
            "memory_queries": ["aiogram", "telegram"],
            "workspace_files": [],
            "code_files": [],
        }
    )
    fa = MagicMock(spec=FileAccessService)
    fa.get_workspace_index = MagicMock(return_value="")
    fa.get_project_index = MagicMock(return_value="")

    ep1 = SimpleNamespace(id=1, content="aiogram best practices")
    ep2 = SimpleNamespace(id=2, content="telegram channel setup")
    episodic = AsyncMock()
    episodic.retrieve = AsyncMock(side_effect=[[ep1], [ep2]])

    engine = _make_engine_with_fa(
        llm_response=plan_json, episodic=episodic, file_access=fa
    )
    result = await engine.retrieve_for_question("расскажи про aiogram")

    assert "aiogram best practices" in result.memory_context
    assert "telegram channel setup" in result.memory_context
    assert episodic.retrieve.await_count == 2


async def test_retrieve_deduplicates_episodes() -> None:
    """Duplicate episodes (same id) are included only once."""
    plan_json = json.dumps(
        {
            "depth": "MEMORY",
            "response": "",
            "memory_queries": ["q1", "q2"],
            "workspace_files": [],
            "code_files": [],
        }
    )
    fa = MagicMock(spec=FileAccessService)
    fa.get_workspace_index = MagicMock(return_value="")
    fa.get_project_index = MagicMock(return_value="")

    ep = SimpleNamespace(id=42, content="unique content")
    episodic = AsyncMock()
    episodic.retrieve = AsyncMock(return_value=[ep])

    engine = _make_engine_with_fa(
        llm_response=plan_json, episodic=episodic, file_access=fa
    )
    result = await engine.retrieve_for_question("test")

    assert result.memory_context.count("unique content") == 1


async def test_retrieve_invalid_json_fallback() -> None:
    """Invalid LLM JSON output falls back to QUICK."""
    fa = MagicMock(spec=FileAccessService)
    fa.get_workspace_index = MagicMock(return_value="")
    fa.get_project_index = MagicMock(return_value="")

    engine = _make_engine_with_fa(llm_response="not valid json {{{", file_access=fa)
    result = await engine.retrieve_for_question("test")

    assert result.depth == "QUICK"
    assert result.quick_response == ""


async def test_retrieve_markdown_wrapped_json() -> None:
    """JSON wrapped in ```json...``` is extracted correctly."""
    inner = json.dumps(
        {
            "depth": "QUICK",
            "response": "markdown ok",
            "memory_queries": [],
            "workspace_files": [],
            "code_files": [],
        }
    )
    wrapped = f"```json\n{inner}\n```"
    fa = MagicMock(spec=FileAccessService)
    fa.get_workspace_index = MagicMock(return_value="")
    fa.get_project_index = MagicMock(return_value="")

    engine = _make_engine_with_fa(llm_response=wrapped, file_access=fa)
    result = await engine.retrieve_for_question("test")

    assert result.quick_response == "markdown ok"


async def test_retrieve_no_file_access_returns_empty() -> None:
    """Without file_access, returns empty RetrievalResult."""
    engine = _make_engine_with_fa(file_access=None)
    result = await engine.retrieve_for_question("test")

    assert result == RetrievalResult()


async def test_plan_includes_file_index() -> None:
    """_plan_and_classify includes file index in the prompt."""
    fa = MagicMock(spec=FileAccessService)
    fa.get_workspace_index = MagicMock(return_value="diary/\npersonality/")
    fa.get_project_index = MagicMock(return_value="src/main.py")

    plan_json = json.dumps({"depth": "QUICK", "response": "ok"})
    engine = _make_engine_with_fa(llm_response=plan_json, file_access=fa)
    await engine.retrieve_for_question("test")

    prompt = engine.llm.generate.call_args.args[0].prompt
    assert "diary/" in prompt
    assert "personality/" in prompt
    assert "src/main.py" in prompt


async def test_plan_system_prompt_includes_personality_anchor() -> None:
    """Planner QUICK responses are user-facing, so they inherit the anchor."""
    fa = MagicMock(spec=FileAccessService)
    fa.get_workspace_index = MagicMock(return_value="")
    fa.get_project_index = MagicMock(return_value="")

    plan_json = json.dumps({"depth": "QUICK", "response": "окей"})
    engine = _make_engine_with_fa(llm_response=plan_json, file_access=fa)
    await engine.retrieve_for_question("привет")

    request = engine.llm.generate.call_args.args[0]
    assert "Непереписываемая личность" in request.system
