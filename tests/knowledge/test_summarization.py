"""Optional retrieval summarization layer."""

from __future__ import annotations

from unittest.mock import AsyncMock

from src.llm.protocols import LLMResponse, LLMUsage


async def test_retrieval_summarization_compresses_top_k_into_single_summary() -> None:
    from src.knowledge.summarization import RetrievalChunk, summarize_retrieval_chunks

    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value=LLMResponse(
            text="Плотная сводка.", model="worker", usage=LLMUsage()
        )
    )
    chunks = [
        RetrievalChunk(id=i, title=f"Title {i}", text="content " * 100, score=1.0 / i)
        for i in range(1, 6)
    ]

    summary = await summarize_retrieval_chunks(chunks, llm_router=llm, top_k=3)

    assert summary.text == "Плотная сводка."
    assert summary.source_ids == [1, 2, 3]
    request = llm.generate.call_args.args[0]
    assert request.tier == "worker"
    assert request.caller == "knowledge_retrieval_summarization"
