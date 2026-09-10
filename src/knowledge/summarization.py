"""Optional LLM summarization layer for retrieval context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.llm.protocols import LLMRequest

if TYPE_CHECKING:
    from src.llm.protocols import LLMGatewayProtocol


@dataclass(frozen=True)
class RetrievalChunk:
    id: int
    title: str
    text: str
    score: float = 0.0


@dataclass(frozen=True)
class SummarizedRetrievalContext:
    text: str
    source_ids: list[int] = field(default_factory=list)
    original_count: int = 0


async def summarize_retrieval_chunks(
    chunks: list[RetrievalChunk],
    *,
    llm_router: LLMGatewayProtocol,
    top_k: int = 50,
) -> SummarizedRetrievalContext:
    selected = sorted(chunks, key=lambda chunk: chunk.score, reverse=True)[:top_k]
    if not selected:
        return SummarizedRetrievalContext(text="", source_ids=[], original_count=0)
    prompt = "\n\n".join(
        f"[#{chunk.id}] {chunk.title}\n{chunk.text[:2000]}" for chunk in selected
    )
    response = await llm_router.generate(
        LLMRequest(
            prompt=prompt,
            system=(
                "Сожми найденные фрагменты в плотную русскую сводку. "
                "Сохрани факты и ссылки на номера источников вида [#id]."
            ),
            tier="worker",
            temperature=0.0,
            caller="knowledge_retrieval_summarization",
        )
    )
    return SummarizedRetrievalContext(
        text=response.text.strip(),
        source_ids=[chunk.id for chunk in selected],
        original_count=len(chunks),
    )
