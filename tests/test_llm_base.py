import pytest
from src.llm.base import BaseLLMAdapter


def test_cannot_instantiate_base():
    with pytest.raises(TypeError):
        BaseLLMAdapter()  # type: ignore[abstract]
