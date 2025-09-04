from typing import TYPE_CHECKING, Any

from .base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from .embeddings import (
        AnthropicEmbedding,
        AzureOpenAIEmbedding,
        GeminiEmbedding,
        HuggingFaceEmbedding,
        OpenAIEmbedding,
    )

__all__ = [
    "BaseEmbedding",
    "AzureOpenAIEmbedding",
    "OpenAIEmbedding",
    "HuggingFaceEmbedding",
    "GeminiEmbedding",
    "AnthropicEmbedding",
]


def __getattr__(name: str) -> Any:
    if name in __all__ and name != "BaseEmbedding":
        from . import embeddings

        return getattr(embeddings, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
