from typing import TYPE_CHECKING, Any

from .base_model import BaseVisionModel

if TYPE_CHECKING:
    from .models import (
        AnthropicVisionModel,
        AzureOpenAIVisionModel,
        GeminiVisionModel,
        GrokVisionModel,
        HuggingFaceVisionModel,
        OpenAIVisionModel,
    )

__all__ = [
    "BaseVisionModel",
    "AzureOpenAIVisionModel",
    "OpenAIVisionModel",
    "HuggingFaceVisionModel",
    "GrokVisionModel",
    "GeminiVisionModel",
    "AnthropicVisionModel",
]


def __getattr__(name: str) -> Any:
    if name in __all__ and name != "BaseVisionModel":
        from . import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
