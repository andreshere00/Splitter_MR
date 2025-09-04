import importlib
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from .anthropic_embedding import AnthropicEmbedding  # noqa: F401
    from .azure_openai_embedding import AzureOpenAIEmbedding  # noqa: F401
    from .gemini_embedding import GeminiEmbedding  # noqa: F401
    from .huggingface_embedding import HuggingFaceEmbedding  # noqa: F401
    from .openai_embedding import OpenAIEmbedding  # noqa: F401

REGISTRY: Dict[str, Tuple[str, str]] = {
    "AzureOpenAIEmbedding": (".azure_openai_embedding", "AzureOpenAIEmbedding"),
    "OpenAIEmbedding": (".openai_embedding", "OpenAIEmbedding"),
    "GeminiEmbedding": (".gemini_embedding", "GeminiEmbedding"),
    "HuggingFaceEmbedding": (".huggingface_embedding", "HuggingFaceEmbedding"),
    "AnthropicEmbedding": (".anthropic_embedding", "AnthropicEmbedding"),
}

__all__ = list(REGISTRY.keys())


def _multimodal_hint() -> str:
    return (
        "This feature requires the 'multimodal' extra.\n"
        "Install it with:\n\n"
        "    pip install 'splitter-mr[multimodal]'\n"
    )


def __getattr__(name: str) -> Any:
    try:
        module_path, class_name = REGISTRY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, class_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(_multimodal_hint()) from e


def __dir__():
    return sorted(__all__)
