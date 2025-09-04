import importlib
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from .anthropic_model import AnthropicVisionModel  # noqa: F401
    from .azure_openai_model import AzureOpenAIVisionModel  # noqa: F401
    from .gemini_model import GeminiVisionModel  # noqa: F401
    from .grok_model import GrokVisionModel  # noqa: F401
    from .huggingface_model import HuggingFaceVisionModel  # noqa: F401
    from .openai_model import OpenAIVisionModel  # noqa: F401

REGISTRY: Dict[str, Tuple[str, str]] = {
    "AzureOpenAIVisionModel": (".azure_openai_model", "AzureOpenAIVisionModel"),
    "OpenAIVisionModel": (".openai_model", "OpenAIVisionModel"),
    "HuggingFaceVisionModel": (".huggingface_model", "HuggingFaceVisionModel"),
    "GrokVisionModel": (".grok_model", "GrokVisionModel"),
    "GeminiVisionModel": (".gemini_model", "GeminiVisionModel"),
    "AnthropicVisionModel": (".anthropic_model", "AnthropicVisionModel"),
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
