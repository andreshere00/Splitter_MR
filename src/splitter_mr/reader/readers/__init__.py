import importlib
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from .docling_reader import DoclingReader  # noqa: F401
    from .markitdown_reader import MarkItDownReader  # noqa: F401
    from .vanilla_reader import VanillaReader  # noqa: F401

REGISTRY: Dict[str, Tuple[str, str]] = {
    "VanillaReader": (".vanilla_reader", "VanillaReader"),
    "MarkItDownReader": (".markitdown_reader", "MarkItDownReader"),
    "DoclingReader": (".docling_reader", "DoclingReader"),
}

__all__ = list(REGISTRY.keys())

# Per-class extra hints for optional deps
EXTRA_BY_NAME: Dict[str, str] = {
    "MarkItDownReader": "markitdown",
    "DoclingReader": "docling",
}


def __getattr__(name: str) -> Any:
    try:
        module_path, class_name = REGISTRY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, class_name)
    except ModuleNotFoundError as e:
        extra = EXTRA_BY_NAME.get(name)
        if extra:
            raise ModuleNotFoundError(
                f"{name} requires the '{extra}' extra. "
                f"Install with: pip install 'splitter-mr[{extra}]'"
            ) from e
        # If there's no declared extra (e.g., VanillaReader), re-raise original
        raise


def __dir__():
    return sorted(__all__)
