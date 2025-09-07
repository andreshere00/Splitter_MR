import importlib
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from .docling_utils import DoclingPipelineFactory  # noqa: F401
    from .html_to_markdown import HtmlToMarkdown  # noqa: F401
    from .pdfplumber_reader import PDFPlumberReader  # noqa: F401

REGISTRY: Dict[str, Tuple[str, str]] = {
    "HtmlToMarkdown": (".html_to_markdown", "HtmlToMarkdown"),
    "DoclingPipelineFactory": (".docling_utils", "DoclingPipelineFactory"),
    "PDFPlumberReader": (".pdfplumber_reader", "PDFPlumberReader"),
}

__all__ = list(REGISTRY.keys())


def __getattr__(name: str) -> Any:
    try:
        module_path, class_name = REGISTRY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, class_name)
    except ModuleNotFoundError as e:
        if name == "DoclingPipelineFactory":
            raise ModuleNotFoundError(
                "DoclingPipelineFactory requires the 'docling' extra.\n"
                "Install with: pip install 'splitter-mr[docling]'"
            ) from e
        raise


def __dir__():
    return sorted(__all__)
