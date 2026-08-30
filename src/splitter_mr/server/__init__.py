"""Optional FastAPI and MCP server for SplitterMR.

Importing ``app``, ``create_app``, or ``create_mcp_server`` requires the ``mcp``
extra:

```bash
pip install "splitter-mr[mcp]"
```
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import app, create_app
    from .mcp import create_mcp_server

__all__ = ["app", "create_app", "create_mcp_server"]

_EXPORTS = {
    "app": (".app", "app"),
    "create_app": (".app", "create_app"),
    "create_mcp_server": (".mcp", "create_mcp_server"),
}

_MISSING_EXTRA_MESSAGE = (
    "The SplitterMR MCP server requires the 'mcp' extra. "
    "Install with: pip install 'splitter-mr[mcp]'"
)


def __getattr__(name: str) -> Any:
    """Lazily export the FastAPI app factory and MCP server.

    Args:
        name: Attribute requested by the caller.

    Returns:
        The requested application object or factory.

    Raises:
        AttributeError: If ``name`` is not a public export.
        ModuleNotFoundError: If FastAPI or FastMCP are not installed.
    """
    try:
        module_path, attr_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error

    try:
        from importlib import import_module

        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(_MISSING_EXTRA_MESSAGE) from error


def __dir__() -> list[str]:
    return sorted(__all__)
