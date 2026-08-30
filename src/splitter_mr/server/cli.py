"""Command-line entry point for the SplitterMR MCP server."""

from __future__ import annotations

import argparse

import uvicorn

from .settings import ServerSettings


def build_parser(settings: ServerSettings) -> argparse.ArgumentParser:
    """Create the argument parser with settings-backed defaults.

    Args:
        settings: Environment-backed server settings.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="splitter-mr-mcp",
        description="Run the SplitterMR FastAPI application with a mounted MCP server.",
    )
    parser.add_argument(
        "--host",
        default=settings.host,
        help="Bind address. Defaults to SPLITTER_MR_HOST or 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help="TCP port. Defaults to SPLITTER_MR_PORT or 8000.",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level. Defaults to SPLITTER_MR_LOG_LEVEL or info.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run Uvicorn with the SplitterMR FastAPI application.

    Args:
        argv: Optional argument vector. Defaults to ``sys.argv[1:]``.
    """
    settings = ServerSettings()
    parser = build_parser(settings)
    args = parser.parse_args(argv)
    uvicorn.run(
        "splitter_mr.server.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
