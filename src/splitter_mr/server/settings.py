"""Typed runtime settings for the SplitterMR MCP and REST server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """Environment-backed configuration for the SplitterMR server.

    Values are read from ``SPLITTER_MR_*`` environment variables. A ``.env`` file
    is not loaded automatically so production deployments stay explicit.

    Attributes:
        host: Bind address used by the CLI. Defaults to loopback.
        port: TCP port used by the CLI and Docker image.
        mcp_path: Mount path for the Streamable HTTP MCP application.
        api_prefix: Prefix for versioned REST endpoints.
        allowed_root: Optional filesystem root that local file sources must stay
            inside after symlink resolution. File sources are disabled when unset.
        allow_urls: Whether URL sources are accepted.
        allowed_url_hosts: Optional hostname allowlist. Empty means any public
            host is allowed when ``allow_urls`` is true.
        max_body_bytes: Maximum JSON body or inlined source size in bytes.
        max_url_redirects: Maximum number of validated HTTP redirects when
            fetching a URL source.
        log_level: Uvicorn log level.

    Raises:
        ValidationError: If a setting is outside the supported range or type.
    """

    model_config = SettingsConfigDict(
        env_prefix="SPLITTER_MR_",
        extra="forbid",
        populate_by_name=True,
    )

    host: str = Field(
        default="127.0.0.1",
        description="Bind address for the HTTP server.",
        examples=["127.0.0.1"],
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="TCP port for the HTTP server.",
        examples=[8000],
    )
    mcp_path: str = Field(
        default="/mcp",
        description="ASGI mount path for the Streamable HTTP MCP server.",
        examples=["/mcp"],
    )
    api_prefix: str = Field(
        default="/api/v1",
        description="Prefix for versioned REST endpoints.",
        examples=["/api/v1"],
    )
    allowed_root: Path | None = Field(
        default=None,
        description=(
            "Filesystem root that local file paths must resolve inside. "
            "File sources are rejected when this is unset."
        ),
        examples=["/data"],
    )
    allow_urls: bool = Field(
        default=False,
        description="Enable HTTP(S) URL sources. Disabled by default.",
        examples=[False],
    )
    allowed_url_hosts: list[str] = Field(
        default_factory=list,
        description=(
            "Optional hostname allowlist for URL sources. Comma-separated or "
            "JSON list in the environment. Empty allows any public host."
        ),
        examples=[["example.com"]],
    )
    max_body_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1,
        description="Maximum request body or inlined source size in bytes.",
        examples=[10485760],
    )
    max_url_redirects: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Maximum number of validated redirects for URL fetches.",
        examples=[5],
    )
    log_level: Literal["critical", "error", "warning", "info", "debug", "trace"] = (
        Field(
            default="info",
            description="Uvicorn log level.",
            examples=["info"],
        )
    )

    @field_validator("mcp_path", "api_prefix")
    @classmethod
    def validate_path_prefix(cls, value: str) -> str:
        """Require path prefixes to start with a single leading slash.

        Args:
            value: Raw path prefix.

        Returns:
            The stripped path prefix.

        Raises:
            ValueError: If the path is empty or does not start with ``/``.
        """
        stripped = value.strip()
        if not stripped.startswith("/"):
            raise ValueError("path prefixes must start with '/'")
        if stripped != "/" and stripped.endswith("/"):
            return stripped.rstrip("/")
        return stripped

    @field_validator("allowed_url_hosts", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, value: object) -> object:
        """Parse comma-separated or JSON host lists from the environment.

        Args:
            value: Raw environment value or already-parsed list.

        Returns:
            A list of hostname strings.

        Raises:
            ValueError: If a JSON string is not a list of strings.
        """
        if value is None or value == "":
            return []
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                parsed = json.loads(stripped)
                if not isinstance(parsed, list) or any(
                    not isinstance(item, str) for item in parsed
                ):
                    raise ValueError("allowed_url_hosts JSON must be a list of strings")
                return [item.strip().lower() for item in parsed if item.strip()]
            return [
                item.strip().lower() for item in stripped.split(",") if item.strip()
            ]
        if isinstance(value, list):
            return [str(item).strip().lower() for item in value if str(item).strip()]
        return value
