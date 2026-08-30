"""Service-boundary exceptions for the SplitterMR MCP and REST server."""


class ServerError(Exception):
    """Base exception for server-side processing failures.

    Args:
        message: Human-readable error description.
        code: Stable machine-readable error code.
    """

    def __init__(self, message: str, *, code: str = "server_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class ServerAccessDeniedError(ServerError):
    """Raised when a file path or URL is blocked by server policy."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="access_denied")


class ServerPayloadTooLargeError(ServerError):
    """Raised when a request or source exceeds the configured size limit."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="payload_too_large")


class ServerComponentUnavailableError(ServerError):
    """Raised when a requested reader or splitter cannot be used."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="component_unavailable")


class ServerConfigurationError(ServerError):
    """Raised when a request is syntactically valid but cannot be executed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="invalid_configuration")
