class SplitterException(Exception):
    """Base exception for splitter-related errors."""


class InvalidChunkException(SplitterException):
    """Raised when chunks cannot be constructed correctly."""


class SplitterOutputException(SplitterException):
    """Raised when SplitterOutput cannot be built or validated."""
