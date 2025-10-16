# ---------------------------------- #
# ------------ Splitter ------------ #
# ---------------------------------- #

# ---- Base Exception ---- #


class SplitterException(Exception):
    """Base exception for splitter-related errors."""


# ---- General exceptions ---- #


class InvalidChunkException(SplitterException):
    """Raised when chunks cannot be constructed correctly."""


class SplitterOutputException(SplitterException):
    """Raised when SplitterOutput cannot be built or validated."""


# ---- CodeSplitter ---- #


class UnsupportedCodeLanguage(Exception):
    """Raised when the requested code language is not supported by the splitter."""
