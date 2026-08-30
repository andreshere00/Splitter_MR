# -------------------------------- #
# ------------ Reader ------------ #
# -------------------------------- #

# ---- Base Exception ---- #


class ReaderException(Exception):
    """Base exception for reader-related errors."""

    pass


class ReaderOutputException(ReaderException):
    """Raised when ReaderOutput has not a valid structure."""


# ---- Conversion Exceptions ---- #


class HtmlConversionError(ReaderException):
    """Raised when HTML→Markdown conversion fails."""


# ---- General Exceptions ---- #


class ReaderConfigException(ReaderException, ValueError):
    """
    Raised when invalid parameters are passed to the Reader configuration.
    """


# ---- VanillaReader Exceptions ---- #


class VanillaReaderException(ReaderException, RuntimeError):
    """
    Raised when VanillaReader–based document conversion fails.
    Wraps exceptions coming from vanilla_reader.exceptions.VanillaReaderError.
    """


# ---- MarkItDown Exceptions ---- #


class MarkItDownReaderException(ReaderException, RuntimeError):
    """
    Raised when MarkItDown–based document conversion fails in MarkItDownReader.
    Wraps exceptions coming from markitdown.exceptions.MarkItDownError.
    """


# ---- Docling Exceptions ---- #


class DoclingReaderException(ReaderException, RuntimeError):
    """
    Raised when IBM Docling–based document conversion fails in DoclingReader.
    Wraps exceptions coming from docling.exceptions.BaseError.
    """


# ---- Textract Exceptions ---- #


class TextractReaderException(ReaderException, RuntimeError):
    """
    Raised when AWS Textract–based document conversion fails in TextractReader.
    Wraps boto3/botocore errors and local conversion failures.
    """


# ---------------------------------- #
# ------------ Splitter ------------ #
# ---------------------------------- #

# ---- Base Exception ---- #


class SplitterException(Exception):
    """Base exception for splitter-related errors."""

    pass


# ---- General exceptions ---- #


class InvalidChunkException(SplitterException, ValueError):
    """Raised when chunks cannot be constructed correctly."""


class SplitterConfigException(SplitterException, ValueError):
    """Raised when the configuration provided to the Splitter class is not correct"""


class SplitterOutputException(SplitterException, TypeError):
    """Raised when SplitterOutput cannot be built or validated."""


# ---- HeaderSplitter ---- #


class InvalidHeaderNameError(SplitterConfigException):
    """Raised when a header string isn't of the expected 'Header N' form."""


class HeaderLevelOutOfRangeError(SplitterConfigException):
    """Raised when the parsed header level is outside 1..6."""


class NormalizationError(ReaderException, TypeError):
    """Raised when Setext→ATX normalization can't be safely applied."""


# ---- HTMLTagSplitter ---- #


class InvalidHtmlTagError(ReaderException, ValueError):
    """
    Raised when an invalid HTML Tag is provided or when it is missing
    in the document.
    """
