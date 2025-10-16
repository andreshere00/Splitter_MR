from __future__ import annotations

import json
import warnings
from typing import List

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from ...schema import ReaderOutput, SplitterOutput
from ...schema.exceptions import (
    InvalidChunkException,
    SplitterOutputException,
    UnsupportedCodeLanguage,
)
from ...schema.warnings import SplitterInputWarning
from ..base_splitter import BaseSplitter


def get_langchain_language(lang_str: str) -> Language:
    """Resolve a string name to a LangChain ``Language`` enum.

    Args:
        lang_str (str): Case-insensitive programming language name
            (e.g., ``"python"``, ``"java"``, ``"kotlin"``).

    Returns:
        Language: The corresponding LangChain language enumeration.

    Raises:
        UnsupportedCodeLanguage: If the provided language is not supported
            by the LangChain ``Language`` enum.
    """
    lookup = {lang.name.lower(): lang for lang in Language}
    key = (lang_str or "").lower()
    if key not in lookup:
        supported = ", ".join(sorted(lookup.keys()))
        raise UnsupportedCodeLanguage(
            f"Unsupported language '{lang_str}'. Supported languages: {supported}"
        )
    return lookup[key]


class CodeSplitter(BaseSplitter):
    """Recursively splits source code into language-aware, semantically meaningful chunks.

    The ``CodeSplitter`` uses LangChain's
    :func:`RecursiveCharacterTextSplitter.from_language` method to generate
    code chunks that align with syntactic boundaries such as functions,
    methods, and classes. This allows for better context preservation during
    code analysis, summarization, or embedding.

    Attributes:
        language (str): Programming language to split (e.g., ``"python"`` or ``"java"``).
        chunk_size (int): Maximum number of characters per chunk.

    Warnings:
        SplitterInputWarning: Emitted when the input text is empty or whitespace-only,
            or when ``conversion_method='json'`` but the text is invalid JSON.

    Raises:
        UnsupportedCodeLanguage: If the requested language is not supported by LangChain.
        InvalidChunkException: If chunk generation fails or produces invalid chunks.
        SplitterOutputException: If the final :class:`SplitterOutput` cannot be built
            or validated.
    """

    def __init__(self, chunk_size: int = 1000, language: str = "python"):
        """Initialize a ``CodeSplitter`` instance.

        Args:
            chunk_size (int, optional): Maximum number of characters per chunk.
                Must be an integer >= 1. Defaults to 1000.
            language (str, optional): Programming language name (case-insensitive),
                e.g., ``"python"`` or ``"java"``. Defaults to ``"python"``.

        Raises:
            ValueError: If ``chunk_size`` is not a positive integer.
        """
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be an integer >= 1")
        super().__init__(chunk_size)
        self.language = language

    def _warn_on_input(self, reader_output: ReaderOutput, text: str) -> None:
        """Emit :class:`SplitterInputWarning` for suspicious or malformed inputs.

        This helper checks for two common problems:

        * Empty or whitespace-only text → emits a warning and continues.
        * Declared JSON input that cannot be parsed → emits a warning and treats
          it as plain text.

        Args:
            reader_output (ReaderOutput): Input object containing text and metadata.
            text (str): Text content to analyze and possibly warn about.

        Warnings:
            SplitterInputWarning: If text is empty or invalid JSON (when declared).
        """
        if (text or "").strip() == "":
            warnings.warn(
                SplitterInputWarning(
                    "ReaderOutput.text is empty or whitespace-only. "
                    "Proceeding; this will yield a single empty chunk."
                )
            )

        if (reader_output.conversion_method or "").lower() == "json":
            try:
                json.loads(text or "")
            except Exception:
                warnings.warn(
                    SplitterInputWarning(
                        "ReaderOutput.conversion_method is 'json' but text "
                        "is not valid JSON. Proceeding as plain text."
                    )
                )

    def _build_chunks(
        self, text: str, lang_enum: Language, chunk_size: int
    ) -> List[str]:
        """Build and validate code chunks using LangChain.

        Args:
            text (str): Source code to split.
            lang_enum (Language): LangChain language enumeration.
            chunk_size (int): Maximum characters per chunk.

        Returns:
            List[str]: List of chunked code strings.

        Raises:
            InvalidChunkException: If no chunks are produced, a chunk is ``None``,
                or all chunks are empty for non-empty text.
        """
        try:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang_enum, chunk_size=chunk_size, chunk_overlap=0
            )
            docs = splitter.create_documents([text or ""])
            chunks = [doc.page_content for doc in docs]

            # Guarantee at least one empty chunk if text is empty
            if len((text or "")) == 0:
                chunks = [""]

            # Sanity checks
            if not isinstance(chunks, list) or len(chunks) == 0:
                raise InvalidChunkException("No chunks were produced.")
            if any(c is None for c in chunks):
                raise InvalidChunkException("A produced chunk is None.")
            if len(text or "") > 0 and all(c == "" for c in chunks):
                raise InvalidChunkException(
                    "All produced chunks are empty for non-empty text."
                )

            return chunks

        except InvalidChunkException:
            raise
        except Exception as e:
            raise InvalidChunkException(
                f"Unexpected error while building code chunks: {e}"
            ) from e

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """Split the provided source code into language-aware chunks.

        The method performs input validation and warning emission, determines
        the appropriate language enum, builds code chunks via LangChain,
        and returns a fully validated :class:`SplitterOutput` instance.

        Args:
            reader_output (ReaderOutput): A validated input object containing
                at least a ``text`` field and optional document metadata.

        Returns:
            SplitterOutput: Structured splitter output containing:
                * ``chunks`` — list of split code segments.
                * ``chunk_id`` — corresponding unique identifiers.
                * document metadata and splitter parameters.

        Raises:
            UnsupportedCodeLanguage: If ``self.language`` is not recognized.
            InvalidChunkException: If chunk construction fails or yields invalid chunks.
            SplitterOutputException: If the :class:`SplitterOutput` cannot be built
                or validated.

        Warnings:
            SplitterInputWarning: If text is empty, whitespace-only, or invalid JSON.

        Example:
            ```python
            from splitter_mr.splitter import CodeSplitter
            from splitter_mr.schema.models import ReaderOutput

            reader_output = ReaderOutput(
                text="def foo():\\n    pass\\n\\nclass Bar:\\n    def baz(self):\\n        pass",
                document_name="example.py",
                document_path="/tmp/example.py",
            )

            splitter = CodeSplitter(chunk_size=50, language="python")
            output = splitter.split(reader_output)
            print(output.chunks)
            # ['def foo():\\n    pass\\n', 'class Bar:\\n    def baz(self):\\n        pass']
            ```
        """
        text = reader_output.text or ""
        chunk_size = self.chunk_size

        # Check input
        self._warn_on_input(reader_output, text)

        # Resolve language
        lang_enum = get_langchain_language(self.language)

        # Build chunks
        chunks = self._build_chunks(text, lang_enum, chunk_size)

        # Produce output
        try:
            chunk_ids = self._generate_chunk_ids(len(chunks))
            metadata = self._default_metadata()
            output = SplitterOutput(
                chunks=chunks,
                chunk_id=chunk_ids,
                document_name=reader_output.document_name,
                document_path=reader_output.document_path or "",
                document_id=reader_output.document_id,
                conversion_method=reader_output.conversion_method,
                reader_method=reader_output.reader_method,
                ocr_method=reader_output.ocr_method,
                split_method="code_splitter",
                split_params={"chunk_size": chunk_size, "language": self.language},
                metadata=metadata,
            )
            return output
        except Exception as e:
            raise SplitterOutputException(f"Failed to build SplitterOutput: {e}") from e
