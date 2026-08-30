"""Shared read, split, and read-and-split pipeline for REST and MCP."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from splitter_mr.reader.base_reader import BaseReader
from splitter_mr.schema.exceptions import (
    ReaderConfigException,
    ReaderException,
    SplitterConfigException,
    SplitterException,
)
from splitter_mr.schema.models import ReaderOutput, SplitterOutput

from . import components
from .enums import SourceType
from .exceptions import (
    ServerConfigurationError,
    ServerError,
    ServerPayloadTooLargeError,
)
from .schemas import (
    READER_SOURCE_TYPES,
    ComponentCatalogResponse,
    ReadAndSplitRequest,
    ReadDocumentRequest,
    SplitDocumentRequest,
)
from .security import fetch_url, resolve_allowed_file, validate_url
from .settings import ServerSettings


class PipelineService:
    """Stateless orchestration of SplitterMR readers and splitters.

    Args:
        settings: Active server settings used for access policy and size limits.
    """

    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings

    def list_components(self) -> ComponentCatalogResponse:
        """Return the reader, splitter, and vision-model catalog.

        Returns:
            Typed component catalog.
        """
        return components.list_components()

    async def read_document(self, request: ReadDocumentRequest) -> ReaderOutput:
        """Read one input and return a validated ``ReaderOutput``.

        Args:
            request: Typed read request matching ``BaseReader.read``.

        Returns:
            Reader output produced by the selected reader.

        Raises:
            ServerAccessDeniedError: If the source is blocked by policy.
            ServerPayloadTooLargeError: If the source exceeds the size limit.
            ServerComponentUnavailableError: If a required extra is missing.
            ServerConfigurationError: If the reader rejects the input.
        """
        file_path, temp_path, kind = self._prepare_file_path(request)
        allowed = READER_SOURCE_TYPES[request.reader.reader]
        if kind not in allowed:
            allowed_list = ", ".join(sorted(allowed))
            raise ServerConfigurationError(
                f"{request.reader.reader} does not support {kind} file_path values. "
                f"Allowed kinds: {allowed_list}."
            )
        try:
            vision_model = None
            if request.model is not None:
                vision_model = components.create_vision_model(request.model)
            reader = components.create_reader(request.reader, vision_model)
            read_kwargs = {
                **components.reader_read_kwargs(request.reader),
                **request.kwargs,
            }
            return await asyncio.to_thread(
                self._call_read, reader.read, file_path, read_kwargs
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    async def split_document(self, request: SplitDocumentRequest) -> SplitterOutput:
        """Split a ``ReaderOutput`` and return a validated ``SplitterOutput``.

        Args:
            request: Typed split request.

        Returns:
            Splitter output. Reader metadata is preserved by the splitter.

        Raises:
            ServerComponentUnavailableError: If the splitter is not supported.
            ServerConfigurationError: If splitting fails configuration checks.
        """
        embedding = None
        if request.embedding is not None:
            embedding = components.create_embedding(request.embedding)
        splitter = components.create_splitter(
            request.splitter,
            extra_kwargs=request.kwargs,
            embedding=embedding,
        )
        return await asyncio.to_thread(
            self._call_split, splitter.split, request.reader_output
        )

    async def read_and_split(self, request: ReadAndSplitRequest) -> SplitterOutput:
        """Read an input and immediately split the resulting ``ReaderOutput``.

        Args:
            request: Typed composite request.

        Returns:
            Splitter output produced from the exact reader output.

        Raises:
            ServerError: If either stage fails.
        """
        reader_output = await self.read_document(
            ReadDocumentRequest(
                file_path=request.file_path,
                reader=request.reader,
                model=request.model,
                kwargs=request.kwargs,
            )
        )
        return await self.split_document(
            SplitDocumentRequest(
                reader_output=reader_output,
                splitter=request.splitter,
                embedding=request.embedding,
                kwargs=request.splitter_kwargs,
            )
        )

    def _prepare_file_path(
        self,
        request: ReadDocumentRequest,
    ) -> tuple[str, Path | None, str]:
        """Classify ``file_path`` and enforce size and access policy.

        Args:
            request: Typed read request.

        Returns:
            Positional file_path, optional temp file to delete, and source kind.

        Raises:
            ServerAccessDeniedError: If a file or URL is blocked by policy.
            ServerPayloadTooLargeError: If the payload exceeds the size limit.
            ServerConfigurationError: If a URL fetch fails.
        """
        value = request.file_path
        if isinstance(value, (dict, list)):
            encoded = json.dumps(value, ensure_ascii=False)
            self._assert_size(len(encoded.encode("utf-8")))
            return encoded, None, SourceType.JSON.value
        if not isinstance(value, str):
            raise ServerConfigurationError(
                "file_path must be a string, object, or array."
            )

        if BaseReader.is_url(value):
            validate_url(value, self.settings)
            body, _final_url, filename = fetch_url(value, self.settings)
            self._assert_size(len(body))
            suffix = Path(filename).suffix or ".bin"
            handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                handle.write(body)
                handle.flush()
            finally:
                handle.close()
            return str(Path(handle.name)), Path(handle.name), SourceType.URL.value

        candidate = Path(value).expanduser()
        if candidate.exists():
            path = resolve_allowed_file(value, self.settings)
            self._assert_size(path.stat().st_size)
            return str(path), None, SourceType.FILE.value

        self._assert_size(len(value.encode("utf-8")))
        return value, None, SourceType.TEXT.value

    def _assert_size(self, size: int) -> None:
        """Reject sources larger than the configured limit.

        Args:
            size: Payload size in bytes.

        Raises:
            ServerPayloadTooLargeError: If ``size`` exceeds ``max_body_bytes``.
        """
        if size > self.settings.max_body_bytes:
            raise ServerPayloadTooLargeError(
                "Source exceeds SPLITTER_MR_MAX_BODY_BYTES."
            )

    @staticmethod
    def _call_read(
        read_fn: Any, file_path: str, kwargs: dict[str, Any]
    ) -> ReaderOutput:
        """Invoke a reader and translate library exceptions.

        Args:
            read_fn: Bound ``read`` method.
            file_path: Positional input matching ``BaseReader.read``.
            kwargs: Keyword arguments forwarded to the reader.

        Returns:
            Validated reader output.

        Raises:
            ServerError: When the reader fails.
        """
        try:
            return read_fn(file_path, **kwargs)
        except ServerError:
            raise
        except (ReaderConfigException, ReaderException) as error:
            raise ServerConfigurationError(str(error)) from error
        except Exception as error:
            raise ServerConfigurationError("Failed to read the document.") from error

    @staticmethod
    def _call_split(split_fn: Any, reader_output: ReaderOutput) -> SplitterOutput:
        """Invoke a splitter and translate library exceptions.

        Args:
            split_fn: Bound ``split`` method.
            reader_output: Input produced by a reader.

        Returns:
            Validated splitter output.

        Raises:
            ServerError: When the splitter fails.
        """
        try:
            return split_fn(reader_output)
        except ServerError:
            raise
        except (SplitterConfigException, SplitterException) as error:
            raise ServerConfigurationError(str(error)) from error
        except Exception as error:
            raise ServerConfigurationError("Failed to split the document.") from error
