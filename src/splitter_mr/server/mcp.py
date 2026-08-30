"""FastMCP tool registration for the SplitterMR server."""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from splitter_mr.schema.models import ReaderOutput, SplitterOutput

from .exceptions import ServerError
from .schemas import (
    ComponentCatalogResponse,
    ReadAndSplitRequest,
    ReadDocumentRequest,
    SplitDocumentRequest,
)
from .service import PipelineService

LIST_COMPONENTS_DESCRIPTION = """List SplitterMR readers, splitters, vision models, and embeddings.

Returns installation extras, availability, supported file_path kinds, compatible
vision models, and the OpenAPI schema name for each configuration model.

Vision models are passed as the top-level model field on read_document and
read_and_split. Embeddings are passed as the top-level embedding field on
split_document and read_and_split. SemanticSplitter requires embedding and the
multimodal extra.

Extra splitter constructor arguments that are not chunk_size or chunk_overlap
belong in kwargs (split_document) or splitter_kwargs (read_and_split).

Use this tool before read_document or read_and_split when you need to choose a
reader extra (markitdown, docling, textract), a vision model, an embedding, or
a splitter strategy.
"""

READ_DOCUMENT_DESCRIPTION = """Read one document and return a SplitterMR ReaderOutput.

The request mirrors BaseReader.read(file_path, model=None, **kwargs):
- file_path: server-local path, http(s) URL, raw string, or JSON object/array.
  Existing files require SPLITTER_MR_ALLOWED_ROOT. URLs require
  SPLITTER_MR_ALLOW_URLS=true. Private, loopback, link-local, multicast, and
  reserved IP addresses are rejected. Redirects are re-validated hop by hop.
- reader: VanillaReader (default), MarkItDownReader, DoclingReader, or
  TextractReader. Optional extras raise a structured install hint when missing.
- model: optional BaseVisionModel constructor configuration. Requires the
  multimodal extra. Omit api_key to use the provider environment variable.
  TextractReader does not accept a model. MarkItDownReader needs an
  OpenAI-compatible client (OpenAI, Azure OpenAI, Anthropic, OpenRouter).
- kwargs: extra read options such as document_name, metadata, prompt, and
  vlm_parameters. Do not include file_path or model.

The result is a ReaderOutput with text, document_id, conversion_method,
reader_method, and metadata. Pass it unchanged to split_document.
"""

SPLIT_DOCUMENT_DESCRIPTION = """Split a previously produced ReaderOutput into a SplitterOutput.

The reader_output field must be the complete object returned by read_document,
including document_id, document_name, conversion_method, and metadata. Splitters
preserve that metadata on the resulting chunks.

Select a splitter with the splitter discriminator. Typed constructor fields may
live on that object. Additional constructor arguments that are not chunk_size
or chunk_overlap — patterns, separators, include_delimiters, headers_to_split_on,
language, tag, num_rows, buffer_size — can also be passed in kwargs. Values in
kwargs override the same keys on splitter.

SemanticSplitter requires a top-level embedding object (OpenAIEmbedding,
AzureOpenAIEmbedding, OpenRouterEmbedding, GeminiEmbedding,
HuggingFaceEmbedding, or AnthropicEmbedding). That needs the multimodal extra.
Other splitters reject embedding.
"""

READ_AND_SPLIT_DESCRIPTION = """Read a document and split it in one call.

This is the preferred agent workflow: it runs the same internal read_document
path, then passes the exact ReaderOutput to split_document so IDs and metadata
are preserved.

Use the same file_path, reader, model, and kwargs fields as read_document.
Add splitter, optional embedding (required for SemanticSplitter), and
splitter_kwargs for extra splitter constructor arguments. The default reader is
VanillaReader and the default splitter is RecursiveCharacterSplitter.

Returns a SplitterOutput with chunks, chunk_id values, and the original document
metadata. The server is stateless and does not persist chunks.
"""


def create_mcp_server(service: PipelineService) -> FastMCP:
    """Create a FastMCP server whose tools delegate to the pipeline service.

    Args:
        service: Shared pipeline service used by REST and MCP.

    Returns:
        Configured FastMCP instance with four typed tools.
    """
    mcp = FastMCP(
        "SplitterMR",
        instructions=(
            "SplitterMR turns documents into LLM-ready text chunks. Prefer "
            "read_and_split for a one-call workflow. Use list_components to "
            "discover readers, extras, vision models, embeddings, and splitters. "
            "File paths are server-local. The server does not persist documents "
            "or embeddings."
        ),
    )

    @mcp.tool(name="list_components", description=LIST_COMPONENTS_DESCRIPTION)
    async def list_components() -> ComponentCatalogResponse:
        return service.list_components()

    @mcp.tool(name="read_document", description=READ_DOCUMENT_DESCRIPTION)
    async def read_document(request: ReadDocumentRequest) -> ReaderOutput:
        try:
            return await service.read_document(request)
        except ServerError as error:
            raise _tool_error(error) from error

    @mcp.tool(name="split_document", description=SPLIT_DOCUMENT_DESCRIPTION)
    async def split_document(request: SplitDocumentRequest) -> SplitterOutput:
        try:
            return await service.split_document(request)
        except ServerError as error:
            raise _tool_error(error) from error

    @mcp.tool(name="read_and_split", description=READ_AND_SPLIT_DESCRIPTION)
    async def read_and_split(request: ReadAndSplitRequest) -> SplitterOutput:
        try:
            return await service.read_and_split(request)
        except ServerError as error:
            raise _tool_error(error) from error

    return mcp


def _tool_error(error: ServerError) -> ToolError:
    """Map a service error onto a FastMCP tool error.

    Args:
        error: Structured server exception.

    Returns:
        ToolError whose message includes the stable code.
    """
    return ToolError(f"{error.code}: {error.message}")
