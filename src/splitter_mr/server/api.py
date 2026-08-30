"""Typed FastAPI REST wrappers for the SplitterMR pipeline."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from splitter_mr.schema.models import ReaderOutput, SplitterOutput

from .exceptions import ServerError
from .schemas import (
    ApiErrorResponse,
    ComponentCatalogResponse,
    ReadAndSplitRequest,
    ReadDocumentRequest,
    SplitDocumentRequest,
)
from .service import PipelineService

ERROR_RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {
        "model": ApiErrorResponse,
        "description": (
            "The reader or splitter cannot be used, or the request is incompatible "
            "with the selected components."
        ),
    },
    403: {
        "model": ApiErrorResponse,
        "description": (
            "File or URL access is disabled, the path is outside the allowed root, "
            "or the URL target is blocked."
        ),
    },
    413: {
        "model": ApiErrorResponse,
        "description": "The JSON body or document source exceeds SPLITTER_MR_MAX_BODY_BYTES.",
    },
    422: {
        "model": ApiErrorResponse,
        "description": "The request body failed schema validation.",
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ApiErrorResponse,
        "description": "An unexpected server error occurred. Details are not disclosed.",
    },
}

LIST_COMPONENTS_DESCRIPTION = """
Return every reader, splitter, vision model, and embedding the server can talk
about.

Readers include VanillaReader (core), MarkItDownReader, DoclingReader, and
TextractReader. Optional readers report `missing_extra` when their extra is not
installed. Each reader lists the file_path kinds it accepts: VanillaReader
supports text, json, file, and url; TextractReader accepts file only.

Vision models require `splitter-mr[multimodal]`. Pass them as the top-level
`model` object on `POST /read` and `POST /read-and-split`. Credentials may be
omitted so the provider environment variable is used. MarkItDownReader only
accepts OpenAI-compatible clients.

Embeddings also require `splitter-mr[multimodal]`. Pass them as the top-level
`embedding` object on `POST /split` and `POST /read-and-split` when using
`SemanticSplitter`.

Splitters include every JSON-serializable constructor, including
`SemanticSplitter`. Extra constructor arguments that are not `chunk_size` or
`chunk_overlap` belong in `kwargs` (`POST /split`) or `splitter_kwargs`
(`POST /read-and-split`).

Use this catalog from Swagger or MCP `list_components` before calling the
processing endpoints.
""".strip()

READ_DESCRIPTION = """
Read one document and return a `ReaderOutput`.

The body mirrors `BaseReader.read(file_path, model=None, **kwargs)`:

* `file_path` — server-local path, `http`/`https` URL, raw string, or JSON
  object/array. Existing files require `SPLITTER_MR_ALLOWED_ROOT`. URLs require
  `SPLITTER_MR_ALLOW_URLS=true` and reject loopback, private, link-local,
  multicast, and reserved addresses. Redirects are re-validated hop by hop.
* `reader` — constructor configuration. Defaults to `VanillaReader`. Optional
  readers need `splitter-mr[markitdown]`, `[docling]`, or `[textract]`.
* `model` — optional vision-model constructor configuration. Requires
  `splitter-mr[multimodal]`. Omit `api_key` to use the provider environment
  variable. `TextractReader` rejects a model.
* `kwargs` — extra `read` arguments such as `document_name`, `metadata`,
  `prompt`, and `vlm_parameters`. Do not set `file_path` or `model` here.

The response is the same `ReaderOutput` contract used by the Python API. Pass it
unchanged to `POST /split` when you want a two-step workflow. For a one-call
workflow use `POST /read-and-split`.
""".strip()

SPLIT_DESCRIPTION = """
Split a previously produced `ReaderOutput` into a `SplitterOutput`.

`reader_output` must be the complete object returned by `POST /read`, including
`document_id`, `document_name`, `conversion_method`, `reader_method`, and
`metadata`. Splitters copy those fields onto the chunked result.

Select a splitter with the `splitter` discriminator. Typed constructor fields
may live on that object. Additional constructor arguments that are not
`chunk_size` or `chunk_overlap` — such as `patterns`, `separators`,
`include_delimiters`, `headers_to_split_on`, `language`, `tag`, `num_rows`,
and `buffer_size` — can also be passed in `kwargs`. Values in `kwargs`
override the same keys on `splitter`. Do not set `splitter` or `embedding`
inside `kwargs`.

`SemanticSplitter` requires a top-level `embedding` object (OpenAI, Azure
OpenAI, OpenRouter, Gemini, Hugging Face, or Anthropic/Voyage). That needs
`splitter-mr[multimodal]`. Omit `api_key` to use the provider environment
variable. Other splitters reject `embedding`.

The server is stateless: nothing is stored after the response is returned.
""".strip()

READ_AND_SPLIT_DESCRIPTION = """
Read a document and split it in one request.

This is the preferred REST workflow. The server runs the same pipeline as
`POST /read` followed by `POST /split`, passing the exact `ReaderOutput` into
the splitter so identifiers and metadata are preserved.

Provide the same read fields as `POST /read`:

* `file_path`, `reader`, optional vision `model`, and read `kwargs`
* `splitter` (default `RecursiveCharacterSplitter`)
* optional `embedding` (required for `SemanticSplitter`)
* `splitter_kwargs` — extra splitter constructor arguments, equivalent to
  `kwargs` on `POST /split`

File and URL policy, optional extras, and payload limits are identical to the
dedicated read endpoint.

The response is a `SplitterOutput` with `chunks`, `chunk_id`, `split_method`,
and the original document metadata.
""".strip()


def get_service(request: Request) -> PipelineService:
    """Return the pipeline service stored on the FastAPI application.

    Args:
        request: Incoming Starlette request.

    Returns:
        Shared ``PipelineService`` instance.
    """
    return request.app.state.service


def create_api_router() -> APIRouter:
    """Build the versioned REST router.

    Returns:
        Router with component discovery, read, split, and read-and-split routes.
    """
    router = APIRouter()

    @router.get(
        "/components",
        response_model=ComponentCatalogResponse,
        status_code=status.HTTP_200_OK,
        summary="List readers, splitters, vision models, and embeddings",
        description=LIST_COMPONENTS_DESCRIPTION,
        operation_id="list_components",
        tags=["Components"],
        responses=ERROR_RESPONSES,
    )
    async def list_components(
        service: PipelineService = Depends(get_service),
    ) -> ComponentCatalogResponse:
        return service.list_components()

    @router.post(
        "/read",
        response_model=ReaderOutput,
        status_code=status.HTTP_200_OK,
        summary="Read a document into structured text",
        description=READ_DESCRIPTION,
        operation_id="read_document",
        tags=["Reading"],
        responses={
            status.HTTP_200_OK: {
                "description": "Validated ReaderOutput produced by the selected reader.",
                "model": ReaderOutput,
            },
            **ERROR_RESPONSES,
        },
    )
    async def read_document(
        payload: ReadDocumentRequest,
        service: PipelineService = Depends(get_service),
    ) -> ReaderOutput:
        return await service.read_document(payload)

    @router.post(
        "/split",
        response_model=SplitterOutput,
        status_code=status.HTTP_200_OK,
        summary="Split a ReaderOutput into chunks",
        description=SPLIT_DESCRIPTION,
        operation_id="split_document",
        tags=["Splitting"],
        responses={
            status.HTTP_200_OK: {
                "description": "Validated SplitterOutput with preserved document metadata.",
                "model": SplitterOutput,
            },
            **ERROR_RESPONSES,
        },
    )
    async def split_document(
        payload: SplitDocumentRequest,
        service: PipelineService = Depends(get_service),
    ) -> SplitterOutput:
        return await service.split_document(payload)

    @router.post(
        "/read-and-split",
        response_model=SplitterOutput,
        status_code=status.HTTP_200_OK,
        summary="Read a document and split it in one call",
        description=READ_AND_SPLIT_DESCRIPTION,
        operation_id="read_and_split_document",
        tags=["Splitting"],
        responses={
            status.HTTP_200_OK: {
                "description": "Validated SplitterOutput from the composed pipeline.",
                "model": SplitterOutput,
            },
            **ERROR_RESPONSES,
        },
    )
    async def read_and_split_document(
        payload: ReadAndSplitRequest,
        service: PipelineService = Depends(get_service),
    ) -> SplitterOutput:
        return await service.read_and_split(payload)

    return router


def server_error_status(error: ServerError) -> int:
    """Map a structured server error to an HTTP status code.

    Args:
        error: Service-boundary exception.

    Returns:
        HTTP status code.
    """
    mapping = {
        "access_denied": 403,
        "payload_too_large": 413,
        "component_unavailable": status.HTTP_400_BAD_REQUEST,
        "invalid_configuration": status.HTTP_400_BAD_REQUEST,
    }
    return mapping.get(error.code, status.HTTP_500_INTERNAL_SERVER_ERROR)


def error_response(error: ServerError, status_code: int | None = None) -> JSONResponse:
    """Serialize a structured server error for HTTP clients.

    Args:
        error: Service-boundary exception.
        status_code: Optional override. Defaults to the mapped status.

    Returns:
        JSON response with ``ApiErrorResponse`` content.
    """
    payload = ApiErrorResponse(code=error.code, message=error.message)
    return JSONResponse(
        status_code=status_code or server_error_status(error),
        content=payload.model_dump(mode="json"),
    )
