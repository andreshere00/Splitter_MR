"""FastAPI application factory that mounts REST routes and the MCP server."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from splitter_mr import __version__

from .api import create_api_router, error_response
from .enums import HealthStatus
from .exceptions import ServerError
from .mcp import create_mcp_server
from .schemas import ApiErrorResponse, HealthResponse, ValidationErrorDetail
from .service import PipelineService
from .settings import ServerSettings

APP_DESCRIPTION = """
SplitterMR HTTP service exposing **read**, **split**, and **read-and-split**
operations through REST and MCP.

REST endpoints under `/api/v1` are documented here in Swagger. The same
pipeline is also available as typed MCP tools at the Streamable HTTP mount
path (default `/mcp`).

The service is **stateless**: every call returns a `ReaderOutput` or
`SplitterOutput` and does not persist documents, embeddings, or chunks.

**Read contract**

Request bodies match `BaseReader.read(file_path, model=None, **kwargs)`:

* `file_path` is a server-local path, URL, raw string, or JSON value.
* Existing files require `SPLITTER_MR_ALLOWED_ROOT`.
* URLs require `SPLITTER_MR_ALLOW_URLS=true` and reject private or loopback
  targets.
* Optional `model` JSON constructs a `BaseVisionModel` (needs the
  `multimodal` extra). Omit API keys to use environment variables.

**Security**

This version does not implement authentication. Deploy only on a private
network or behind an authenticated reverse proxy. Inline API keys may be
captured by proxies, MCP clients, or traces.

**Not in v1**

`SemanticSplitter`, persistent retrieval, and document query tools are out
of scope.
""".strip()

HEALTH_DESCRIPTION = """
Liveness probe used by Docker and orchestrators.

Returns the package version and the paths of Swagger, ReDoc, OpenAPI, REST,
and MCP. It never includes credentials, filesystem roots, or allowlists.
""".strip()


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    """Build the FastAPI application with REST routes and a mounted MCP app.

    Args:
        settings: Optional settings override. Defaults to environment values.

    Returns:
        Configured FastAPI application.

    Raises:
        ModuleNotFoundError: If FastAPI or FastMCP are not installed.
    """
    resolved = settings or ServerSettings()
    service = PipelineService(resolved)
    mcp = create_mcp_server(service)
    mcp_app = mcp.http_app(path="/")

    application = FastAPI(
        title="SplitterMR MCP Server",
        summary="Read and split documents through REST and MCP.",
        description=APP_DESCRIPTION,
        version=__version__,
        contact={
            "name": "Andrés Herencia",
            "url": "https://github.com/andreshere00/Splitter_MR",
            "email": "andresherencia2000@gmail.com",
        },
        license_info={
            "name": "See repository LICENSE",
            "url": "https://github.com/andreshere00/Splitter_MR/blob/main/LICENSE",
        },
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=mcp_app.lifespan,
        openapi_tags=[
            {
                "name": "System",
                "description": "Operational endpoints such as liveness probes.",
            },
            {
                "name": "Components",
                "description": "Discovery of readers, splitters, extras, and schemas.",
            },
            {
                "name": "Reading",
                "description": "Turn a file_path into a ReaderOutput.",
            },
            {
                "name": "Splitting",
                "description": "Turn a ReaderOutput, or a source, into a SplitterOutput.",
            },
        ],
    )
    application.state.settings = resolved
    application.state.service = service
    application.state.mcp = mcp

    @application.middleware("http")
    async def limit_request_body(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
            except ValueError:
                length = 0
            if length > resolved.max_body_bytes:
                return error_response(
                    ServerError(
                        "Request body exceeds SPLITTER_MR_MAX_BODY_BYTES.",
                        code="payload_too_large",
                    )
                )
        return await call_next(request)

    @application.exception_handler(ServerError)
    async def handle_server_error(_: Request, error: ServerError) -> JSONResponse:
        return error_response(error)

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _: Request, error: RequestValidationError
    ) -> JSONResponse:
        details = [
            ValidationErrorDetail(
                loc=[str(part) for part in item.get("loc", [])],
                msg=str(item.get("msg", "Invalid value")),
                type=str(item.get("type", "value_error")),
            )
            for item in error.errors()
        ]
        payload = ApiErrorResponse(
            code="validation_error",
            message="Request failed schema validation.",
            details=details,
        )
        return JSONResponse(
            status_code=422,
            content=payload.model_dump(mode="json"),
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, error: Exception) -> JSONResponse:
        if isinstance(
            error, (ServerError, RequestValidationError, StarletteHTTPException)
        ):
            raise error
        payload = ApiErrorResponse(
            code="internal_error",
            message="An unexpected server error occurred.",
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=payload.model_dump(mode="json"),
        )

    @application.get(
        "/health",
        response_model=HealthResponse,
        status_code=status.HTTP_200_OK,
        summary="Service liveness",
        description=HEALTH_DESCRIPTION,
        operation_id="get_health",
        tags=["System"],
    )
    async def get_health() -> HealthResponse:
        return HealthResponse(
            service="splitter-mr-mcp",
            status=HealthStatus.OK,
            version=__version__,
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
            mcp_path=resolved.mcp_path,
            api_prefix=resolved.api_prefix,
        )

    application.include_router(create_api_router(), prefix=resolved.api_prefix)
    application.mount(resolved.mcp_path, mcp_app)
    return application


app = create_app()
