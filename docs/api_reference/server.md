# MCP and REST server

The optional `mcp` extra serves SplitterMR as a **stateless** FastAPI application.
REST endpoints under `/api/v1` are documented in Swagger. Typed MCP tools share the
same pipeline and are mounted at `/mcp`.

```text
MCP client  -->  FastMCP tools  -->  PipelineService  -->  Reader / Splitter
REST client -->  FastAPI routes -->  PipelineService  -->  ReaderOutput / SplitterOutput
```

Install and run:

```bash
pip install "splitter-mr[mcp]"
splitter-mr-mcp
```

Or `poe serve-mcp` from a repository checkout. For vision-language models and
`SemanticSplitter` embeddings, also install `splitter-mr[multimodal]` and provide
provider credentials in the environment or optionally in the request `model` /
`embedding` objects.

## Endpoints

| Method | Path | Operation ID | Response |
| ------ | ---- | ------------ | -------- |
| `GET` | `/health` | `get_health` | `HealthResponse` |
| `GET` | `/api/v1/components` | `list_components` | `ComponentCatalogResponse` |
| `POST` | `/api/v1/read` | `read_document` | `ReaderOutput` |
| `POST` | `/api/v1/split` | `split_document` | `SplitterOutput` |
| `POST` | `/api/v1/read-and-split` | `read_and_split_document` | `SplitterOutput` |
| MCP | `/mcp` | `list_components`, `read_document`, `split_document`, `read_and_split` | same models |

Interactive docs:

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI schema: `/openapi.json`

Request bodies follow the Python method signatures:

- `file_path`: server-local path, `http`/`https` URL, raw string, or JSON value
- `reader.reader`: `VanillaReader`, `MarkItDownReader`, `DoclingReader`, or `TextractReader`
- `model.model`: optional `BaseVisionModel` class name on `/read` and `/read-and-split`
- `kwargs`: extra `read` arguments (`document_name`, `prompt`, `vlm_parameters`, …)
- `splitter.splitter`: one of the JSON-serializable splitter classes, including `SemanticSplitter`
- `embedding.embedding`: optional `BaseEmbedding` class name on `/split` and `/read-and-split`
- `kwargs` on `/split` and `splitter_kwargs` on `/read-and-split`: extra splitter constructor arguments
- `reader_output`: full `ReaderOutput` object for `POST /split`

`SemanticSplitter` requires a top-level `embedding` object and `splitter-mr[multimodal]`.
Other splitters reject `embedding`. Omit `api_key` on `model` and `embedding` to use
provider environment variables.

This is a **breaking** change from the previous `source.source_type` contract.

## Example requests

Read inline text:

```json
{
  "file_path": "Lorem ipsum dolor sit amet.",
  "reader": { "reader": "VanillaReader" },
  "kwargs": { "document_name": "lorem.txt" }
}
```

Split a previous `ReaderOutput`:

```json
{
  "reader_output": {
    "text": "Lorem ipsum dolor sit amet.",
    "document_name": "lorem.txt",
    "document_path": "",
    "document_id": "732b9530-3e41-4a1a-a4ea-1d9d6fe815d3",
    "conversion_method": "txt",
    "reader_method": "vanilla",
    "metadata": {}
  },
  "splitter": { "splitter": "CharacterSplitter" },
  "kwargs": { "chunk_size": 50, "chunk_overlap": 10 }
}
```

Keyword patterns and other constructor fields that are not `chunk_size` /
`chunk_overlap` can live on `splitter` or in `kwargs`:

```json
{
  "reader_output": { "...": "previous ReaderOutput" },
  "splitter": {
    "splitter": "KeywordSplitter",
    "patterns": ["CHAPTER\\s+\\d+"]
  },
  "kwargs": { "include_delimiters": "before", "chunk_size": 100000 }
}
```

Preferred one-call workflow:

```json
{
  "file_path": "Lorem ipsum dolor sit amet.",
  "reader": { "reader": "VanillaReader" },
  "kwargs": { "document_name": "lorem.txt" },
  "splitter": { "splitter": "RecursiveCharacterSplitter" },
  "splitter_kwargs": { "chunk_size": 100, "chunk_overlap": 0.1 }
}
```

Optional vision model (credentials omitted so the server uses `OPENAI_API_KEY`):

```json
{
  "file_path": "/data/docs/manual.pdf",
  "reader": { "reader": "VanillaReader" },
  "model": { "model": "OpenAIVisionModel", "model_name": "gpt-4.1" },
  "kwargs": { "prompt": "Extract the visible text." }
}
```

Optional embedding for `SemanticSplitter` (uses `OPENAI_API_KEY` when `api_key` is omitted):

```json
{
  "file_path": "Lorem ipsum dolor sit amet.",
  "reader": { "reader": "VanillaReader" },
  "kwargs": { "document_name": "lorem.txt" },
  "splitter": { "splitter": "SemanticSplitter" },
  "embedding": {
    "embedding": "OpenAIEmbedding",
    "model_name": "text-embedding-3-large"
  },
  "splitter_kwargs": { "chunk_size": 1000, "buffer_size": 1 }
}
```

## Inputs and security

| `file_path` kind | Default | Notes |
| ---------------- | ------- | ----- |
| Raw string | Enabled | Treated as inline text when it is not an existing file or URL. Limited by `SPLITTER_MR_MAX_BODY_BYTES`. |
| JSON object/array | Enabled | `VanillaReader` only. |
| Existing file | Disabled | Path on the **server**, not the client. Requires `SPLITTER_MR_ALLOWED_ROOT`. Symlinks are resolved; paths outside the root are rejected. |
| URL | Disabled | `http`/`https` only. Requires `SPLITTER_MR_ALLOW_URLS=true`. Private, loopback, link-local, multicast, and reserved IPs are rejected. Redirects are re-validated hop by hop. Optional host allowlist: `SPLITTER_MR_ALLOWED_URL_HOSTS`. |

This version does **not** implement authentication. Deploy the image only on a
private network or behind an authenticated reverse proxy or API gateway.

Optional `model.api_key` values are write-only `SecretStr` fields. Prefer
environment variables. Inline keys may be captured by proxies, MCP clients, or
traces. `MarkItDownReader` accepts OpenAI-compatible clients only (OpenAI, Azure
OpenAI, Anthropic, OpenRouter). `TextractReader` rejects a vision model.

## Environment

Prefix: `SPLITTER_MR_`.

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `HOST` | `127.0.0.1` | Bind address. Docker overrides to `0.0.0.0`. |
| `PORT` | `8000` | TCP port. |
| `MCP_PATH` | `/mcp` | Streamable HTTP mount path. |
| `API_PREFIX` | `/api/v1` | REST prefix. |
| `ALLOWED_ROOT` | unset | Filesystem root for existing file paths. |
| `ALLOW_URLS` | `false` | Enable URL fetches. |
| `ALLOWED_URL_HOSTS` | empty | Optional comma-separated hostname allowlist. |
| `MAX_BODY_BYTES` | `10485760` | JSON body and inlined source size limit. |
| `LOG_LEVEL` | `info` | Uvicorn log level. |

Provider credentials such as `OPENAI_API_KEY` are read by vision-model and
embedding constructors when `model.api_key` or `embedding.api_key` is omitted.

## Docker

```bash
docker build -f Dockerfile.server -t splitter-mr-mcp .
docker run --rm -p 8000:8000 \
  -e SPLITTER_MR_ALLOWED_ROOT=/data \
  -v /path/to/docs:/data:ro \
  splitter-mr-mcp
```

The default image installs `splitter-mr[mcp]` only (no PyTorch). Add
`--extra multimodal` at build time when vision models are required. The image
runs as a non-root user, exposes port 8000, and health-checks `/health`.
Document access stays disabled unless you mount a directory and set
`SPLITTER_MR_ALLOWED_ROOT`.

## Development ports

`poe serve-mcp` binds the API server to port **8000**. `poe docs` serves MkDocs
on port **8001** so a docs tab cannot poll `/livereload` against the API
process. Close any leftover `http://localhost:8000` docs tabs after upgrading.

## MCP client configuration

Example Cursor MCP server entry against a locally running process:

```json
{
  "mcpServers": {
    "splitter-mr": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Tools:

- `list_components` — readers, extras, vision models, embeddings, splitter catalog
- `read_document` — `ReadDocumentRequest` → `ReaderOutput`
- `split_document` — `SplitDocumentRequest` → `SplitterOutput`
- `read_and_split` — preferred one-call workflow → `SplitterOutput`

## Troubleshooting

| Symptom | Likely cause |
| ------- | ------------ |
| `403 access_denied` on a file path | `SPLITTER_MR_ALLOWED_ROOT` is unset or the path escapes the root. |
| `403 access_denied` on a URL | URLs are disabled, the host is not allowlisted, or the IP is private. |
| `400 component_unavailable` | Optional extra missing (`markitdown`, `docling`, `textract`, `multimodal`). |
| `413 payload_too_large` | Body or source exceeds `SPLITTER_MR_MAX_BODY_BYTES`. |
| `422 validation_error` | Discriminator, reserved kwargs, or field constraint failed. Inspect `details`. |
| MCP session errors after mount | The FastAPI app must use the MCP ASGI lifespan. `create_app()` already does this. |
| `/livereload/...` 404 in API logs | A stale MkDocs tab is polling port 8000. Use `poe docs` on 8001 and close old tabs. |

Python API reference for the underlying contracts:

- [Readers](reader.md)
- [Splitters](splitter.md)
- [Vision models](model.md)
- [Exceptions](exceptions.md)
