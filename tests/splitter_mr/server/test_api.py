from pathlib import Path

from fastapi.testclient import TestClient

from splitter_mr.schema.models import ReaderOutput, SplitterOutput
from splitter_mr.server.app import create_app
from splitter_mr.server.settings import ServerSettings

# ---- Mocks, fixtures & helpers ---- #


def _settings(**overrides) -> ServerSettings:
    payload = {"allow_urls": False, "max_body_bytes": 1024 * 1024}
    payload.update(overrides)
    return ServerSettings.model_validate(payload)


def _client(monkeypatch, **settings_overrides) -> TestClient:
    fake_reader_output = ReaderOutput(
        text="hello world",
        document_name="doc.txt",
        document_path="",
        document_id="doc-1",
        conversion_method="txt",
        reader_method="vanilla",
    )
    fake_splitter_output = SplitterOutput(
        chunks=["hello", "world"],
        chunk_id=["id-1", "id-2"],
        document_name="doc.txt",
        document_path="",
        document_id="doc-1",
        conversion_method="txt",
        reader_method="vanilla",
        split_method="character_splitter",
        split_params={"chunk_size": 5},
    )

    async def fake_read(self, request):
        return fake_reader_output

    async def fake_split(self, request):
        return fake_splitter_output

    async def fake_read_and_split(self, request):
        return fake_splitter_output

    monkeypatch.setattr(
        "splitter_mr.server.service.PipelineService.read_document",
        fake_read,
    )
    monkeypatch.setattr(
        "splitter_mr.server.service.PipelineService.split_document",
        fake_split,
    )
    monkeypatch.setattr(
        "splitter_mr.server.service.PipelineService.read_and_split",
        fake_read_and_split,
    )
    app = create_app(_settings(**settings_overrides))
    return TestClient(app)


READ_PAYLOAD = {
    "file_path": "hello world",
    "reader": {"reader": "VanillaReader"},
    "kwargs": {"document_name": "doc.txt"},
}

SPLIT_PAYLOAD = {
    "reader_output": {
        "text": "hello world",
        "document_name": "doc.txt",
        "document_path": "",
        "document_id": "doc-1",
        "conversion_method": "txt",
        "reader_method": "vanilla",
        "metadata": {},
    },
    "splitter": {"splitter": "CharacterSplitter", "chunk_size": 5},
}


# ---- Happy path ---- #


def test_get_health_returns_transport_paths(monkeypatch):
    client = _client(monkeypatch)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["docs_url"] == "/docs"
    assert body["mcp_path"] == "/mcp"
    assert body["api_prefix"] == "/api/v1"


def test_list_components_returns_catalog(monkeypatch):
    client = _client(monkeypatch)

    response = client.get("/api/v1/components")

    assert response.status_code == 200
    body = response.json()
    reader_names = {item["name"] for item in body["readers"]}
    splitter_names = {item["name"] for item in body["splitters"]}
    model_names = {item["name"] for item in body["vision_models"]}
    embedding_names = {item["name"] for item in body["embeddings"]}
    assert "VanillaReader" in reader_names
    assert "RecursiveCharacterSplitter" in splitter_names
    assert "SemanticSplitter" in splitter_names
    assert "OpenAIVisionModel" in model_names
    assert "OpenAIEmbedding" in embedding_names


def test_post_read_returns_reader_output(monkeypatch):
    client = _client(monkeypatch)

    response = client.post("/api/v1/read", json=READ_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["text"] == "hello world"
    assert response.json()["document_id"] == "doc-1"


def test_post_split_returns_splitter_output(monkeypatch):
    client = _client(monkeypatch)

    response = client.post("/api/v1/split", json=SPLIT_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["chunks"] == ["hello", "world"]


def test_post_read_and_split_returns_splitter_output(monkeypatch):
    client = _client(monkeypatch)

    response = client.post(
        "/api/v1/read-and-split",
        json={
            **READ_PAYLOAD,
            "splitter": {"splitter": "CharacterSplitter", "chunk_size": 5},
        },
    )

    assert response.status_code == 200
    assert response.json()["document_id"] == "doc-1"


def test_docs_redoc_and_openapi_render(monkeypatch):
    client = _client(monkeypatch)

    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    payload = spec.json()
    assert payload["info"]["title"] == "SplitterMR MCP Server"


def test_openapi_has_unique_operation_ids_and_typed_schemas(monkeypatch):
    client = _client(monkeypatch)
    spec = client.get("/openapi.json").json()

    operation_ids = []
    for path_item in spec["paths"].values():
        for operation in path_item.values():
            if isinstance(operation, dict) and "operationId" in operation:
                operation_ids.append(operation["operationId"])
                assert operation.get("summary")
                assert operation.get("description")
    assert len(operation_ids) == len(set(operation_ids))
    assert "read_document" in operation_ids
    assert "split_document" in operation_ids
    assert "read_and_split_document" in operation_ids
    assert "list_components" in operation_ids
    assert "get_health" in operation_ids

    schemas = spec["components"]["schemas"]
    for name in (
        "ReadDocumentRequest",
        "SplitDocumentRequest",
        "ReadAndSplitRequest",
        "ReaderOutput",
        "SplitterOutput",
        "CharacterSplitterConfiguration",
        "VanillaReaderConfiguration",
        "OpenAIVisionModelConfiguration",
        "OpenAIEmbeddingConfiguration",
        "SemanticSplitterConfiguration",
        "EmbeddingDescriptor",
        "ApiErrorResponse",
        "HealthResponse",
        "VisionModelDescriptor",
    ):
        assert name in schemas
        assert schemas[name].get("type") == "object"


def test_openapi_tags_and_discriminators(monkeypatch):
    client = _client(monkeypatch)
    spec = client.get("/openapi.json").json()
    tag_names = {tag["name"] for tag in spec["tags"]}

    assert tag_names == {"System", "Components", "Reading", "Splitting"}
    read_schema = spec["components"]["schemas"]["ReadDocumentRequest"]
    assert "file_path" in read_schema["properties"]
    assert "description" in read_schema["properties"]["file_path"]
    openai = spec["components"]["schemas"]["OpenAIVisionModelConfiguration"]
    assert openai["properties"]["model"]["description"]


# ---- Error paths ---- #


def test_post_read_returns_422_for_invalid_payload(monkeypatch):
    client = _client(monkeypatch)

    response = client.post("/api/v1/read", json={"kwargs": {"document_name": "a.txt"}})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["details"]


def test_post_read_returns_403_when_file_access_disabled(tmp_path: Path):
    document = tmp_path / "doc.txt"
    document.write_text("hello", encoding="utf-8")
    from splitter_mr.server.app import create_app as factory

    app = factory(_settings())
    client = TestClient(app)
    response = client.post(
        "/api/v1/read",
        json={
            "file_path": str(document),
            "reader": {"reader": "VanillaReader"},
        },
    )

    assert response.status_code == 403
    assert response.json()["code"] == "access_denied"


def test_post_read_returns_413_when_content_length_exceeds_limit():
    app = create_app(_settings(max_body_bytes=10))
    client = TestClient(app)

    response = client.post(
        "/api/v1/read",
        json=READ_PAYLOAD,
        headers={"content-length": "9999"},
    )

    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"


def test_post_read_returns_422_for_incompatible_reader():
    app = create_app(_settings())
    client = TestClient(app)

    response = client.post(
        "/api/v1/read",
        json={
            "file_path": {"title": "Report"},
            "reader": {"reader": "TextractReader"},
        },
    )

    assert response.status_code == 422


# ---- Edge cases ---- #


def test_health_does_not_expose_allowed_root(monkeypatch, tmp_path: Path):
    client = _client(monkeypatch, allowed_root=tmp_path)

    body = client.get("/health").json()

    assert "allowed_root" not in body
    assert str(tmp_path) not in str(body)
