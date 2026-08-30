import asyncio

from splitter_mr.schema.models import ReaderOutput, SplitterOutput
from splitter_mr.server.exceptions import ServerAccessDeniedError
from splitter_mr.server.mcp import create_mcp_server
from splitter_mr.server.service import PipelineService
from splitter_mr.server.settings import ServerSettings

# ---- Mocks, fixtures & helpers ---- #


def _service() -> PipelineService:
    return PipelineService(ServerSettings())


def _reader_output() -> ReaderOutput:
    return ReaderOutput(
        text="hello world",
        document_name="doc.txt",
        document_path="",
        document_id="doc-1",
        conversion_method="txt",
        reader_method="vanilla",
    )


def _splitter_output() -> SplitterOutput:
    return SplitterOutput(
        chunks=["hello", "world"],
        chunk_id=["id-1", "id-2"],
        document_name="doc.txt",
        document_path="",
        document_id="doc-1",
        conversion_method="txt",
        reader_method="vanilla",
        split_method="character_splitter",
    )


READ_ARGS = {
    "request": {
        "file_path": "hello world",
        "reader": {"reader": "VanillaReader"},
        "kwargs": {"document_name": "doc.txt"},
    }
}


# ---- Happy path ---- #


def test_mcp_lists_four_tools_with_detailed_descriptions(monkeypatch):
    from fastmcp import Client

    mcp = create_mcp_server(_service())

    async def _run():
        async with Client(mcp) as client:
            tools = await client.list_tools()
            names = sorted(tool.name for tool in tools)
            assert names == [
                "list_components",
                "read_and_split",
                "read_document",
                "split_document",
            ]
            by_name = {tool.name: tool for tool in tools}
            assert "SemanticSplitter" in by_name["list_components"].description
            assert "embedding" in by_name["split_document"].description.lower()
            assert "splitter_kwargs" in by_name["read_and_split"].description
            assert "SPLITTER_MR_ALLOWED_ROOT" in by_name["read_document"].description
            assert "ReaderOutput" in by_name["split_document"].description
            assert "one call" in by_name["read_and_split"].description.lower()
            assert by_name["read_document"].inputSchema
            assert by_name["read_and_split"].inputSchema

    asyncio.run(_run())


def test_mcp_read_document_delegates_to_service(monkeypatch):
    from fastmcp import Client

    captured = {}

    async def fake_read(self, request):
        captured["request"] = request
        return _reader_output()

    monkeypatch.setattr(
        "splitter_mr.server.service.PipelineService.read_document",
        fake_read,
    )
    mcp = create_mcp_server(_service())

    async def _run():
        async with Client(mcp) as client:
            result = await client.call_tool("read_document", READ_ARGS)
            payload = result.data if hasattr(result, "data") else result
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump()
            elif hasattr(result, "structured_content"):
                payload = result.structured_content
            assert payload["text"] == "hello world"
            assert captured["request"].file_path == "hello world"

    asyncio.run(_run())


def test_mcp_read_and_split_returns_splitter_output(monkeypatch):
    from fastmcp import Client

    async def fake_read_and_split(self, request):
        return _splitter_output()

    monkeypatch.setattr(
        "splitter_mr.server.service.PipelineService.read_and_split",
        fake_read_and_split,
    )
    mcp = create_mcp_server(_service())

    async def _run():
        async with Client(mcp) as client:
            result = await client.call_tool(
                "read_and_split",
                {
                    "request": {
                        **READ_ARGS["request"],
                        "splitter": {
                            "splitter": "CharacterSplitter",
                            "chunk_size": 5,
                        },
                    }
                },
            )
            payload = getattr(result, "structured_content", None) or getattr(
                result, "data", result
            )
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump()
            assert payload["chunks"] == ["hello", "world"]

    asyncio.run(_run())


def test_mcp_list_components_returns_catalog():
    from fastmcp import Client

    mcp = create_mcp_server(_service())

    async def _run():
        async with Client(mcp) as client:
            result = await client.call_tool("list_components", {})
            payload = getattr(result, "structured_content", None) or getattr(
                result, "data", result
            )
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump()
            names = {item["name"] for item in payload["splitters"]}
            vision = {item["name"] for item in payload["vision_models"]}
            embeddings = {item["name"] for item in payload["embeddings"]}
            assert "SemanticSplitter" in names
            assert "OpenAIVisionModel" in vision
            assert "OpenAIEmbedding" in embeddings

    asyncio.run(_run())


# ---- Error paths ---- #


def test_mcp_read_document_maps_access_denied(monkeypatch):
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    async def fake_read(self, request):
        raise ServerAccessDeniedError("blocked")

    monkeypatch.setattr(
        "splitter_mr.server.service.PipelineService.read_document",
        fake_read,
    )
    mcp = create_mcp_server(_service())

    async def _run():
        async with Client(mcp) as client:
            try:
                await client.call_tool("read_document", READ_ARGS)
            except ToolError as error:
                assert "access_denied" in str(error)
                assert "blocked" in str(error)
                return
            raise AssertionError("expected ToolError")

    asyncio.run(_run())


# ---- Edge cases ---- #


def test_mounted_mcp_initialize_with_fastapi_lifespan():
    from fastapi.testclient import TestClient

    from splitter_mr.server.app import create_app

    app = create_app(ServerSettings())
    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            headers={
                "accept": "application/json, text/event-stream",
                "content-type": "application/json",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "tests", "version": "0"},
                },
            },
        )

    assert response.status_code in {200, 406}
    if response.status_code == 200:
        body = response.text
        assert "SplitterMR" in body or "protocolVersion" in body or "result" in body
