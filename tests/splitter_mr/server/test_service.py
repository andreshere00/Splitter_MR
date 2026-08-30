import asyncio
import threading
from pathlib import Path

import pytest

from splitter_mr.schema.exceptions import ReaderConfigException, SplitterConfigException
from splitter_mr.schema.models import ReaderOutput, SplitterOutput
from splitter_mr.server.exceptions import (
    ServerAccessDeniedError,
    ServerComponentUnavailableError,
    ServerConfigurationError,
    ServerPayloadTooLargeError,
)
from splitter_mr.server.schemas import (
    CharacterSplitterConfiguration,
    KeywordSplitterConfiguration,
    OpenAIEmbeddingConfiguration,
    OpenAIVisionModelConfiguration,
    ReadAndSplitRequest,
    ReadDocumentRequest,
    SemanticSplitterConfiguration,
    SplitDocumentRequest,
    TextractReaderConfiguration,
    VanillaReaderConfiguration,
)
from splitter_mr.server.service import PipelineService
from splitter_mr.server.settings import ServerSettings

# ---- Mocks, fixtures & helpers ---- #


def _settings(**overrides) -> ServerSettings:
    payload = {"allow_urls": False, "max_body_bytes": 1024}
    payload.update(overrides)
    return ServerSettings.model_validate(payload)


def _reader_output(text: str = "hello world") -> ReaderOutput:
    return ReaderOutput(
        text=text,
        document_name="doc.txt",
        document_path="",
        document_id="doc-1",
        conversion_method="txt",
        reader_method="vanilla",
        metadata={"k": "v"},
    )


class _FakeReader:
    def __init__(self) -> None:
        self.file_path = None
        self.kwargs = None
        self.thread_id = None

    def read(self, file_path=None, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs
        self.thread_id = threading.get_ident()
        if isinstance(file_path, str):
            return _reader_output(file_path)
        return _reader_output()


class _FakeSplitter:
    def __init__(self) -> None:
        self.input = None
        self.thread_id = None

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        self.input = reader_output
        self.thread_id = threading.get_ident()
        return SplitterOutput(
            chunks=["hello", "world"],
            chunk_id=["id-1", "id-2"],
            document_name=reader_output.document_name,
            document_path=reader_output.document_path,
            document_id=reader_output.document_id,
            conversion_method=reader_output.conversion_method,
            reader_method=reader_output.reader_method,
            split_method="character_splitter",
            metadata=reader_output.metadata,
        )


# ---- Happy path ---- #


def test_read_document_returns_reader_output(monkeypatch):
    fake = _FakeReader()
    monkeypatch.setattr(
        "splitter_mr.server.components.create_reader",
        lambda config, vision_model=None: fake,
    )
    service = PipelineService(_settings())

    result = asyncio.run(
        service.read_document(
            ReadDocumentRequest(
                file_path="hello world",
                reader=VanillaReaderConfiguration(),
                kwargs={"document_name": "doc.txt"},
            )
        )
    )

    assert result.text == "hello world"
    assert result.document_id == "doc-1"
    assert fake.file_path == "hello world"
    assert fake.kwargs["document_name"] == "doc.txt"


def test_split_document_preserves_reader_metadata(monkeypatch):
    fake = _FakeSplitter()
    monkeypatch.setattr(
        "splitter_mr.server.components.create_splitter",
        lambda config, extra_kwargs=None, embedding=None: fake,
    )
    service = PipelineService(_settings())
    output = _reader_output()

    result = asyncio.run(
        service.split_document(
            SplitDocumentRequest(
                reader_output=output,
                splitter=CharacterSplitterConfiguration(chunk_size=5),
            )
        )
    )

    assert result.document_id == "doc-1"
    assert result.metadata == {"k": "v"}
    assert fake.input is output


def test_read_and_split_uses_exact_reader_output(monkeypatch):
    fake_reader = _FakeReader()
    fake_splitter = _FakeSplitter()
    monkeypatch.setattr(
        "splitter_mr.server.components.create_reader",
        lambda config, vision_model=None: fake_reader,
    )
    monkeypatch.setattr(
        "splitter_mr.server.components.create_splitter",
        lambda config, extra_kwargs=None, embedding=None: fake_splitter,
    )
    service = PipelineService(_settings())

    result = asyncio.run(
        service.read_and_split(
            ReadAndSplitRequest(
                file_path="hello world",
                reader=VanillaReaderConfiguration(),
                splitter=CharacterSplitterConfiguration(),
            )
        )
    )

    assert result.chunks == ["hello", "world"]
    assert fake_splitter.input.document_id == "doc-1"


def test_read_document_dispatches_to_worker_thread(monkeypatch):
    fake = _FakeReader()
    monkeypatch.setattr(
        "splitter_mr.server.components.create_reader",
        lambda config, vision_model=None: fake,
    )
    service = PipelineService(_settings())
    main_thread = threading.get_ident()

    asyncio.run(service.read_document(ReadDocumentRequest(file_path="hello world")))

    assert fake.thread_id is not None
    assert fake.thread_id != main_thread


def test_read_document_passes_json_as_positional_file_path(monkeypatch):
    fake = _FakeReader()
    monkeypatch.setattr(
        "splitter_mr.server.components.create_reader",
        lambda config, vision_model=None: fake,
    )
    service = PipelineService(_settings())

    asyncio.run(
        service.read_document(
            ReadDocumentRequest(file_path={"title": "Report", "pages": 3})
        )
    )

    assert '"title"' in fake.file_path
    assert "Report" in fake.file_path


def test_read_document_builds_vision_model_before_reader(monkeypatch):
    captured = {}

    def fake_create_model(config):
        captured["model_config"] = config
        return object()

    def fake_create_reader(config, vision_model=None):
        captured["vision_model"] = vision_model
        return _FakeReader()

    monkeypatch.setattr(
        "splitter_mr.server.components.create_vision_model",
        fake_create_model,
    )
    monkeypatch.setattr(
        "splitter_mr.server.components.create_reader",
        fake_create_reader,
    )
    service = PipelineService(_settings())
    model = OpenAIVisionModelConfiguration(model_name="gpt-4.1")

    asyncio.run(
        service.read_document(ReadDocumentRequest(file_path="hello", model=model))
    )

    assert captured["model_config"] is model
    assert captured["vision_model"] is not None


def test_split_document_forwards_kwargs_and_embedding(monkeypatch):
    captured = {}

    def fake_create_embedding(config):
        captured["embedding_config"] = config
        return object()

    def fake_create_splitter(config, extra_kwargs=None, embedding=None):
        captured["extra_kwargs"] = extra_kwargs
        captured["embedding"] = embedding
        return _FakeSplitter()

    monkeypatch.setattr(
        "splitter_mr.server.components.create_embedding",
        fake_create_embedding,
    )
    monkeypatch.setattr(
        "splitter_mr.server.components.create_splitter",
        fake_create_splitter,
    )
    service = PipelineService(_settings())
    embedding = OpenAIEmbeddingConfiguration(model_name="text-embedding-3-large")

    asyncio.run(
        service.split_document(
            SplitDocumentRequest(
                reader_output=_reader_output(),
                splitter=SemanticSplitterConfiguration(),
                embedding=embedding,
                kwargs={"buffer_size": 2, "chunk_size": 400},
            )
        )
    )

    assert captured["embedding_config"] is embedding
    assert captured["embedding"] is not None
    assert captured["extra_kwargs"]["buffer_size"] == 2


def test_read_and_split_forwards_splitter_kwargs(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "splitter_mr.server.components.create_reader",
        lambda config, vision_model=None: _FakeReader(),
    )

    def fake_create_splitter(config, extra_kwargs=None, embedding=None):
        captured["extra_kwargs"] = extra_kwargs
        captured["config"] = config
        return _FakeSplitter()

    monkeypatch.setattr(
        "splitter_mr.server.components.create_splitter",
        fake_create_splitter,
    )
    service = PipelineService(_settings())

    asyncio.run(
        service.read_and_split(
            ReadAndSplitRequest(
                file_path="hello world",
                splitter=KeywordSplitterConfiguration(patterns=["CHAPTER"]),
                splitter_kwargs={"include_delimiters": "before"},
            )
        )
    )

    assert captured["config"].splitter == "KeywordSplitter"
    assert captured["extra_kwargs"]["include_delimiters"] == "before"


# ---- Error paths ---- #


def test_read_document_translates_reader_config_exception(monkeypatch):
    class BoomReader:
        def read(self, file_path=None, **kwargs):
            raise ReaderConfigException("bad source")

    monkeypatch.setattr(
        "splitter_mr.server.components.create_reader",
        lambda config, vision_model=None: BoomReader(),
    )
    service = PipelineService(_settings())

    with pytest.raises(ServerConfigurationError) as error:
        asyncio.run(service.read_document(ReadDocumentRequest(file_path="hello world")))

    assert "bad source" in error.value.message


def test_split_document_translates_splitter_config_exception(monkeypatch):
    class BoomSplitter:
        def split(self, reader_output):
            raise SplitterConfigException("bad overlap")

    monkeypatch.setattr(
        "splitter_mr.server.components.create_splitter",
        lambda config, extra_kwargs=None, embedding=None: BoomSplitter(),
    )
    service = PipelineService(_settings())

    with pytest.raises(ServerConfigurationError) as error:
        asyncio.run(
            service.split_document(
                SplitDocumentRequest(
                    reader_output=_reader_output(),
                    splitter=CharacterSplitterConfiguration(),
                )
            )
        )

    assert "bad overlap" in error.value.message


def test_read_document_file_source_denied_without_root(tmp_path: Path):
    document = tmp_path / "doc.txt"
    document.write_text("hello", encoding="utf-8")
    service = PipelineService(_settings())

    with pytest.raises(ServerAccessDeniedError):
        asyncio.run(
            service.read_document(
                ReadDocumentRequest(
                    file_path=str(document),
                    reader=VanillaReaderConfiguration(),
                )
            )
        )


def test_read_document_rejects_oversized_text():
    service = PipelineService(_settings(max_body_bytes=4))

    with pytest.raises(ServerPayloadTooLargeError):
        asyncio.run(service.read_document(ReadDocumentRequest(file_path="too-big")))


def test_read_document_missing_reader_extra(monkeypatch):
    monkeypatch.setattr(
        "splitter_mr.server.components.create_reader",
        lambda config, vision_model=None: (_ for _ in ()).throw(
            ServerComponentUnavailableError("missing extra")
        ),
    )
    service = PipelineService(_settings())

    with pytest.raises(ServerComponentUnavailableError):
        asyncio.run(service.read_document(ReadDocumentRequest(file_path="hello")))


def test_read_and_split_real_vanilla_character_pipeline():
    service = PipelineService(_settings())

    result = asyncio.run(
        service.read_and_split(
            ReadAndSplitRequest(
                file_path="Lorem ipsum dolor sit amet.",
                reader=VanillaReaderConfiguration(),
                kwargs={
                    "document_name": "lorem.txt",
                    "document_id": "fixed-id",
                    "metadata": {"origin": "test"},
                },
                splitter=CharacterSplitterConfiguration(chunk_size=12, chunk_overlap=0),
            )
        )
    )

    assert result.document_id == "fixed-id"
    assert result.document_name == "lorem.txt"
    assert result.chunks
    assert len(result.chunks) == len(result.chunk_id)


def test_read_document_file_source_under_allowed_root(monkeypatch, tmp_path: Path):
    document = tmp_path / "note.txt"
    document.write_text("from-disk", encoding="utf-8")
    fake = _FakeReader()
    monkeypatch.setattr(
        "splitter_mr.server.components.create_reader",
        lambda config, vision_model=None: fake,
    )
    service = PipelineService(_settings(allowed_root=tmp_path))

    result = asyncio.run(
        service.read_document(
            ReadDocumentRequest(
                file_path=str(document),
                reader=VanillaReaderConfiguration(),
                kwargs={"document_name": "note.txt"},
            )
        )
    )

    assert result.document_id == "doc-1"
    assert Path(fake.file_path) == document.resolve()
    assert fake.kwargs["document_name"] == "note.txt"


def test_read_document_rejects_textract_inline_text():
    service = PipelineService(_settings())

    with pytest.raises(ServerConfigurationError) as error:
        asyncio.run(
            service.read_document(
                ReadDocumentRequest(
                    file_path="hello world",
                    reader=TextractReaderConfiguration(),
                )
            )
        )

    assert "does not support text" in error.value.message


def test_read_document_url_denied_when_disabled():
    service = PipelineService(_settings(allow_urls=False))

    with pytest.raises(ServerAccessDeniedError):
        asyncio.run(
            service.read_document(
                ReadDocumentRequest(file_path="https://example.com/manual.pdf")
            )
        )
