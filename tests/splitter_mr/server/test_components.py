import pytest

from splitter_mr.server.components import (
    SEMANTIC_SPLITTER_NAME,
    SPLITTER_FACTORIES,
    create_embedding,
    create_reader,
    create_splitter,
    create_vision_model,
    list_components,
)
from splitter_mr.server.enums import ComponentStatus
from splitter_mr.server.exceptions import (
    ServerComponentUnavailableError,
    ServerConfigurationError,
)
from splitter_mr.server.schemas import (
    CharacterSplitterConfiguration,
    KeywordSplitterConfiguration,
    MarkItDownReaderConfiguration,
    OpenAIEmbeddingConfiguration,
    OpenAIVisionModelConfiguration,
    SemanticSplitterConfiguration,
    VanillaReaderConfiguration,
)
from splitter_mr.splitter import CharacterSplitter, KeywordSplitter

# ---- Mocks, fixtures & helpers ---- #


def _config_for(name: str):
    from splitter_mr.server import schemas as server_schemas

    payload = {"splitter": name}
    if name == "KeywordSplitter":
        payload["patterns"] = ["CHAPTER"]
    model = getattr(server_schemas, f"{name}Configuration")
    return model.model_validate(payload)


# ---- Happy path ---- #


def test_create_splitter_resolves_each_supported_class():
    for name, expected in SPLITTER_FACTORIES.items():
        if name == SEMANTIC_SPLITTER_NAME:
            continue
        splitter = create_splitter(_config_for(name))
        assert isinstance(splitter, expected)


def test_create_reader_instantiates_vanilla_reader():
    reader = create_reader(VanillaReaderConfiguration())

    assert reader.__class__.__name__ == "VanillaReader"


def test_list_components_includes_semantic_splitter_as_supported():
    catalog = list_components()
    semantic = next(
        item for item in catalog.splitters if item.name == SEMANTIC_SPLITTER_NAME
    )

    assert semantic.supported is True
    assert semantic.status is ComponentStatus.AVAILABLE
    assert semantic.available is True
    assert semantic.extra == "multimodal"
    assert "embedding" in (semantic.limitation or "")


def test_list_components_marks_vanilla_reader_available():
    catalog = list_components()
    vanilla = next(
        item for item in catalog.readers if item.name.value == "VanillaReader"
    )

    assert vanilla.available is True
    assert vanilla.status is ComponentStatus.AVAILABLE
    assert vanilla.extra is None
    assert vanilla.compatible_vision_models


def test_list_components_includes_vision_models():
    catalog = list_components()
    openai = next(
        item for item in catalog.vision_models if item.name.value == "OpenAIVisionModel"
    )

    assert openai.extra == "multimodal"
    assert openai.configuration_schema == "OpenAIVisionModelConfiguration"
    assert "VanillaReader" in [item.value for item in openai.compatible_readers]


def test_list_components_includes_embeddings():
    catalog = list_components()
    openai = next(
        item for item in catalog.embeddings if item.name.value == "OpenAIEmbedding"
    )

    assert openai.extra == "multimodal"
    assert openai.configuration_schema == "OpenAIEmbeddingConfiguration"


def test_create_splitter_merges_patterns_from_kwargs():
    splitter = create_splitter(
        KeywordSplitterConfiguration(),
        extra_kwargs={"patterns": ["CHAPTER"], "include_delimiters": "both"},
    )

    assert isinstance(splitter, KeywordSplitter)
    assert splitter.include_delimiters == "both"


def test_create_semantic_splitter_injects_embedding():
    class FakeEmbedding:
        def embed_documents(self, texts):
            return [[0.1, 0.2] for _ in texts]

    splitter = create_splitter(
        SemanticSplitterConfiguration(buffer_size=0, chunk_size=10),
        embedding=FakeEmbedding(),
    )

    assert splitter.buffer_size == 0
    assert splitter.chunk_size == 10
    assert isinstance(splitter.embedding, FakeEmbedding)


# ---- Error paths ---- #


def test_create_splitter_rejects_unknown_name():
    config = CharacterSplitterConfiguration()
    config.__dict__["splitter"] = "DoesNotExist"

    with pytest.raises(ServerComponentUnavailableError) as error:
        create_splitter(config)

    assert "Unsupported splitter" in error.value.message


def test_create_splitter_rejects_keyword_splitter_without_patterns():
    with pytest.raises(ServerConfigurationError) as error:
        create_splitter(KeywordSplitterConfiguration())

    assert "requires patterns" in error.value.message
    with pytest.raises(ServerConfigurationError):
        create_splitter(
            CharacterSplitterConfiguration(),
            extra_kwargs={"patterns": ["CHAPTER"]},
        )


def test_create_semantic_splitter_without_embedding_raises():
    with pytest.raises(ServerConfigurationError) as error:
        create_splitter(SemanticSplitterConfiguration())

    assert "requires a top-level embedding" in error.value.message


def test_create_reader_reports_missing_extra(monkeypatch):
    import importlib

    from splitter_mr.reader import readers as readers_mod

    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name == ".markitdown_reader":
            raise ModuleNotFoundError("No module named 'markitdown'")
        return real_import(name, package)

    monkeypatch.setattr(readers_mod.importlib, "import_module", fake_import)

    with pytest.raises(ServerComponentUnavailableError) as error:
        create_reader(MarkItDownReaderConfiguration())

    assert "markitdown" in error.value.message
    assert "pip install" in error.value.message


def test_create_vision_model_reports_missing_extra(monkeypatch):
    import importlib

    from splitter_mr.model import models as models_mod

    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name == ".openai_model":
            raise ModuleNotFoundError("No module named 'openai'")
        return real_import(name, package)

    monkeypatch.setattr(models_mod.importlib, "import_module", fake_import)

    with pytest.raises(ServerComponentUnavailableError) as error:
        create_vision_model(OpenAIVisionModelConfiguration())

    assert "multimodal" in error.value.message
    assert "pip install" in error.value.message


def test_create_vision_model_unwraps_secret_key(monkeypatch):
    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "splitter_mr.model.models.__getattr__",
        lambda name: FakeModel,
        raising=False,
    )
    import splitter_mr.model.models as models_mod

    monkeypatch.setattr(models_mod, "OpenAIVisionModel", FakeModel)

    create_vision_model(
        OpenAIVisionModelConfiguration(api_key="sk-secret", model_name="gpt-4.1")
    )

    assert captured["api_key"] == "sk-secret"
    assert captured["model_name"] == "gpt-4.1"


def test_create_embedding_reports_missing_extra(monkeypatch):
    import importlib

    from splitter_mr.embedding import embeddings as embeddings_mod

    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name == ".openai_embedding":
            raise ModuleNotFoundError("No module named 'openai'")
        return real_import(name, package)

    monkeypatch.setattr(embeddings_mod.importlib, "import_module", fake_import)

    with pytest.raises(ServerComponentUnavailableError) as error:
        create_embedding(OpenAIEmbeddingConfiguration())

    assert "multimodal" in error.value.message
    assert "pip install" in error.value.message


def test_create_embedding_unwraps_secret_key(monkeypatch):
    captured = {}

    class FakeEmbedding:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import splitter_mr.embedding.embeddings as embeddings_mod

    monkeypatch.setattr(embeddings_mod, "OpenAIEmbedding", FakeEmbedding)

    create_embedding(
        OpenAIEmbeddingConfiguration(
            api_key="sk-embed",
            model_name="text-embedding-3-large",
        )
    )

    assert captured["api_key"] == "sk-embed"
    assert captured["model_name"] == "text-embedding-3-large"


# ---- Edge cases ---- #


def test_create_keyword_splitter_forwards_patterns():
    splitter = create_splitter(
        KeywordSplitterConfiguration(patterns=["SECTION"], flags=2)
    )

    assert isinstance(splitter, KeywordSplitter)
    assert splitter.flags == 2


def test_create_character_splitter_forwards_overlap():
    splitter = create_splitter(
        CharacterSplitterConfiguration(chunk_size=50, chunk_overlap=5)
    )

    assert isinstance(splitter, CharacterSplitter)
    assert splitter.chunk_size == 50
    assert splitter.chunk_overlap == 5
