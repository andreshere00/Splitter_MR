from unittest.mock import MagicMock

import pytest

from splitter_mr.embedding.embeddings.gemini_embedding import GeminiEmbedding

# ---- Helpers, mocks & fixtures ---- #

DUMMY_API_KEY = "test-gemini-api-key"


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def patch_genai(monkeypatch):
    fake_client = MagicMock(name="Client")
    fake_models = MagicMock(name="Models")
    fake_client.models = fake_models
    # Correct patch: patch the actual import path your class uses
    monkeypatch.setattr(
        "splitter_mr.embedding.embeddings.gemini_embedding.genai.Client",
        lambda api_key: fake_client,
    )
    return fake_client, fake_models


# ---- Test cases ---- #


def test_api_key_env(monkeypatch):
    patch_genai(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_API_KEY)
    embedder = GeminiEmbedding()
    assert embedder.api_key == DUMMY_API_KEY
    assert embedder.model_name == "models/embedding-001"


def test_api_key_missing(monkeypatch):
    patch_genai(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError) as e:
        GeminiEmbedding()
    assert "GEMINI_API_KEY" in str(e.value)


def test_get_client(monkeypatch):
    fake_client, _ = patch_genai(monkeypatch)
    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    assert embedder.get_client() is fake_client


def test_embed_text_success(monkeypatch):
    _, fake_models = patch_genai(monkeypatch)
    fake_result = MagicMock(embedding=[0.1, 0.2, 0.3])
    fake_models.embed_content.return_value = fake_result

    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    vec = embedder.embed_text("Hello world!")
    assert vec == [0.1, 0.2, 0.3]
    fake_models.embed_content.assert_called_once_with(
        model="models/embedding-001", contents="Hello world!"
    )  # CHANGED: content → contents


def test_embed_text_invalid(monkeypatch):
    patch_genai(monkeypatch)
    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    for bad in [None, "", "   ", 123]:
        with pytest.raises(ValueError):
            embedder.embed_text(bad)


def test_embed_text_missing_embedding(monkeypatch):
    _, fake_models = patch_genai(monkeypatch)
    fake_models.embed_content.return_value = MagicMock(embedding=None)
    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    with pytest.raises(RuntimeError) as e:
        embedder.embed_text("something")
    assert "no 'embedding' field" in str(e.value)


def test_embed_text_api_error(monkeypatch):
    _, fake_models = patch_genai(monkeypatch)
    fake_models.embed_content.side_effect = Exception("Gemini down")
    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    with pytest.raises(RuntimeError) as e:
        embedder.embed_text("test")
    assert "Failed to get embedding from Gemini: Gemini down" in str(e.value)


def test_embed_documents_success(monkeypatch):
    _, fake_models = patch_genai(monkeypatch)
    fake_models.embed_content.return_value = MagicMock(
        embeddings=[[1.0, 2.0], [3.0, 4.0]]
    )
    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    vecs = embedder.embed_documents(["foo", "bar"])
    assert vecs == [[1.0, 2.0], [3.0, 4.0]]
    fake_models.embed_content.assert_called_once_with(
        model="models/embedding-001", contents=["foo", "bar"]
    )  # CHANGED: content → contents


def test_embed_documents_invalid_input(monkeypatch):
    patch_genai(monkeypatch)
    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    bad_cases = [None, 123, [], ["", " "], ["good", ""], ["good", 123], [123, 456]]
    for bad in bad_cases:
        with pytest.raises(ValueError):
            embedder.embed_documents(bad)


def test_embed_documents_missing_embeddings(monkeypatch):
    _, fake_models = patch_genai(monkeypatch)
    fake_models.embed_content.return_value = MagicMock(embeddings=None)
    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    with pytest.raises(RuntimeError) as e:
        embedder.embed_documents(["A", "B"])
    assert "no 'embeddings' field" in str(e.value)


def test_embed_documents_api_error(monkeypatch):
    _, fake_models = patch_genai(monkeypatch)
    fake_models.embed_content.side_effect = Exception("fail")
    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    with pytest.raises(RuntimeError) as e:
        embedder.embed_documents(["A", "B"])
    assert "Failed to get document embeddings from Gemini: fail" in str(e.value)
