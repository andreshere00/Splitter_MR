import sys
from unittest.mock import MagicMock

import pytest

from splitter_mr.embedding.embeddings.gemini_embedding import GeminiEmbedding

# ---- Helpers, mocks & fixtures ---- #

DUMMY_API_KEY = "test-gemini-api-key"


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Ensure GEMINI_API_KEY is not set for any test unless explicitly needed."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def patch_genai(monkeypatch):
    """Patch the google.generativeai and genai.Client for lazy import."""
    fake_client = MagicMock(name="Client")
    fake_models = MagicMock(name="Models")
    fake_client.models = fake_models

    fake_genai = MagicMock()
    fake_genai.Client.return_value = fake_client

    # Insert into sys.modules for __import__ and importlib
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google", MagicMock(generativeai=fake_genai))
    return fake_genai, fake_client, fake_models


# ---- Test cases ---- #


def test_import_error(monkeypatch):
    """Raises ImportError with install message if extra is missing."""
    monkeypatch.setitem(sys.modules, "google.genai", None)
    monkeypatch.setitem(sys.modules, "google", None)
    # Remove module for clean import
    import importlib

    importlib.reload(sys.modules["splitter_mr.embedding.embeddings.gemini_embedding"])
    with pytest.raises(ImportError) as e:
        GeminiEmbedding(api_key=DUMMY_API_KEY)
    assert "requires the 'multimodal' extra" in str(e.value)


def test_api_key_env(monkeypatch):
    """Uses GEMINI_API_KEY from environment if not provided."""
    patch_genai(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_API_KEY)
    embedder = GeminiEmbedding()
    assert embedder.api_key == DUMMY_API_KEY
    assert embedder.model_name == "models/embedding-001"


def test_api_key_missing(monkeypatch):
    """Raises ValueError if no API key is supplied or found in env."""
    patch_genai(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError) as e:
        GeminiEmbedding()
    assert "GEMINI_API_KEY" in str(e.value)


def test_get_client(monkeypatch):
    """get_client returns the genai.Client instance."""
    _, fake_client, _ = patch_genai(monkeypatch)
    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    assert embedder.get_client() is fake_client


def test_embed_text_success(monkeypatch):
    """Returns embedding for valid text."""
    _, _, fake_models = patch_genai(monkeypatch)
    fake_result = MagicMock(embedding=[0.1, 0.2, 0.3])
    fake_models.embed_content.return_value = fake_result

    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    vec = embedder.embed_text("Hello world!")
    assert vec == [0.1, 0.2, 0.3]
    fake_models.embed_content.assert_called_once_with(
        model="models/embedding-001", content="Hello world!"
    )


def test_embed_text_invalid(monkeypatch):
    """Raises ValueError on empty or non-string text."""
    patch_genai(monkeypatch)
    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    for bad in [None, "", "   ", 123]:
        with pytest.raises(ValueError):
            embedder.embed_text(bad)


def test_embed_text_missing_embedding(monkeypatch):
    """Raises RuntimeError if embedding field is missing from result."""
    _, _, fake_models = patch_genai(monkeypatch)
    fake_models.embed_content.return_value = MagicMock(embedding=None)
    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    with pytest.raises(RuntimeError) as e:
        embedder.embed_text("something")
    assert "no 'embedding' field" in str(e.value)


def test_embed_text_api_error(monkeypatch):
    """Raises RuntimeError on Gemini API error."""
    _, _, fake_models = patch_genai(monkeypatch)
    fake_models.embed_content.side_effect = Exception("Gemini down")
    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    with pytest.raises(RuntimeError) as e:
        embedder.embed_text("test")
    assert "Failed to get embedding from Gemini: Gemini down" in str(e.value)


def test_embed_documents_success(monkeypatch):
    """Returns list of embeddings for valid input list."""
    _, _, fake_models = patch_genai(monkeypatch)
    fake_models.embed_content.return_value = MagicMock(
        embeddings=[[1.0, 2.0], [3.0, 4.0]]
    )
    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    vecs = embedder.embed_documents(["foo", "bar"])
    assert vecs == [[1.0, 2.0], [3.0, 4.0]]
    fake_models.embed_content.assert_called_once_with(
        model="models/embedding-001", content=["foo", "bar"]
    )


def test_embed_documents_invalid_input(monkeypatch):
    """Raises ValueError for non-list or empty input or any bad string."""
    patch_genai(monkeypatch)
    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    bad_cases = [None, 123, [], ["", " "], ["good", ""], ["good", 123], [123, 456]]
    for bad in bad_cases:
        with pytest.raises(ValueError):
            embedder.embed_documents(bad)


def test_embed_documents_missing_embeddings(monkeypatch):
    """Raises RuntimeError if .embeddings missing from Gemini result."""
    _, _, fake_models = patch_genai(monkeypatch)
    fake_models.embed_content.return_value = MagicMock(embeddings=None)
    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    with pytest.raises(RuntimeError) as e:
        embedder.embed_documents(["A", "B"])
    assert "no 'embeddings' field" in str(e.value)


def test_embed_documents_api_error(monkeypatch):
    """Raises RuntimeError if Gemini API errors out."""
    _, _, fake_models = patch_genai(monkeypatch)
    fake_models.embed_content.side_effect = Exception("fail")
    embedder = GeminiEmbedding(api_key=DUMMY_API_KEY)
    with pytest.raises(RuntimeError) as e:
        embedder.embed_documents(["A", "B"])
    assert "Failed to get document embeddings from Gemini: fail" in str(e.value)
