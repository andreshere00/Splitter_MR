from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.embedding.embeddings.anthropic_embedding import AnthropicEmbedding


# ---- Setup/fixtures ----
@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Clear VOYAGE_API_KEY for each test (isolate env)"""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)


@pytest.fixture
def mock_client():
    """Mock voyageai.Client instance"""
    return MagicMock()


@pytest.fixture
def mock_voyage_client_class(mock_client):
    """Patch voyageai.Client to return mock_client"""
    with patch(
        "splitter_mr.embedding.embeddings.anthropic_embedding.voyageai.Client",
        return_value=mock_client,
    ):
        yield mock_client


# ---- Initialization ----


def test_init_with_api_key(mock_voyage_client_class):
    embed = AnthropicEmbedding(api_key="abc123")
    assert embed.client is mock_voyage_client_class
    assert embed.model_name == "voyage-3.5"
    assert embed.default_input_type == "document"


def test_init_env(monkeypatch, mock_voyage_client_class):
    monkeypatch.setenv("VOYAGE_API_KEY", "env-key")
    embed = AnthropicEmbedding()
    assert embed.client is mock_voyage_client_class


def test_init_missing_key():
    with pytest.raises(ValueError) as e:
        AnthropicEmbedding(api_key=None)
    assert "Voyage API key not provided" in str(e.value)


# ---- get_client ----


def test_get_client_returns_client(mock_voyage_client_class):
    embed = AnthropicEmbedding(api_key="key")
    assert embed.get_client() is mock_voyage_client_class


# ---- _ensure_input_type ----


def test_ensure_input_type_sets_default(mock_voyage_client_class):
    embed = AnthropicEmbedding(api_key="key", default_input_type="document")
    # No input_type
    params = embed._ensure_input_type({})
    assert params["input_type"] == "document"
    # Already set
    params2 = embed._ensure_input_type({"input_type": "query"})
    assert params2["input_type"] == "query"


def test_ensure_input_type_empty_params(mock_voyage_client_class):
    embed = AnthropicEmbedding(api_key="key", default_input_type=None)
    params = embed._ensure_input_type({})
    assert "input_type" not in params


# ---- embed_text ----


def test_embed_text_success(mock_voyage_client_class):
    embed = AnthropicEmbedding(api_key="key")
    # Mock Voyage .embed() return object
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1, 0.2, 0.3]]
    embed.client.embed.return_value = mock_result

    vec = embed.embed_text("hello world")
    embed.client.embed.assert_called_once()
    assert vec == [0.1, 0.2, 0.3]


@pytest.mark.parametrize("bad_text", ["", "   ", None, 42, [], {}])
def test_embed_text_invalid_input(bad_text, mock_voyage_client_class):
    embed = AnthropicEmbedding(api_key="key")
    with pytest.raises(ValueError):
        embed.embed_text(bad_text)


def test_embed_text_voyage_empty_response(mock_voyage_client_class):
    embed = AnthropicEmbedding(api_key="key")
    # No embeddings
    mock_result = MagicMock()
    mock_result.embeddings = []
    embed.client.embed.return_value = mock_result
    with pytest.raises(RuntimeError):
        embed.embed_text("hello world")


def test_embed_text_voyage_malformed(mock_voyage_client_class):
    embed = AnthropicEmbedding(api_key="key")
    # .embeddings missing or not a list
    mock_result = MagicMock()
    del mock_result.embeddings
    embed.client.embed.return_value = mock_result
    with pytest.raises(RuntimeError):
        embed.embed_text("hello world")


def test_embed_text_voyage_invalid_vector(mock_voyage_client_class):
    embed = AnthropicEmbedding(api_key="key")
    # embedding not a list
    mock_result = MagicMock()
    mock_result.embeddings = [None]
    embed.client.embed.return_value = mock_result
    with pytest.raises(RuntimeError):
        embed.embed_text("hello world")


# ---- embed_documents ----


def test_embed_documents_success(mock_voyage_client_class):
    embed = AnthropicEmbedding(api_key="key")
    texts = ["foo", "bar"]
    mock_result = MagicMock()
    mock_result.embeddings = [[1, 2], [3, 4]]
    embed.client.embed.return_value = mock_result
    vecs = embed.embed_documents(texts)
    embed.client.embed.assert_called_once_with(
        texts, model="voyage-3.5", input_type="document"
    )
    assert vecs == [[1, 2], [3, 4]]


@pytest.mark.parametrize("bad_texts", [[], None])
def test_embed_documents_empty_list(bad_texts, mock_voyage_client_class):
    embed = AnthropicEmbedding(api_key="key")
    with pytest.raises(ValueError):
        embed.embed_documents(bad_texts)


def test_embed_documents_non_string_items(mock_voyage_client_class):
    embed = AnthropicEmbedding(api_key="key")
    with pytest.raises(ValueError):
        embed.embed_documents(["hello", None])


def test_embed_documents_blank_string(mock_voyage_client_class):
    embed = AnthropicEmbedding(api_key="key")
    with pytest.raises(ValueError):
        embed.embed_documents(["hello", "   "])


def test_embed_documents_voyage_empty(mock_voyage_client_class):
    embed = AnthropicEmbedding(api_key="key")
    mock_result = MagicMock()
    mock_result.embeddings = []
    embed.client.embed.return_value = mock_result
    with pytest.raises(RuntimeError):
        embed.embed_documents(["foo"])


def test_embed_documents_voyage_missing_attr(mock_voyage_client_class):
    embed = AnthropicEmbedding(api_key="key")
    mock_result = MagicMock()
    del mock_result.embeddings
    embed.client.embed.return_value = mock_result
    with pytest.raises(RuntimeError):
        embed.embed_documents(["foo"])


def test_embed_documents_wrong_number_embeddings(mock_voyage_client_class):
    embed = AnthropicEmbedding(api_key="key")
    mock_result = MagicMock()
    mock_result.embeddings = [[1, 2]]  # Only 1 embedding, but 2 inputs
    embed.client.embed.return_value = mock_result
    with pytest.raises(RuntimeError):
        embed.embed_documents(["foo", "bar"])


# --- Customization for model_name & input_type ---


def test_custom_model_and_input_type(mock_voyage_client_class):
    embed = AnthropicEmbedding(
        api_key="key", model_name="voyage-3-large", default_input_type="query"
    )
    mock_result = MagicMock()
    mock_result.embeddings = [[0.5, 0.6]]
    embed.client.embed.return_value = mock_result
    vec = embed.embed_text("hey", input_type="custom")
    embed.client.embed.assert_called_once_with(
        ["hey"], model="voyage-3-large", input_type="custom"
    )
    assert vec == [0.5, 0.6]
