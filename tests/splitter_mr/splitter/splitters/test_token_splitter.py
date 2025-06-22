from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.schema import ReaderOutput
from splitter_mr.splitter import TokenSplitter


@pytest.fixture
def simple_reader_output():
    return ReaderOutput(
        text="The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
        document_name="pangrams.txt",
        document_path="/https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/pangrams.txt",
        document_id="doc1",
        conversion_method="text",
        reader_method="plain",
        ocr_method=None,
        metadata={},
    )


def test_split_tiktoken(monkeypatch, simple_reader_output):
    # Patch the RecursiveCharacterTextSplitter
    mock_splitter = MagicMock()
    mock_splitter.split_text.return_value = [
        "The quick brown fox jumps",
        "over the lazy dog. Pack my",
        "box with five dozen liquor jugs.",
    ]
    with patch(
        "splitter_mr.splitter.splitters.token_splitter.RecursiveCharacterTextSplitter"
    ) as mock_class:
        mock_class.from_tiktoken_encoder.return_value = mock_splitter
        splitter = TokenSplitter(chunk_size=5, model_name="tiktoken/cl100k_base")
        output = splitter.split(simple_reader_output)
        assert output.chunks == mock_splitter.split_text.return_value
        assert output.split_method == "token_splitter"
        assert output.split_params["model_name"] == "tiktoken/cl100k_base"


def test_split_spacy(monkeypatch, simple_reader_output):
    # Patch spacy.util.is_package, spacy.cli.download, SpacyTextSplitter
    with patch("splitter_mr.splitter.splitters.token_splitter.spacy") as mock_spacy:
        mock_spacy.util.is_package.return_value = True
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
        ]
        with patch(
            "splitter_mr.splitter.splitters.token_splitter.SpacyTextSplitter",
            return_value=mock_splitter,
        ):
            splitter = TokenSplitter(chunk_size=50, model_name="spacy/en_core_web_sm")
            output = splitter.split(simple_reader_output)
            assert output.chunks == mock_splitter.split_text.return_value


def test_split_nltk(monkeypatch, simple_reader_output):
    # Patch nltk.data.find, nltk.download, NLTKTextSplitter
    with patch("splitter_mr.splitter.splitters.token_splitter.nltk") as mock_nltk:
        mock_nltk.data.find.side_effect = None  # No exception = model exists
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
        ]
        with patch(
            "splitter_mr.splitter.splitters.token_splitter.NLTKTextSplitter",
            return_value=mock_splitter,
        ):
            splitter = TokenSplitter(chunk_size=50, model_name="nltk/punkt")
            output = splitter.split(simple_reader_output)
            assert output.chunks == mock_splitter.split_text.return_value


def test_split_invalid_tokenizer(simple_reader_output):
    splitter = TokenSplitter(chunk_size=10, model_name="unknown/foobar")
    with pytest.raises(ValueError, match="Unsupported tokenizer 'unknown'"):
        splitter.split(simple_reader_output)


def test_spacy_model_download_if_not_present(simple_reader_output):
    # is_package returns False, so download should be attempted
    with patch("splitter_mr.splitter.splitters.token_splitter.spacy") as mock_spacy:
        mock_spacy.util.is_package.return_value = False
        mock_spacy.cli.download.return_value = None  # pretend download works
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["chunk1", "chunk2"]
        with patch(
            "splitter_mr.splitter.splitters.token_splitter.SpacyTextSplitter",
            return_value=mock_splitter,
        ):
            splitter = TokenSplitter(chunk_size=50, model_name="spacy/en_core_web_sm")
            output = splitter.split(simple_reader_output)
            assert output.chunks == ["chunk1", "chunk2"]
        mock_spacy.cli.download.assert_called_once_with("en_core_web_sm")
