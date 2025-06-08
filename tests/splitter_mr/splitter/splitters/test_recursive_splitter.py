from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.schema import ReaderOutput
from splitter_mr.splitter import RecursiveCharacterSplitter

# Helpers


@pytest.fixture
def reader_output():
    return ReaderOutput(
        text="A long test text that should be split recursively.",
        document_name="sample.txt",
        document_path="/tmp/sample.txt",
        document_id="123",
        conversion_method="text",
        ocr_method=None,
    )


# Tests cases


def test_recursive_character_splitter_instantiates_and_calls_splitter(reader_output):
    with patch(
        "splitter_mr.splitter.splitters.recursive_splitter.RecursiveCharacterTextSplitter"
    ) as MockSplitter:
        # Setup the mock to return fake chunks as page_content
        mock_splitter = MockSplitter.return_value
        mock_doc1 = MagicMock(page_content="Chunk 1")
        mock_doc2 = MagicMock(page_content="Chunk 2")
        mock_splitter.create_documents.return_value = [mock_doc1, mock_doc2]

        splitter = RecursiveCharacterSplitter(
            chunk_size=10, chunk_overlap=2, separators=["."]
        )
        result = splitter.split(reader_output)

        # Check instantiation
        MockSplitter.assert_called_once_with(
            chunk_size=10, chunk_overlap=2, separators=["."]
        )
        # Check method called
        mock_splitter.create_documents.assert_called_once_with([reader_output.text])

        # Check output structure
        assert hasattr(result, "chunks")
        assert result.chunks == ["Chunk 1", "Chunk 2"]
        assert hasattr(result, "split_method")
        assert result.split_method == "recursive_character_splitter"
        assert result.split_params["chunk_size"] == 10
        assert result.split_params["chunk_overlap"] == 2
        assert result.split_params["separators"] == ["."]
        for field in [
            "chunks",
            "chunk_id",
            "document_name",
            "document_path",
            "document_id",
            "conversion_method",
            "ocr_method",
            "split_method",
            "split_params",
            "metadata",
        ]:
            assert hasattr(result, field)


def test_empty_text():
    with patch(
        "splitter_mr.splitter.splitters.recursive_splitter.RecursiveCharacterTextSplitter"
    ) as MockSplitter:
        mock_splitter = MockSplitter.return_value
        mock_splitter.create_documents.return_value = []

        splitter = RecursiveCharacterSplitter(
            chunk_size=10, chunk_overlap=0, separators=["."]
        )
        reader_output = ReaderOutput(text="")
        result = splitter.split(reader_output)
        assert result.chunks == []
