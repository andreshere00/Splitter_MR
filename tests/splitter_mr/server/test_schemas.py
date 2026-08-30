import pytest
from pydantic import SecretStr, ValidationError

from splitter_mr.server.schemas import (
    CharacterSplitterConfiguration,
    HeaderSplitterConfiguration,
    KeywordSplitterConfiguration,
    OpenAIEmbeddingConfiguration,
    OpenAIVisionModelConfiguration,
    ReadAndSplitRequest,
    ReadDocumentRequest,
    RecursiveCharacterSplitterConfiguration,
    RowColumnSplitterConfiguration,
    SplitDocumentRequest,
    TextractReaderConfiguration,
    VanillaReaderConfiguration,
    WordSplitterConfiguration,
)

# ---- Mocks, fixtures & helpers ---- #


# ---- Happy path ---- #


def test_read_document_request_accepts_text_and_vanilla_reader():
    request = ReadDocumentRequest.model_validate(
        {
            "file_path": "Lorem ipsum dolor sit amet.",
            "reader": {"reader": "VanillaReader"},
            "kwargs": {"document_name": "lorem.txt"},
        }
    )

    assert request.file_path == "Lorem ipsum dolor sit amet."
    assert request.reader.reader == "VanillaReader"
    assert request.kwargs["document_name"] == "lorem.txt"
    assert request.model is None


@pytest.mark.parametrize(
    "splitter",
    [
        "CharacterSplitter",
        "WordSplitter",
        "SentenceSplitter",
        "ParagraphSplitter",
        "RecursiveCharacterSplitter",
        "KeywordSplitter",
        "TokenSplitter",
        "PagedSplitter",
        "RowColumnSplitter",
        "RecursiveJSONSplitter",
        "HTMLTagSplitter",
        "HeaderSplitter",
        "CodeSplitter",
    ],
)
def test_splitter_configuration_accepts_supported_discriminator(splitter):
    payload = {"splitter": splitter}
    if splitter == "KeywordSplitter":
        payload["patterns"] = ["CHAPTER"]
    model = SplitDocumentRequest.model_validate(
        {
            "reader_output": {
                "text": "hello world",
                "document_path": "",
            },
            "splitter": payload,
        }
    )

    assert model.splitter.splitter == splitter


def test_split_document_request_accepts_extra_constructor_kwargs():
    model = SplitDocumentRequest.model_validate(
        {
            "reader_output": {
                "text": "CHAPTER 1 Once upon a time.",
                "document_path": "",
            },
            "splitter": {"splitter": "KeywordSplitter"},
            "kwargs": {
                "patterns": ["CHAPTER"],
                "include_delimiters": "before",
                "chunk_size": 5000,
            },
        }
    )

    assert model.kwargs["patterns"] == ["CHAPTER"]
    assert model.kwargs["include_delimiters"] == "before"
    assert model.kwargs["chunk_size"] == 5000


def test_split_document_request_accepts_semantic_splitter_with_embedding():
    model = SplitDocumentRequest.model_validate(
        {
            "reader_output": {"text": "hello world", "document_path": ""},
            "splitter": {"splitter": "SemanticSplitter", "buffer_size": 1},
            "embedding": {
                "embedding": "OpenAIEmbedding",
                "model_name": "text-embedding-3-large",
            },
            "kwargs": {"chunk_size": 800},
        }
    )

    assert model.splitter.splitter == "SemanticSplitter"
    assert model.embedding.embedding == "OpenAIEmbedding"
    assert model.kwargs["chunk_size"] == 800


def test_keyword_splitter_configuration_accepts_named_patterns():
    config = KeywordSplitterConfiguration.model_validate(
        {"splitter": "KeywordSplitter", "patterns": {"chapter": "CHAPTER\\s+\\d+"}}
    )

    assert config.patterns["chapter"] == "CHAPTER\\s+\\d+"


def test_read_and_split_request_defaults_reader_and_splitter():
    request = ReadAndSplitRequest.model_validate({"file_path": "hello world"})

    assert request.reader.reader == "VanillaReader"
    assert request.splitter.splitter == "RecursiveCharacterSplitter"


def test_json_schema_includes_descriptions_discriminators_and_examples():
    schema = ReadDocumentRequest.model_json_schema()

    file_path = schema["properties"]["file_path"]
    assert "description" in file_path
    character = CharacterSplitterConfiguration.model_json_schema()
    assert "description" in character["properties"]["chunk_size"]
    assert character["properties"]["chunk_size"]["minimum"] == 1
    openai = OpenAIVisionModelConfiguration.model_json_schema()
    assert openai["properties"]["model"]["description"]
    embedding = OpenAIEmbeddingConfiguration.model_json_schema()
    assert embedding["properties"]["embedding"]["description"]
    assert "api_key" not in str(embedding.get("examples", []))


def test_openai_vision_model_accepts_optional_secret_key():
    config = OpenAIVisionModelConfiguration.model_validate(
        {"model": "OpenAIVisionModel", "api_key": "sk-test", "model_name": "gpt-4.1"}
    )

    assert isinstance(config.api_key, SecretStr)
    assert config.api_key.get_secret_value() == "sk-test"
    dumped = config.model_dump()
    assert dumped["api_key"] != "sk-test"


def test_read_document_request_accepts_json_file_path_for_vanilla():
    request = ReadDocumentRequest.model_validate(
        {"file_path": {"title": "Report", "pages": 3}}
    )

    assert request.file_path == {"title": "Report", "pages": 3}


# ---- Error paths ---- #


def test_read_document_request_rejects_unknown_reader():
    with pytest.raises(ValidationError):
        ReadDocumentRequest.model_validate(
            {
                "file_path": "hello",
                "reader": {"reader": "UnknownReader"},
            }
        )


def test_split_document_request_rejects_semantic_splitter_without_embedding():
    with pytest.raises(ValidationError) as error:
        SplitDocumentRequest.model_validate(
            {
                "reader_output": {"text": "hello", "document_path": ""},
                "splitter": {"splitter": "SemanticSplitter"},
            }
        )

    assert "requires a top-level embedding" in str(error.value)


def test_split_document_request_rejects_embedding_for_character_splitter():
    with pytest.raises(ValidationError) as error:
        SplitDocumentRequest.model_validate(
            {
                "reader_output": {"text": "hello", "document_path": ""},
                "splitter": {"splitter": "CharacterSplitter"},
                "embedding": {"embedding": "OpenAIEmbedding"},
            }
        )

    assert "only valid when splitter is SemanticSplitter" in str(error.value)


def test_split_document_request_rejects_reserved_kwargs():
    with pytest.raises(ValidationError) as error:
        SplitDocumentRequest.model_validate(
            {
                "reader_output": {"text": "hello", "document_path": ""},
                "splitter": {"splitter": "CharacterSplitter"},
                "kwargs": {"splitter": "WordSplitter", "embedding": {}},
            }
        )

    assert "reserved keys" in str(error.value)


def test_read_document_request_rejects_empty_file_path():
    with pytest.raises(ValidationError):
        ReadDocumentRequest.model_validate({"file_path": "   "})


def test_read_document_request_rejects_reserved_kwargs():
    with pytest.raises(ValidationError) as error:
        ReadDocumentRequest.model_validate(
            {"file_path": "hello", "kwargs": {"file_path": "other", "model": {}}}
        )

    assert "reserved keys" in str(error.value)


def test_read_document_request_rejects_textract_with_json_file_path():
    with pytest.raises(ValidationError) as error:
        ReadDocumentRequest.model_validate(
            {
                "file_path": {"a": 1},
                "reader": {"reader": "TextractReader"},
            }
        )

    assert "does not support JSON file_path" in str(error.value)


def test_read_document_request_rejects_textract_with_model():
    with pytest.raises(ValidationError) as error:
        ReadDocumentRequest.model_validate(
            {
                "file_path": "/data/doc.pdf",
                "reader": {"reader": "TextractReader"},
                "model": {"model": "OpenAIVisionModel"},
            }
        )

    assert "does not accept a vision model" in str(error.value)


def test_read_document_request_rejects_markitdown_with_grok():
    with pytest.raises(ValidationError) as error:
        ReadDocumentRequest.model_validate(
            {
                "file_path": "/data/doc.pdf",
                "reader": {"reader": "MarkItDownReader"},
                "model": {"model": "GrokVisionModel"},
            }
        )

    assert "does not support model='GrokVisionModel'" in str(error.value)


def test_read_document_request_rejects_scan_pdf_pages_without_model():
    with pytest.raises(ValidationError) as error:
        ReadDocumentRequest.model_validate(
            {"file_path": "/data/doc.pdf", "kwargs": {"scan_pdf_pages": True}}
        )

    assert "scan_pdf_pages requires a model" in str(error.value)


def test_character_splitter_rejects_overlap_equal_to_chunk_size():
    with pytest.raises(ValidationError):
        CharacterSplitterConfiguration(
            splitter="CharacterSplitter",
            chunk_size=10,
            chunk_overlap=10,
        )


def test_character_splitter_rejects_float_overlap_outside_unit_interval():
    with pytest.raises(ValidationError):
        CharacterSplitterConfiguration(
            splitter="CharacterSplitter",
            chunk_size=10,
            chunk_overlap=1.5,
        )


def test_row_column_splitter_rejects_both_rows_and_cols():
    with pytest.raises(ValidationError):
        RowColumnSplitterConfiguration(
            splitter="RowColumnSplitter",
            num_rows=2,
            num_cols=2,
        )


def test_keyword_splitter_rejects_empty_patterns():
    with pytest.raises(ValidationError):
        KeywordSplitterConfiguration(splitter="KeywordSplitter", patterns=[])


def test_header_splitter_rejects_unknown_header_name():
    with pytest.raises(ValidationError):
        HeaderSplitterConfiguration(
            splitter="HeaderSplitter",
            headers_to_split_on=["Not a header"],
        )


def test_read_request_rejects_unknown_top_level_fields():
    with pytest.raises(ValidationError):
        ReadDocumentRequest.model_validate(
            {"file_path": "hello", "source": {"source_type": "text"}}
        )


# ---- Edge cases ---- #


def test_word_splitter_accepts_fractional_overlap():
    config = WordSplitterConfiguration(
        splitter="WordSplitter",
        chunk_size=10,
        chunk_overlap=0.2,
    )

    assert config.chunk_overlap == 0.2


def test_recursive_splitter_accepts_default_empty_separator():
    config = RecursiveCharacterSplitterConfiguration()

    assert "" in config.separators


def test_textract_configuration_keeps_optional_aws_fields():
    config = TextractReaderConfiguration(
        reader="TextractReader",
        region_name="us-east-1",
    )

    assert config.region_name == "us-east-1"
    assert config.profile_name is None


def test_vanilla_reader_configuration_is_default_for_read_request():
    request = ReadDocumentRequest(file_path="hello")

    assert isinstance(request.reader, VanillaReaderConfiguration)


def test_splitter_output_rejects_empty_chunks():
    from splitter_mr.schema.models import SplitterOutput

    with pytest.raises(ValidationError):
        SplitterOutput(chunks=[], document_path="")


def test_read_and_split_request_forwards_model_embedding_and_kwargs():
    request = ReadAndSplitRequest.model_validate(
        {
            "file_path": "hello",
            "model": {"model": "OpenAIVisionModel", "model_name": "gpt-4.1"},
            "kwargs": {"document_name": "a.txt", "prompt": "Extract text"},
            "splitter": {"splitter": "KeywordSplitter", "patterns": ["CHAPTER"]},
            "splitter_kwargs": {"include_delimiters": "both", "chunk_size": 12},
        }
    )

    assert request.model.model == "OpenAIVisionModel"
    assert request.kwargs["prompt"] == "Extract text"
    assert request.splitter_kwargs["include_delimiters"] == "both"
    assert request.splitter_kwargs["chunk_size"] == 12


def test_read_and_split_request_accepts_semantic_splitter_with_embedding():
    request = ReadAndSplitRequest.model_validate(
        {
            "file_path": "hello",
            "splitter": {"splitter": "SemanticSplitter"},
            "embedding": {"embedding": "OpenAIEmbedding"},
            "splitter_kwargs": {"buffer_size": 2},
        }
    )

    assert request.embedding.embedding == "OpenAIEmbedding"
    assert request.splitter_kwargs["buffer_size"] == 2
