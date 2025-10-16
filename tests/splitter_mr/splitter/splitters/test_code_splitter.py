import warnings

import pytest

from splitter_mr.schema import ReaderOutput
from splitter_mr.schema.exceptions import (
    InvalidChunkException,
    SplitterOutputException,
    UnsupportedCodeLanguage,
)
from splitter_mr.schema.warnings import SplitterInputWarning
from splitter_mr.splitter import CodeSplitter
from splitter_mr.splitter.splitters import code_splitter as cs

# ---------------------------------------------------------------------
# Sample code for each language
# ---------------------------------------------------------------------

PYTHON_CODE: str = """
def foo():
    pass

class Bar:
    def baz(self):
        pass
"""

JAVA_CODE: str = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""

KOTLIN_CODE: str = """
fun main() {
    println("Hello, World!")
}

class Greeter {
    fun greet() = println("Hi!")
}
"""


# ---------------------------------------------------------------------
# Happy paths across multiple languages
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "language, code, ext",
    [
        ("python", PYTHON_CODE, "py"),
        ("java", JAVA_CODE, "java"),
        ("kotlin", KOTLIN_CODE, "kt"),
    ],
)
def test_splits_various_languages(language, code, ext):
    reader = ReaderOutput(
        text=code,
        document_name=f"example.{ext}",
        document_path=f"/tmp/example.{ext}",
    )
    splitter = CodeSplitter(chunk_size=50, language=language)
    output = splitter.split(reader)

    # basic sanity
    assert isinstance(output.chunks, list)
    assert all(isinstance(c, str) for c in output.chunks)
    # at least one chunk should be part of the original code
    assert any(chunk.strip() and chunk.strip() in code for chunk in output.chunks)
    # metadata consistency
    assert len(output.chunk_id) == len(output.chunks)
    assert output.split_method == "code_splitter"
    assert output.split_params["language"].lower() == language.lower()


def test_language_is_case_insensitive():
    reader = ReaderOutput(text=PYTHON_CODE)
    splitter = CodeSplitter(chunk_size=40, language="PyThOn")
    out = splitter.split(reader)
    assert isinstance(out.chunks, list)
    assert out.split_params["language"] == "PyThOn"  # preserves input casing in params


# ---------------------------------------------------------------------
# Constructor validation (ValueError)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("bad_size", [0, -1, 2.5, "100"])
def test_invalid_chunk_size_raises_value_error(bad_size):
    with pytest.raises(ValueError):
        CodeSplitter(chunk_size=bad_size, language="python")


# ---------------------------------------------------------------------
# Unsupported language
# ---------------------------------------------------------------------


def test_invalid_language_raises_unsupported_code_language():
    reader = ReaderOutput(text="print('hi')")
    splitter = CodeSplitter(chunk_size=50, language="notalang")
    with pytest.raises(UnsupportedCodeLanguage, match="Unsupported language"):
        splitter.split(reader)


# ---------------------------------------------------------------------
# Metadata pass-through
# ---------------------------------------------------------------------


def test_metadata_pass_through():
    reader = ReaderOutput(
        text=PYTHON_CODE,
        document_name="x.py",
        document_path="/tmp/x.py",
        document_id="docid123",
        conversion_method="manual",
        reader_method="text",
        ocr_method=None,
    )
    splitter = CodeSplitter(chunk_size=30, language="python")
    output = splitter.split(reader)

    assert output.document_name == "x.py"
    assert output.document_path == "/tmp/x.py"
    assert output.document_id == "docid123"
    assert output.conversion_method == "manual"
    assert output.reader_method == "text"
    assert output.ocr_method is None


# ---------------------------------------------------------------------
# Warnings (SplitterInputWarning)
# ---------------------------------------------------------------------


def test_empty_text_warns_and_returns_single_empty_chunk():
    reader = ReaderOutput(text="")
    splitter = CodeSplitter(chunk_size=50, language="python")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = splitter.split(reader)
        assert out.chunks == [""]
        assert any(isinstance(rec.message, SplitterInputWarning) for rec in w)


def test_whitespace_text_warns_and_raises_invalid_chunk():
    reader = ReaderOutput(text="     ")  # 5 spaces
    splitter = CodeSplitter(chunk_size=2, language="python")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with pytest.raises(InvalidChunkException, match="No chunks were produced"):
            splitter.split(reader)
        # Warning should have been emitted before the failure
        assert any(isinstance(rec.message, SplitterInputWarning) for rec in w)


def test_json_declared_invalid_emits_warning():
    reader = ReaderOutput(text="{not json", conversion_method="json")
    splitter = CodeSplitter(chunk_size=50, language="python")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = splitter.split(reader)
        assert len(out.chunks) >= 1
        assert any(isinstance(rec.message, SplitterInputWarning) for rec in w)


def test_json_declared_valid_no_warning():
    reader = ReaderOutput(text='{"a": 1, "b": [2,3]}', conversion_method="json")
    splitter = CodeSplitter(chunk_size=50, language="python")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = splitter.split(reader)
        assert len(out.chunks) >= 1
        assert not any(isinstance(rec.message, SplitterInputWarning) for rec in w)


# ---------------------------------------------------------------------
# InvalidChunkException wrapping
# ---------------------------------------------------------------------


def test_invalid_chunk_build_is_wrapped_as_invalid_chunk_exception(monkeypatch):
    reader = ReaderOutput(text="def f():\n    pass\n")
    splitter = CodeSplitter(chunk_size=50, language="python")

    # Make from_language raise when called, simulating an internal failure
    def raise_on_call(*args, **kwargs):
        raise RuntimeError("kapow")

    # Patch the symbol as imported in the module under test
    monkeypatch.setattr(
        cs.RecursiveCharacterTextSplitter,
        "from_language",
        staticmethod(raise_on_call),
    )

    with pytest.raises(
        InvalidChunkException, match="Unexpected error while building code chunks"
    ):
        splitter.split(reader)


def test_invalid_chunk_semantics_raise_invalid_chunk_exception(monkeypatch):
    class DummyDoc:
        def __init__(self, page_content):
            self.page_content = page_content

    class DummySplitter:
        def create_documents(self, _):
            # Simulate LangChain producing a document with None content (invalid)
            return [DummyDoc(None)]

    reader = ReaderOutput(text="def f():\n    pass\n")
    splitter = CodeSplitter(chunk_size=50, language="python")

    # Make from_language return our dummy splitter
    monkeypatch.setattr(
        cs.RecursiveCharacterTextSplitter,
        "from_language",
        staticmethod(lambda *a, **k: DummySplitter()),
    )

    with pytest.raises(InvalidChunkException, match="A produced chunk is None"):
        splitter.split(reader)


# ---------------------------------------------------------------------
# SplitterOutputException wrapping
# ---------------------------------------------------------------------


def test_splitter_output_exception_wrapped_when_output_validation_fails(monkeypatch):
    # Cause a chunk_id length mismatch to trigger SplitterOutputException
    reader = ReaderOutput(text=PYTHON_CODE)
    splitter = CodeSplitter(chunk_size=50, language="python")

    def bad_generate_chunk_ids(n):
        return ["only-one-id"]

    monkeypatch.setattr(splitter, "_generate_chunk_ids", bad_generate_chunk_ids)

    with pytest.raises(SplitterOutputException):
        splitter.split(reader)
