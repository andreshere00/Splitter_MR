import pytest

from splitter_mr.schema import ReaderOutput
from splitter_mr.splitter import CodeSplitter

# Sample code for each language
PYTHON_CODE = """
def foo():
    pass

class Bar:
    def baz(self):
        pass
"""

JAVA_CODE = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""

KOTLIN_CODE = """
fun main() {
    println("Hello, World!")
}

class Greeter {
    fun greet() = println("Hi!")
}
"""


@pytest.mark.parametrize(
    "language, code",
    [
        ("python", PYTHON_CODE),
        ("java", JAVA_CODE),
        ("kotlin", KOTLIN_CODE),
    ],
)
def test_splits_various_languages(language, code):
    reader = ReaderOutput(
        text=code,
        document_name=f"example.{language}",
        document_path=f"/tmp/example.{language}",
    )
    splitter = CodeSplitter(chunk_size=50, language=language)
    output = splitter.split(reader)

    # basic sanity
    assert isinstance(output.chunks, list)
    assert all(isinstance(c, str) for c in output.chunks)
    # should cover some snippet of original text
    assert any(code_part.strip() in code for code_part in output.chunks)
    # metadata consistency
    assert len(output.chunk_id) == len(output.chunks)
    assert output.split_method == "code_splitter"
    assert output.split_params["language"] == language


def test_invalid_language_raises_value_error():
    reader = ReaderOutput(text="print('hi')")
    splitter = CodeSplitter(chunk_size=50, language="notalang")
    with pytest.raises(ValueError, match="Unsupported language"):
        splitter.split(reader)


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
