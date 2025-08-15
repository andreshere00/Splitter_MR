import builtins
import types
import uuid
import warnings

import pytest

from splitter_mr.reader.readers.docling_reader import DoclingReader, _require_docling
from splitter_mr.reader.readers.vanilla_reader import VanillaReader
from splitter_mr.reader.utils import DoclingPipelineFactory
from splitter_mr.schema import ReaderOutput

# Helpers


# Dummy Model for tests
class DummyModel:
    def __init__(self, model_name="dummy"):
        self.model_name = model_name
        self._client = types.SimpleNamespace(
            _azure_endpoint="https://example.com",
            _azure_deployment="dep",
            _api_version="v1",
            api_key="key",
        )

    def get_client(self):
        return self._client


@pytest.fixture(autouse=True)
def patch_pipeline(monkeypatch):
    """Patch DoclingPipelineFactory.run and VanillaReader.read for all tests."""
    monkeypatch.setattr(
        DoclingPipelineFactory,
        "run",
        lambda name, path, **kwargs: f"{name}-pipeline-{kwargs.get('prompt', '')}",
    )
    monkeypatch.setattr(
        VanillaReader,
        "read",
        lambda self, file_path, **kwargs: ReaderOutput(
            text="vanilla-output",
            document_name="file.xyz",
            document_path=file_path,
            document_id="id",
            conversion_method="vanilla",
            reader_method="vanilla",
            ocr_method=None,
            metadata={},
        ),
    )
    yield


# Test cases


def test_init_with_and_without_model():
    reader1 = DoclingReader()
    assert reader1.model is None
    assert reader1.client is None
    assert reader1.model_name is None

    model = DummyModel("abc")
    reader2 = DoclingReader(model=model)
    assert reader2.model is model
    assert reader2.client == model.get_client()
    assert reader2.model_name == "abc"


def test_unsupported_extension_warns(monkeypatch):
    reader = DoclingReader()
    # Not in supported ext
    with warnings.catch_warnings(record=True) as w:
        out = reader.read("foo.unsupported")
        assert isinstance(out, ReaderOutput)
        assert out.text == "vanilla-output"
        assert any("Unsupported extension" in str(warn.message) for warn in w)


def test_pdf_scan_pdf_pages(monkeypatch):
    model = DummyModel()
    reader = DoclingReader(model)
    out = reader.read("x.pdf", scan_pdf_pages=True, prompt="foo")
    # Uses 'page_image' pipeline and passes arguments
    assert out.text.startswith("page_image-pipeline-foo")
    assert out.document_name == "x.pdf"
    assert out.conversion_method == "markdown"
    assert out.reader_method == "docling"
    assert out.ocr_method == model.model_name


def test_pdf_with_model_no_scan(monkeypatch):
    model = DummyModel()
    reader = DoclingReader(model)
    with warnings.catch_warnings(record=True) as w:
        out = reader.read("y.pdf", prompt="my prompt", show_base64_images=True)
        # Should raise a warning about base64 images with model
        assert any("base64 images are not rendered" in str(warn.message) for warn in w)
    # Uses 'vlm' pipeline
    assert out.text.startswith("vlm-pipeline-my prompt")
    assert out.ocr_method == model.model_name


def test_pdf_without_model(monkeypatch):
    reader = DoclingReader(model=None)
    out = reader.read("abc.pdf", show_base64_images=True)
    # Should use markdown pipeline
    assert out.text.startswith("markdown-pipeline-")
    assert out.reader_method == "docling"


def test_nonpdf_with_model(monkeypatch):
    # Key fix: The output is still "markdown-pipeline-", model is NOT used for pipeline selection.
    model = DummyModel()
    reader = DoclingReader(model)
    out = reader.read("file.md", show_base64_images=True)
    assert out.text.startswith("markdown-pipeline-")
    # However, ocr_method should reflect the model's name
    assert out.ocr_method == model.model_name
    assert out.reader_method == "docling"


def test_nonpdf_without_model(monkeypatch):
    reader = DoclingReader()
    out = reader.read("file.md", show_base64_images=True)
    assert out.text.startswith("markdown-pipeline-")
    assert out.ocr_method is None
    assert out.reader_method == "docling"


def test_metadata_and_docid_passthrough(monkeypatch):
    reader = DoclingReader()
    custom_id = str(uuid.uuid4())
    meta = {"x": 1}
    out = reader.read("abc.pdf", document_id=custom_id, metadata=meta)
    assert out.document_id == custom_id
    assert out.metadata == meta


def test__select_pipeline_pdf_scan_pdf_pages():
    model = DummyModel()
    reader = DoclingReader(model)
    pipeline, args = reader._select_pipeline(
        "doc.pdf", "pdf", scan_pdf_pages=True, prompt="x"
    )
    assert pipeline == "page_image"
    assert args["model"] == model
    assert args["prompt"] == "x"


def test__select_pipeline_pdf_with_model_no_scan():
    model = DummyModel()
    reader = DoclingReader(model)
    pipeline, args = reader._select_pipeline(
        "doc.pdf", "pdf", scan_pdf_pages=False, prompt="y", show_base64_images=True
    )
    assert pipeline == "vlm"
    assert args["model"] == model
    assert args["prompt"] == "y"


def test__select_pipeline_pdf_without_model():
    reader = DoclingReader()
    pipeline, args = reader._select_pipeline("doc.pdf", "pdf")
    assert pipeline == "markdown"
    assert args["show_base64_images"] is False


def test__select_pipeline_nonpdf():
    reader = DoclingReader()
    pipeline, args = reader._select_pipeline(
        "file.html", "html", show_base64_images=True
    )
    assert pipeline == "markdown"
    assert args["show_base64_images"] is True
    assert args["ext"] == "html"


@pytest.mark.parametrize(
    "text, page_placeholder, expected",
    [
        ("hello <!-- page --> world", "<!-- page -->", "<!-- page -->"),
        ("no placeholder here", "<!-- page -->", None),
        ("PAGEBREAK", "PAGEBREAK", "PAGEBREAK"),
        ("some\ntext\n", "NON_EXISTENT", None),
        ("a\nb\nc", "", None),
    ],
)
def test_page_placeholder_detection(monkeypatch, text, page_placeholder, expected):
    """Check if page_placeholder attribute is set correctly in ReaderOutput."""

    # Patch DoclingPipelineFactory.run to produce the desired output text
    monkeypatch.setattr(
        DoclingPipelineFactory, "run", lambda name, path, **kwargs: text
    )

    reader = DoclingReader()
    out = reader.read("x.pdf", page_placeholder=page_placeholder)
    assert out.page_placeholder == expected


def test_page_placeholder_default(monkeypatch):
    """Check default placeholder detection works."""
    output_text = "foo <!-- page --> bar"
    monkeypatch.setattr(
        DoclingPipelineFactory, "run", lambda name, path, **kwargs: output_text
    )
    reader = DoclingReader()
    # No page_placeholder passed, so default should be used and found in output
    out = reader.read("z.pdf")
    assert out.page_placeholder == "<!-- page -->"


def test_page_placeholder_none_when_absent(monkeypatch):
    """If output has no placeholder, attribute should be None."""
    output_text = "plain output, no page marker"
    monkeypatch.setattr(
        DoclingPipelineFactory, "run", lambda name, path, **kwargs: output_text
    )
    reader = DoclingReader()
    out = reader.read("a.pdf", page_placeholder="<!-- page -->")
    assert out.page_placeholder is None


def test__require_docling_raises_when_missing(monkeypatch):
    """If docling isn't installed, surface a clear extras message."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "docling":
            raise ImportError("No module named 'docling'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError) as ei:
        _require_docling()

    msg = str(ei.value)
    assert "requires the 'docling' extra" in msg
    assert "pip install splitter-mr[docling]" in msg


def test_docling_reader_ctor_raises_when_missing(monkeypatch):
    """Constructor should fail early with the same clear error when extra is missing."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "docling":
            raise ImportError("No module named 'docling'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError) as ei:
        DoclingReader()

    msg = str(ei.value)
    assert "requires the 'docling' extra" in msg
    assert "pip install splitter-mr[docling]" in msg


def test__require_docling_noop_when_present(monkeypatch):
    """If a (stub) docling module is present, no error should be raised."""
    stub = types.ModuleType("docling")
    monkeypatch.setitem(__import__("sys").modules, "docling", stub)

    # Should not raise
    _require_docling()
