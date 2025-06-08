import pytest

from splitter_mr.reader import BaseReader
from splitter_mr.schema import ReaderOutput


def test_base_reader_is_abstract():
    with pytest.raises(TypeError):
        BaseReader()


def test_base_reader_subclass_must_implement_read():
    class BadReader(BaseReader):
        pass

    with pytest.raises(TypeError):
        BadReader()


def test_base_reader_concrete_subclass_can_be_instantiated(tmp_path):
    # Minimal implementation
    class DummyReader(BaseReader):
        def read(self, file_path: str, **kwargs):
            return ReaderOutput(
                text="dummy",
                document_name="dummy.txt",
                document_path=file_path,
                document_id="dummy-id",
                conversion_method=None,
                reader_method="dummy",
                ocr_method=None,
                metadata=None,
            )

    reader = DummyReader()
    result = reader.read("somepath.txt")
    assert result.text == "dummy"
    assert result.document_path == "somepath.txt"
