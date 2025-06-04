import pytest

from splitter_mr.reader.base_reader import BaseReader


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
            return {"text": "dummy", "document_path": file_path}

    reader = DummyReader()
    result = reader.read("somepath.txt")
    assert result["text"] == "dummy"
    assert result["document_path"] == "somepath.txt"
