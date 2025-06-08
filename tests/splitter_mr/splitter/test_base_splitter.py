import uuid

import pytest

from splitter_mr.schema import SplitterOutput
from splitter_mr.splitter import BaseSplitter


def test_base_splitter_is_abstract():
    with pytest.raises(TypeError):
        BaseSplitter()


def test_base_splitter_subclass_must_implement_split():
    class BadSplitter(BaseSplitter):
        pass

    with pytest.raises(TypeError):
        BadSplitter()


def test_base_splitter_minimal_concrete_subclass(tmp_path):
    class DummySplitter(BaseSplitter):
        def split(self, reader_output):
            # Return a SplitterOutput instance as expected by downstream code
            return SplitterOutput(
                chunks=["a", "b"],
                chunk_id=self._generate_chunk_ids(2),
                document_name="test",
                document_path="path",
                document_id="id",
                conversion_method=None,
                ocr_method=None,
                split_method="dummy_splitter",
                split_params={"chunk_size": self.chunk_size},
                metadata={},
            )

    s = DummySplitter(chunk_size=5)
    # Provide a dummy ReaderOutput or dict as needed (only text used)
    result = s.split({"text": "abc"})
    assert hasattr(result, "chunks")
    assert result.chunks == ["a", "b"]
    assert s.chunk_size == 5


def test_generate_chunk_ids_are_unique():
    class DummySplitter(BaseSplitter):
        def split(self, reader_output):
            return {}

    s = DummySplitter()
    chunk_ids = s._generate_chunk_ids(5)
    assert isinstance(chunk_ids, list)
    assert len(chunk_ids) == 5
    # Should all be valid UUID4s
    for cid in chunk_ids:
        uuid_obj = uuid.UUID(cid)
        assert uuid_obj.version == 4


def test_default_metadata_returns_empty_dict():
    class DummySplitter(BaseSplitter):
        def split(self, reader_output):
            return {}

    s = DummySplitter()
    meta = s._default_metadata()
    assert meta == {}
