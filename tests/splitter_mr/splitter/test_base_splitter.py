import uuid

import pytest

from splitter_mr.splitter.base_splitter import BaseSplitter


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
            return {"chunks": ["a", "b"], "extra": "ok"}

    s = DummySplitter(chunk_size=5)
    result = s.split({"text": "abc"})
    assert result["chunks"] == ["a", "b"]
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
