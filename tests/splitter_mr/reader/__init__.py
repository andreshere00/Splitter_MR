import sys
import types

import pytest

import splitter_mr.reader as reader


def _make_dummy_readers(**attrs):
    mod = types.SimpleNamespace(**attrs)
    mod.__name__ = "splitter_mr.reader.readers"
    return mod


def test___all___contains_expected_names():
    assert set(reader.__all__) == {
        "BaseReader",
        "VanillaReader",
        "MarkItDownReader",
        "DoclingReader",
    }


def test_base_reader_is_exposed():
    # Should be imported at module import time (not lazy)
    assert hasattr(reader, "BaseReader")
    assert reader.BaseReader is not None


@pytest.mark.parametrize("name", ["VanillaReader", "MarkItDownReader", "DoclingReader"])
def test___getattr___delegates_to_readers(monkeypatch, name):
    sentinels = {
        "VanillaReader": object(),
        "MarkItDownReader": object(),
        "DoclingReader": object(),
    }
    dummy = _make_dummy_readers(**sentinels)

    # Ensure our dummy readers module is used when __getattr__ imports it
    monkeypatch.dict(sys.modules, {"splitter_mr.reader.readers": dummy}, clear=False)

    obj = getattr(reader, name)
    assert obj is sentinels[name]


def test___getattr___repeated_access_returns_same_object(monkeypatch):
    sentinel = object()
    dummy = _make_dummy_readers(VanillaReader=sentinel)
    monkeypatch.dict(sys.modules, {"splitter_mr.reader.readers": dummy}, clear=False)

    a = reader.VanillaReader
    b = reader.VanillaReader
    assert a is b is sentinel


def test___getattr___unknown_name_raises_attributeerror():
    with pytest.raises(AttributeError) as exc:
        getattr(reader, "DoesNotExist")
    msg = str(exc.value)
    assert f"module {reader.__name__!r} has no attribute 'DoesNotExist'" in msg


def test_lazy_import_only_when_needed(monkeypatch):
    """
    Accessing BaseReader should not force importing the lazy 'readers' module;
    accessing a lazy symbol should require it. We simulate by starting with no
    'splitter_mr.reader.readers' entry and injecting a dummy just in time.
    """
    # Ensure 'readers' not preloaded
    sys.modules.pop("splitter_mr.reader.readers", None)

    # Access a non-lazy symbol: should NOT create the 'readers' entry
    _ = reader.BaseReader
    assert "splitter_mr.reader.readers" not in sys.modules

    # Now set up a dummy readers module to be imported on demand
    sentinel = object()
    dummy = _make_dummy_readers(MarkItDownReader=sentinel)
    monkeypatch.dict(sys.modules, {"splitter_mr.reader.readers": dummy}, clear=False)

    # Accessing a lazy symbol should import/use the dummy readers module
    got = reader.MarkItDownReader
    assert got is sentinel
