from unittest import mock

import pytest

# Import the module under test as 'readers'
import splitter_mr.reader.readers as readers


def test___dir__lists_all():
    assert sorted(readers.__dir__()) == sorted(readers.__all__)


def test___getattr__success(monkeypatch):
    # Simulate a module and class to be imported
    dummy_class = type("VanillaReader", (), {})
    dummy_module = mock.Mock()
    setattr(dummy_module, "VanillaReader", dummy_class)

    with mock.patch("importlib.import_module", return_value=dummy_module) as import_mod:
        cls = readers.__getattr__("VanillaReader")
        import_mod.assert_called_with(".vanilla_reader", package="my_package.readers")
        assert cls is dummy_class


def test___getattr__raises_attributeerror():
    # Attribute not in REGISTRY
    with pytest.raises(AttributeError) as e:
        readers.__getattr__("NotAReader")
    assert "has no attribute 'NotAReader'" in str(e.value)


def test___getattr__modulenotfounderror_with_extra(monkeypatch):
    # e.g., MarkItDownReader requires 'markitdown' extra
    with mock.patch(
        "importlib.import_module", side_effect=ModuleNotFoundError("No module")
    ):
        with pytest.raises(ModuleNotFoundError) as e:
            readers.__getattr__("MarkItDownReader")
        assert "requires the 'markitdown' extra" in str(e.value)
        assert "pip install 'splitter-mr[markitdown]'" in str(e.value)


def test___getattr__modulenotfounderror_without_extra(monkeypatch):
    # e.g., VanillaReader does not require an extra
    with mock.patch(
        "importlib.import_module", side_effect=ModuleNotFoundError("No module")
    ):
        with pytest.raises(ModuleNotFoundError) as e:
            readers.__getattr__("VanillaReader")
        # Should NOT mention pip install or extra
        assert "pip install" not in str(e.value)
