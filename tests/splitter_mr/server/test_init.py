import sys

import pytest

# ---- Mocks, fixtures & helpers ---- #


# ---- Happy path ---- #


def test_server_dir_lists_public_exports():
    import splitter_mr.server as server

    names = set(dir(server))

    assert {"app", "create_app", "create_mcp_server"} <= names


# ---- Error paths ---- #


def test_unknown_attribute_raises_attribute_error():
    import splitter_mr.server as server

    with pytest.raises(AttributeError):
        getattr(server, "DoesNotExist")


def test_missing_fastapi_raises_actionable_extra_message(monkeypatch):
    import importlib

    import splitter_mr.server as server

    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name == ".app" and package == "splitter_mr.server":
            raise ModuleNotFoundError("No module named 'fastapi'")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(ModuleNotFoundError) as error:
        server.__getattr__("create_app")

    assert "splitter-mr[mcp]" in str(error.value)


# ---- Edge cases ---- #


def test_create_app_is_lazy_until_accessed():
    sys.modules.pop("splitter_mr.server.app", None)
    import splitter_mr.server as server

    assert "splitter_mr.server.app" not in sys.modules
    _ = server.__all__
    assert "splitter_mr.server.app" not in sys.modules


def test_schema_models_imports_without_torch(monkeypatch):
    import importlib

    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    sys.modules.pop("splitter_mr.schema.models", None)

    models = importlib.import_module("splitter_mr.schema.models")

    assert models.HFClient.model_fields["device"].default == "cpu"
