import importlib
import sys

import pytest

PKG = "splitter_mr.reader.utils"


# ---- Helpers, mocks and fixtures


def _reload_pkg():
    """
    Force a clean re-import of the package under test so monkeypatching
    importlib.import_module affects the import path resolution each time.
    """
    for key in list(sys.modules.keys()):
        if key == PKG or key.startswith(PKG + "."):
            sys.modules.pop(key, None)
    return importlib.import_module(PKG)


def test_unknown_attribute_raises_attributeerror():
    mod = _reload_pkg()
    with pytest.raises(AttributeError):
        getattr(mod, "DoesNotExist")


def test___dir___matches___all__():
    mod = _reload_pkg()
    assert set(dir(mod)) >= set(
        mod.__all__
    )  # dir() can include more, but must include __all__


# ---- Test cases ---- #


@pytest.mark.parametrize(
    "export_name,module_suffix",
    [
        ("PDFPlumberReader", ".pdfplumber_reader"),
        ("DoclingPipelineFactory", ".docling_utils"),
    ],
)
def test_registry_points_to_valid_modules(monkeypatch, export_name, module_suffix):
    """
    Smoke test that the registry tries to import the expected submodule name.
    We intercept import calls to confirm the right target is requested.
    """
    calls = []

    real_import_module = importlib.import_module

    def spy_import_module(name, package=None):
        abs_name = importlib.util.resolve_name(name, package) if package else name
        calls.append(abs_name)
        return real_import_module(name, package=package)

    monkeypatch.setattr(importlib, "import_module", spy_import_module)
    mod = _reload_pkg()

    try:
        getattr(mod, export_name)
    except ModuleNotFoundError:
        # If your environment truly lacks the optional extra, that's fine; we only
        # care that the import attempted the expected target module.
        pass

    assert any(call.endswith(module_suffix) for call in calls)
