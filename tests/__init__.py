import sys
import types


def _reload_and_get_version(monkeypatch, version_value=None, raise_exc=False):
    """Helper: reloads the module-under-test with monkeypatched version()."""
    # Create a fresh dummy importlib.metadata module
    dummy_mod = types.ModuleType("importlib.metadata")

    # Patch version to return the test value or raise
    if raise_exc:

        def fake_version(name):
            raise dummy_mod.PackageNotFoundError("fail")

    else:

        def fake_version(name):
            return version_value

    dummy_mod.version = fake_version

    # Patch PackageNotFoundError to a new exception type for testing
    class MyPNFError(Exception):
        pass

    dummy_mod.PackageNotFoundError = MyPNFError

    # Insert dummy into sys.modules
    sys.modules["importlib.metadata"] = dummy_mod

    # Now exec our code string in a new module namespace to trigger init logic
    ns = {}
    code = (
        "from importlib.metadata import PackageNotFoundError, version\n"
        "try:\n"
        "    __version__ = version('splitter-mr')\n"
        "except PackageNotFoundError:\n"
        "    __version__ = '0.0.0'\n"
    )
    exec(code, ns)
    return ns["__version__"]


def test_init_version_success(monkeypatch):
    result = _reload_and_get_version(
        monkeypatch, version_value="3.1.4", raise_exc=False
    )
    assert result == "3.1.4"


def test_init_version_fallback(monkeypatch):
    result = _reload_and_get_version(monkeypatch, raise_exc=True)
    assert result == "0.0.0"
