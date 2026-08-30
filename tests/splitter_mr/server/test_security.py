from pathlib import Path
from urllib.error import HTTPError

import pytest

from splitter_mr.server.exceptions import (
    ServerAccessDeniedError,
    ServerConfigurationError,
    ServerPayloadTooLargeError,
)
from splitter_mr.server.security import fetch_url, resolve_allowed_file, validate_url
from splitter_mr.server.settings import ServerSettings

# ---- Mocks, fixtures & helpers ---- #


def _settings(**overrides) -> ServerSettings:
    payload = {
        "allow_urls": False,
        "allowed_url_hosts": [],
        "max_url_redirects": 2,
        "max_body_bytes": 1024,
    }
    payload.update(overrides)
    return ServerSettings.model_validate(payload)


class _FakeResponse:
    def __init__(self, body: bytes, url: str) -> None:
        self._body = body
        self._url = url

    def read(self) -> bytes:
        return self._body

    def geturl(self) -> str:
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None


# ---- Happy path ---- #


def test_resolve_allowed_file_accepts_path_inside_root(tmp_path: Path):
    document = tmp_path / "doc.txt"
    document.write_text("ok", encoding="utf-8")
    settings = _settings(allowed_root=tmp_path)

    resolved = resolve_allowed_file(str(document), settings)

    assert resolved == document.resolve()


def test_validate_url_accepts_public_host(monkeypatch):
    monkeypatch.setattr(
        "splitter_mr.server.security.socket.getaddrinfo",
        lambda host, port: [(0, 0, 0, "", ("1.1.1.1", 0))],
    )
    settings = _settings(allow_urls=True, allowed_url_hosts=["example.com"])

    validate_url("https://example.com/doc.txt", settings)


def test_fetch_url_returns_body_and_filename(monkeypatch):
    monkeypatch.setattr(
        "splitter_mr.server.security.socket.getaddrinfo",
        lambda host, port: [(0, 0, 0, "", ("1.1.1.1", 0))],
    )

    class FakeOpener:
        def open(self, request, timeout=30):
            return _FakeResponse(b"hello", request.full_url)

    monkeypatch.setattr(
        "splitter_mr.server.security.build_opener",
        lambda *args: FakeOpener(),
    )
    settings = _settings(allow_urls=True)

    body, final_url, filename = fetch_url("https://example.com/manual.pdf", settings)

    assert body == b"hello"
    assert filename == "manual.pdf"
    assert final_url.endswith("manual.pdf")


# ---- Error paths ---- #


def test_resolve_allowed_file_denies_when_root_unset():
    with pytest.raises(ServerAccessDeniedError):
        resolve_allowed_file("/tmp/doc.txt", _settings())


def test_resolve_allowed_file_denies_path_outside_root(tmp_path: Path):
    outside = Path("/etc/hosts")
    settings = _settings(allowed_root=tmp_path)

    with pytest.raises(ServerAccessDeniedError):
        resolve_allowed_file(str(outside), settings)


def test_resolve_allowed_file_rejects_missing_file(tmp_path: Path):
    settings = _settings(allowed_root=tmp_path)

    with pytest.raises(ServerConfigurationError):
        resolve_allowed_file(str(tmp_path / "missing.txt"), settings)


def test_validate_url_denies_when_disabled():
    with pytest.raises(ServerAccessDeniedError):
        validate_url("https://example.com/a.txt", _settings())


def test_validate_url_denies_loopback_literal():
    settings = _settings(allow_urls=True)

    with pytest.raises(ServerAccessDeniedError):
        validate_url("http://127.0.0.1/secret", settings)


def test_validate_url_denies_private_literal():
    settings = _settings(allow_urls=True)

    with pytest.raises(ServerAccessDeniedError):
        validate_url("http://192.168.1.10/doc", settings)


def test_validate_url_denies_host_not_in_allowlist(monkeypatch):
    monkeypatch.setattr(
        "splitter_mr.server.security.socket.getaddrinfo",
        lambda host, port: [(0, 0, 0, "", ("1.1.1.1", 0))],
    )
    settings = _settings(allow_urls=True, allowed_url_hosts=["allowed.example"])

    with pytest.raises(ServerAccessDeniedError):
        validate_url("https://other.example/doc", settings)


def test_validate_url_denies_resolved_private_ip(monkeypatch):
    monkeypatch.setattr(
        "splitter_mr.server.security.socket.getaddrinfo",
        lambda host, port: [(0, 0, 0, "", ("10.0.0.5", 0))],
    )
    settings = _settings(allow_urls=True)

    with pytest.raises(ServerAccessDeniedError):
        validate_url("https://evil.example/doc", settings)


def test_fetch_url_revalidates_redirect_to_loopback(monkeypatch):
    from splitter_mr.server.security import _Redirect

    monkeypatch.setattr(
        "splitter_mr.server.security.socket.getaddrinfo",
        lambda host, port: [(0, 0, 0, "", ("1.1.1.1", 0))],
    )

    class RedirectOpener:
        def open(self, request, timeout=30):
            if "example.com" in request.full_url:
                raise _Redirect("http://127.0.0.1/secret")
            return _FakeResponse(b"nope", request.full_url)

    monkeypatch.setattr(
        "splitter_mr.server.security.build_opener",
        lambda *args: RedirectOpener(),
    )
    settings = _settings(allow_urls=True)

    with pytest.raises(ServerAccessDeniedError):
        fetch_url("https://example.com/doc", settings)


def test_fetch_url_maps_http_errors(monkeypatch):
    monkeypatch.setattr(
        "splitter_mr.server.security.socket.getaddrinfo",
        lambda host, port: [(0, 0, 0, "", ("1.1.1.1", 0))],
    )

    class ErrorOpener:
        def open(self, request, timeout=30):
            raise HTTPError(request.full_url, 404, "nope", hdrs=None, fp=None)

    monkeypatch.setattr(
        "splitter_mr.server.security.build_opener",
        lambda *args: ErrorOpener(),
    )
    settings = _settings(allow_urls=True)

    with pytest.raises(ServerConfigurationError):
        fetch_url("https://example.com/missing.pdf", settings)


# ---- Edge cases ---- #


def test_resolve_allowed_file_rejects_directory(tmp_path: Path):
    nested = tmp_path / "dir"
    nested.mkdir()
    settings = _settings(allowed_root=tmp_path)

    with pytest.raises(ServerConfigurationError):
        resolve_allowed_file(str(nested), settings)


def test_resolve_allowed_file_rejects_symlink_escape(tmp_path: Path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = allowed / "escape.txt"
    link.symlink_to(outside)
    settings = _settings(allowed_root=allowed)

    with pytest.raises(ServerAccessDeniedError):
        resolve_allowed_file(str(link), settings)


def test_payload_too_large_error_has_stable_code():
    error = ServerPayloadTooLargeError("too big")

    assert error.code == "payload_too_large"
