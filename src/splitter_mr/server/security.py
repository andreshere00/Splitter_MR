"""Filesystem and URL access policy for the SplitterMR server."""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .exceptions import ServerAccessDeniedError, ServerConfigurationError
from .settings import ServerSettings

_BLOCKED_IP_CHECKS = (
    "is_private",
    "is_loopback",
    "is_link_local",
    "is_multicast",
    "is_reserved",
    "is_unspecified",
)
_ALLOWED_SCHEMES = frozenset({"http", "https"})


class _Redirect(Exception):
    """Internal control-flow exception used to inspect redirect hops."""

    def __init__(self, location: str) -> None:
        super().__init__(location)
        self.location = location


class _RaiseOnRedirect(HTTPRedirectHandler):
    """Turn HTTP redirects into exceptions so each hop can be validated."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise _Redirect(newurl)


def resolve_allowed_file(path: str, settings: ServerSettings) -> Path:
    """Resolve a client-supplied path and enforce the allowed-root policy.

    Args:
        path: Filesystem path supplied in a file source.
        settings: Active server settings.

    Returns:
        The resolved file path.

    Raises:
        ServerAccessDeniedError: If file access is disabled or the path escapes
            the allowed root.
        ServerConfigurationError: If the path does not exist or is not a file.
    """
    if settings.allowed_root is None:
        raise ServerAccessDeniedError(
            "File sources are disabled until SPLITTER_MR_ALLOWED_ROOT is set."
        )
    root = settings.allowed_root.expanduser().resolve()
    candidate = Path(path).expanduser().resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ServerAccessDeniedError(
            "File path is outside SPLITTER_MR_ALLOWED_ROOT."
        ) from error
    if not candidate.exists() or not candidate.is_file():
        raise ServerConfigurationError("File path does not exist or is not a file.")
    return candidate


def validate_url(url: str, settings: ServerSettings) -> None:
    """Reject disallowed URL schemes, hosts, and resolved IP addresses.

    Args:
        url: Absolute HTTP(S) URL.
        settings: Active server settings.

    Raises:
        ServerAccessDeniedError: If URL access is disabled or the target is not
            allowed.
    """
    if not settings.allow_urls:
        raise ServerAccessDeniedError(
            "URL sources are disabled until SPLITTER_MR_ALLOW_URLS is true."
        )
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ServerAccessDeniedError("Only http and https URLs are allowed.")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ServerAccessDeniedError("URL is missing a hostname.")
    if settings.allowed_url_hosts and host not in settings.allowed_url_hosts:
        raise ServerAccessDeniedError(
            "URL host is not in SPLITTER_MR_ALLOWED_URL_HOSTS."
        )
    _assert_public_host(host)


def fetch_url(url: str, settings: ServerSettings) -> tuple[bytes, str, str]:
    """Fetch a URL while re-validating every redirect hop.

    Args:
        url: Absolute HTTP(S) URL.
        settings: Active server settings.

    Returns:
        Tuple of response body, final URL, and inferred filename.

    Raises:
        ServerAccessDeniedError: If a hop violates URL policy.
        ServerConfigurationError: If the fetch fails or the redirect budget is
            exceeded.
    """
    current = url
    opener = build_opener(_RaiseOnRedirect)
    for _ in range(settings.max_url_redirects + 1):
        validate_url(current, settings)
        request = Request(current, method="GET")
        try:
            with opener.open(request, timeout=30) as response:
                body = response.read()
                final_url = response.geturl() or current
                filename = _filename_from_url(final_url)
                return body, final_url, filename
        except _Redirect as redirect:
            current = redirect.location
            continue
        except HTTPError as error:
            raise ServerConfigurationError("Failed to fetch URL source.") from error
        except ServerAccessDeniedError:
            raise
        except ServerConfigurationError:
            raise
        except Exception as error:
            raise ServerConfigurationError("Failed to fetch URL source.") from error
    raise ServerConfigurationError("Too many HTTP redirects while fetching URL source.")


def _assert_public_host(host: str) -> None:
    """Resolve a hostname and reject non-public addresses.

    Args:
        host: Hostname or literal IP.

    Raises:
        ServerAccessDeniedError: If resolution fails or any address is blocked.
    """
    try:
        parsed_ip = ipaddress.ip_address(host)
    except ValueError:
        parsed_ip = None
    if parsed_ip is not None:
        _reject_blocked_ip(parsed_ip)
        return
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as error:
        raise ServerAccessDeniedError("URL hostname could not be resolved.") from error
    if not infos:
        raise ServerAccessDeniedError("URL hostname could not be resolved.")
    for info in infos:
        sockaddr = info[4]
        ip_text = str(sockaddr[0]).split("%", 1)[0]
        _reject_blocked_ip(ipaddress.ip_address(ip_text))


def _reject_blocked_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    """Reject private, loopback, and other reserved addresses.

    Args:
        address: Resolved IP address.

    Raises:
        ServerAccessDeniedError: If the address is not a public unicast target.
    """
    if any(getattr(address, check) for check in _BLOCKED_IP_CHECKS):
        raise ServerAccessDeniedError("URL target resolves to a blocked IP address.")


def _filename_from_url(url: str) -> str:
    """Infer a document name from a URL path.

    Args:
        url: Final URL after redirects.

    Returns:
        Basename or a generic downloaded-file name.
    """
    path = urlparse(url).path.rstrip("/")
    name = path.split("/")[-1] if path else ""
    return name or "downloaded_file"
