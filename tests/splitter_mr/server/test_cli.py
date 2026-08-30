# ---- Mocks, fixtures & helpers ---- #


# ---- Happy path ---- #


def test_main_forwards_cli_overrides_to_uvicorn(monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)

    monkeypatch.setattr("splitter_mr.server.cli.uvicorn.run", fake_run)

    from splitter_mr.server.cli import main

    main(["--host", "0.0.0.0", "--port", "9000", "--log-level", "debug"])

    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9000
    assert captured["log_level"] == "debug"
    assert captured["args"][0] == "splitter_mr.server.app:app"


def test_main_uses_settings_defaults(monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("splitter_mr.server.cli.uvicorn.run", fake_run)
    monkeypatch.delenv("SPLITTER_MR_HOST", raising=False)
    monkeypatch.delenv("SPLITTER_MR_PORT", raising=False)
    monkeypatch.delenv("SPLITTER_MR_LOG_LEVEL", raising=False)

    from splitter_mr.server.cli import main

    main([])

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000
    assert captured["log_level"] == "info"


# ---- Error paths ---- #


def test_parser_rejects_invalid_log_level():
    from splitter_mr.server.cli import build_parser
    from splitter_mr.server.settings import ServerSettings

    parser = build_parser(ServerSettings())

    try:
        parser.parse_args(["--log-level", "verbose"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("expected argparse to reject the log level")


# ---- Edge cases ---- #


def test_parser_accepts_environment_backed_port(monkeypatch):
    monkeypatch.setenv("SPLITTER_MR_PORT", "8123")
    from splitter_mr.server.cli import build_parser
    from splitter_mr.server.settings import ServerSettings

    parser = build_parser(ServerSettings())
    args = parser.parse_args([])

    assert args.port == 8123
