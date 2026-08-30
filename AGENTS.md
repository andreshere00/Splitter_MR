# SplitterMR Agent Guide

## Project summary

SplitterMR is a Python 3.11+ library that turns files and raw data into structured,
LLM-ready text chunks. Its modular pipeline uses readers to produce `ReaderOutput`
objects, optional vision models to process non-text content, splitters to produce
`SplitterOutput` objects, and embedding providers for semantic workflows. The package
supports lightweight core installation plus optional `markitdown`, `docling`, and
`multimodal` integrations.

Source code lives in `src/splitter_mr/`; tests mirror it under `tests/splitter_mr/`.
The root `README.md` and `CHANGELOG.md` are the documentation sources synchronized to
`docs/index.md` and `docs/CHANGELOG.md` by the documentation tooling.

## First-time setup

- Before the first task in a fresh checkout, determine whether the environment is
  initialized (normally a usable `.venv` and installed pre-commit hooks).
- If it is not initialized, run `poe install` from the repository root before coding.
- Run this setup once per checkout, not before every task.
- Use Poe tasks from `pyproject.toml`: `poe format`, `poe test`, `poe docs`, and
  `poe build`. Prefer targeted pytest commands during development, then use `poe test`
  when full verification is appropriate; the full suite enforces at least 70% coverage.

## Development architecture

- Read neighboring implementations before adding code. Reuse existing schemas,
  constants, exceptions, warnings, helpers, and provider patterns instead of creating
  parallel abstractions.
- New readers must inherit `BaseReader`, implement `read(...)`, and return a validated
  `ReaderOutput`.
- New splitters must inherit `BaseSplitter`, implement `split(ReaderOutput)`, return a
  validated `SplitterOutput`, and reuse `_generate_chunk_ids` and `_default_metadata`
  where applicable. Preserve document metadata from the reader output.
- New vision providers must inherit `BaseVisionModel` and implement `__init__`,
  `get_client`, and `analyze_content(prompt, file, file_ext, **parameters)`.
- New embedding providers must inherit `BaseEmbedding` and implement `__init__`,
  `get_client`, and `embed_text`. Keep the default `embed_documents` behavior unless
  the backend supports a more efficient, equivalent batch implementation.
- Use `ReaderOutput` and `SplitterOutput` as the public contracts and access their
  fields with dot notation. Do not return ad-hoc dictionaries.
- Preserve the lightweight core: keep optional dependencies behind lazy imports and
  follow the existing registries, `TYPE_CHECKING`, `__getattr__`, and `__all__`
  patterns in each package.
- Export public implementations from the relevant package `__init__.py`, and add
  focused tests in the matching test package.

## Python standards

- Follow PEP 8, PEP 257, and PEP 484. Keep lines at or below 100 characters.
- Use `CamelCase` for classes, `snake_case` for functions, methods, and variables, and
  `UPPER_CASE` for constants.
- Add explicit type annotations to methods, parameters, return values, variables, and
  constants. Prefer Python 3.11 syntax such as `str | None`, `list[str]`, and
  `dict[str, Any]`.
- Write concise English Google-style docstrings for public classes and methods. Include
  `Args`, `Returns`, `Raises`, and `Warns` sections whenever applicable. Private methods
  and tiny helpers may omit docstrings when their behavior is self-explanatory.
- Keep functions small and solutions simple. Extend the established base-class design
  before introducing new layers or dependencies.
- Use project-specific exceptions and warnings at component boundaries, preserve the
  original exception with `raise ... from error`, and validate configuration early.
- Avoid unnecessary comments; use clear names and structure to explain the code.
- Run `poe format` after Python changes and fix any resulting lint issues.

## Tests

- Use pytest and follow Arrange-Act-Assert, with one behavior or scenario per test.
- Name tests `test_{method_name}_{state_under_test}_{expected_behavior}`.
- Organize each test module, as applicable, under exactly these section markers:
  `# ---- Mocks, fixtures & helpers ---- #`, `# ---- Happy path ---- #`,
  `# ---- Error paths ---- #`, and `# ---- Edge cases ---- #`.
- Test public contracts, validation, custom exceptions or warnings, metadata
  propagation, and optional-dependency behavior affected by the change.

## Documentation definition of done

Documentation is part of every programming task and must be completed before the
agent finishes its execution flow:

1. Update `README.md` so its feature summary, supported components, installation,
   public API, and examples remain consistent with the implemented behavior.
2. Update the root `CHANGELOG.md` with a concise entry describing the user-visible
   feature, fix, breaking change, documentation change, or developer change. Add or
   maintain an `Unreleased` section when the change has no assigned release.
3. Treat the root files as sources of truth; do not hand-edit `docs/index.md` or
   `docs/CHANGELOG.md`. Let `scripts/documentation.sh` synchronize the generated copies.
4. Verify code, tests, README, and changelog agree before reporting the task complete.
