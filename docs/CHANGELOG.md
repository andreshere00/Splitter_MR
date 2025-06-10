# v0.2.x

## v0.2.1

### Features

Implement `CodeSplitter`.

### Fixes

- Change `docstring` for `BaseSplitter` to update with current parameters.
- Some minor warnings in documentations when deploying Mkdocs server.

## v0.2.0

> [!IMPORTANT]
> Breaking change!
> 
> - All Readers now return `ReaderOutput` dataclass objects.
> - All Splitters now return `SplitterOutput` dataclass objects.
> 
> You must access fields using **dot notation** (e.g., `result.text`, `result.chunks`), not dictionary keys.

### Features

New splitting strategy: `RowColumnSplitter` for flexible splitting of tabular data.

New reader_method attribute in output dataclasses.

### Migration

Update all code/tests to use attribute access for results from Readers and Splitters.

Use `.to_dict()` on output if a dictionary is required.

> Update any custom splitter/reader implementations to use the new output dataclasses.

# v0.1.x

## v0.1.3

### Features

- Add a new splitting strategy: `RowColumnSplitter`.

### Fixes

- Change Readers to properly handle JSON files.

### Documentation

- Update documentation.

## v0.1.2

### Features

- Now `VanillaReader` can read from multiple sources: URL, text, file_path and dictionaries.

### Fixes

- By default, the document_id was `None` instead of and `uuid` in `VanillaReader`. 
- Some name changes for `splitter_method` attribute in `SplitterOutput` method.

### Developer features

- Extend CI/CD lifecycle. Now it uses Dockerfile to check tests and deploy docs.
- Automate versioning for the Python project.
- The project has been published to [PyPI.org](https://pypi.org/project/splitter-mr/). New versions will be deployed using CI/CD script.
- `requirements.txt` has been added in the root of the project.
- A new stage in `pre-commit` has been introduced for generating the `requirements.txt` file.
- `Makefile` extended: new commands to serve `mkdocs`. Now make clean remove more temporary files.

### Documentation

- Update documentation in examples for the Documentation server.
- Documentation server now can be served using Make.

## v0.1.1

Some bug fixes in HeaderSplitter and RecursiveCharacterSplitter, and documentation updates.

### Bug fixes

- `chunk_overlap` (between 0 and 1) was not working in the `split` method from `RecursiveCharacterSplitter`.
- Some markdown code was not properly formatted in `README.md`.
- Reformat examples from docstring documentation in every Reader and Splitter classes.
- `HeaderSplitter` was not properly handling the headers in some `markdown` and `HTML` files.

### Documentation

- Some examples have been provided in the documentation (`docs/`, and in the [documentation server](https://andreshere00.github.io/Splitter_MR/)).
- New examples in docstrings.

## v0.1.0

First version of the project

### Functional features

- Add first readers, `VanillaReader`: reader which reads the files and format them into a string.
  - `DoclingReader`: reader which uses the docling package to read the files.
  - `MarkItDownReader`: reader which uses the markitdown package to read the files.
- Add first splitters: `CharacterSplitter`, `RecursiveSplitter`, `WordSplitter`, `SentenceSplitter`, `ParagraphSplitter`, `HTMLTagSplitter`, `JSONSplitter`, `HeaderSplitter`: 
- The package can be installed using pip.
- `README.md` has been updated.
- Tests cases for main functionalities are available.
- Some data has been added for testing purposes.
- A documentation server is deployed with up-to-date information.

### Developer features

- Update `pyproject.toml` project information.
- Add pre-commit configurations (`flake8`, check commit messages, run test coverage, and update documentation).
- Add first Makefile commands (focused on developers):
  - `make help`: Provide a list with all the Make commands.
  - `make clean`: Clean temporal files and cache
  - `make shell`: Run a `uv` shell.
  - `make install`: Install uv CLI and pre-commit.
  - `make precommit`: Install pre-commit hooks.
  - `make format`: Run pyupgrade, isort, black, and flake8 for code style.