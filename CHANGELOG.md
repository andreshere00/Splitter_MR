# CHANGELOG.md

## v0.1.0

### Functional features

- Add first readers, `VanillaReader`: reader which reads the files and format them into a string.
  - `DoclingReader`: reader which uses the docling package to read the files.
  - `MarkItDownReader`: reader which uses the markitdown package to read the files.
- Add first splitters: `CharacterSplitter`, `RecursiveSplitter`, `WordSplitter`, `SentenceSplitter`, `ParagraphSplitter`, `HTMLTagSplitter`, `JSONSplitter`, `HeaderSplitter`: 
- The package can be installed using pip.
- Documentation ([README.md](./README.md)) has been updated.
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