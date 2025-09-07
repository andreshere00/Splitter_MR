# v1.0.x

> **Version 1.0.0: First stable release with full Reader, Splitter, Embedding and Vision model support.**

## v1.0.0

First stable version.

### Features

- Consolidated all features introduced in v0.x series into a stable API.
- **Readers**: `VanillaReader`, `MarkItDownReader`, `DoclingReader` with support for multiple formats (text, Office, JSON/YAML, images, HTML, etc.).
- **Splitters**: character, word, sentence, paragraph, recursive, token, paged, row/column, JSON, semantic, HTML tag, header, and code splitting strategies.
- **Models**: support for multimodal Vision-Language Models (OpenAI, Azure, Grok, HuggingFace, Gemini, Claude).
- **Embeddings**: OpenAI, Azure, HuggingFace, Gemini, Claude (via Voyage) supported.

### Developer features

- **Optional extras system**: install lightweight **core** by default, extend with `markitdown`, `docling`, `multimodal`, `azure`, or `all`.
- **CI/CD pipeline**, **PyPI** release, and **pre-commit** checks in place.

### Documentation

**Extensive documentation** with API reference, examples, and architecture diagrams.

### New improvements + Bug fixes

- **Fix:** NLTK tokenizers in `TokenSplitter` are now correct base tokenizer when using `nltk` tokenizers.
- **Fix:** RecursiveJSONSplitter could not produce outputs since it did not validate correct data type.
- Now the examples are based on real Jupyter Notebooks executions to ensure that the behavior is the expected one.
- Added the Notebooks which are used as examples in the `notebooks` section.
- Update `clean` instruction with `poe`.
- New constans have been defined.
- Add new class to transform HTML to Markdown.

# v0.6.x

> [!IMPORTANT]
> **Breaking Change! Version v0.6.0**
>
> Dependencies are now split into **core** (installed by default) and **optional extras** for heavy or specialized features.
> - Example: to use MarkItDown and Docling readers, install with:
>   ```bash
>   pip install "splitter-mr[markitdown,docling]"
>   ```
> - To install *all* optional features:
>   ```bash
>   pip install "splitter-mr[all]"
>   ```

> - This change reduces install time and keeps core installs lightweight.

## v0.6.5

**Hotfix**: dependency isolation was not guaranteed.

### Features

- Add a util class to convert HTML to Markdown content.
- Improve Header Splitter to always return its content in markdown format.
- Add the option to return text in markdown format for HTMLTagSplitter.
- Add the option to batch content when using HTMLTagSplitter: If `batch=True`, it returns the chunks grouped by tags up to the numbers of characters specified by `chunk_size`. If False, it will return one register per tag.

### Bug fixes

- Dependency isolation was not guaranteed: implement safe lazy imports in all the `__init__` methods.
- Raise test coverage up to 90%. 

## 0.6.4

> **Version 0.6.4:**
> 
> SplitterMR now supports **Anthropic Claude** as a backend for both [**embedding**](https://andreshere00.github.io/Splitter_MR/api_reference/embedding#anthropicembeddings) (via [Voyage](https://docs.voyageai.com/docs/embeddings) AI) and [**vision**](https://andreshere00.github.io/Splitter_MR/api_reference/model#anthropicvisionmodel) models.

### Features

- Add new Vision Model: Claude Anthropic models.
- Add new Embedding Model: Voyage Anthropic models.

### Documentation

- Change font type to Documentation server.
- Update API reference guide with new links and resources.

## v0.6.3

> **Version 0.6.3:** SplitterMR now supports **Gemini** as a backend for both [**embedding**](https://andreshere00.github.io/Splitter_MR/api_reference/embedding#geminiembedding) and [**vision**](https://andreshere00.github.io/Splitter_MR/api_reference/model/#geminivisionmodel) models.
>
> **To use HuggingFace, Gemini, Claude or Grok models, you must install SplitterMR with the `multimodal` extra:**
>
> ```bash
> pip install "splitter-mr[multimodal]"
> ```

### Features

- Add `GeminiVisionModel` class to Vision models.
- Add `GeminiEmbedding` class to embedding models.
- Apply lazy import strategy to classes which require `extra`s to be installed (e.g., `docling`, `markitdown`, etc.).

### Documentation

- Update documentation.

## v0.6.2

> **Version 0.6.2:** SplitterMR now supports **HuggingFace** as a backend for both embedding and vision models:
>
> * [**HuggingFaceEmbedding**](https://andreshere00.github.io/Splitter_MR/api_reference/embedding/#huggingfaceembedding): Use any Sentence Transformers model (local or from Hugging Face Hub) for fast, local, or cloud embeddings.
> * [**HuggingFaceVisionModel**](https://andreshere00.github.io/Splitter_MR/api_reference/model#huggingfacevisionmodel): Leverage Hugging Face’s vision-language models for image-to-text and image captioning.
>
> **To use HuggingFace, Gemini, Claude or Grok models, you must install SplitterMR with the `multimodal` extra:**
>
> ```bash
> pip install "splitter-mr[multimodal]"
> ```

Add HuggingFace Model and Embedding support.

### Features

- Add `HuggingFaceVisionModel` class. Note that the support is limited until now.
- Add `HuggingFaceEmbedding` class.

### Documentation

- Add `HuggingFaceVisionModel` to documentation.
- Update architecture diagram.
- Update `README.md`.

### Developer features

- Add new dependencies to multimodal group.

## v0.6.1

Add Grok Vision Model.

> **Version 0.6.1:** SplitterMR now supports `GrokVisionModel`. See documentation [here](https://andreshere00.github.io/Splitter_MR/api_reference/model#grokvisionmodel).
> 
> **To use HuggingFace, Gemini, Claude or Grok models, you must install SplitterMR with the `multimodal` extra:**
>
> ```bash
> pip install "splitter-mr[multimodal]"
> `

### Features

- Add `GrokVisionModel`. 
- Redefine constants.
- Add new tests.

### Documentation

- Add `GrokVisionModel` documentation.
- Fix format bugs.
- Add new documentation in Readers about how to install necessary dependencies.
- Add plugin to read formulas appropiately.

## v0.6.0

Divide library into sub packages.

### Features

- Divide the library into sub-modules.

### Developer features

- Add new steps to Dockerfile images.
- Change `requirements.txt` to don't save editable builds as dependencies.
- Change how the `splitter_mr` library is installed within Dockerfiles.
- Lighten the weight of the library by making some dependencies optional.
- Change how `poe test` is executed.

### Documentation

- Fix Embedding models not showing on Developer Guide overview page.

# v0.5.x

> [!IMPORTANT]
> **New version v0.5.0**
>  
> - Add **embedding models** to encode the text into distributed vectorized representations. See documentation [here](https://andreshere00.github.io/Splitter_MR/api_reference/embedding/). 
> - Add support for chunking files based on **semantic similarity** between sentences. See documentation [here](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#semanticsplitter).

## v0.5.0

Add **SemanticSplitter** first implementation

### Features

- Add `embedding` module.
  - Add `AzureOpenAI embeddings`.
  - Add `OpenAI embeddings`.
  - Add `BaseEmbeddings`, to create your own class.
- Add `SemanticSplitter` class.

### Fixes

- Fix `SentenceSplitter` class to be more robust and flexible (separators can be customized using regex pattern).

### Developer features

- Update tests.

### Documentation

- Update documentation with new embedding module.
- Fix some format errors in Documentation server.
- Add new example documentation page for `SemanticSplitter`.

# **v0.4.x**

## v0.4.0

PagedSplitter full implementation

> [!IMPORTANT]
> **New version v0.4.0**
>  
> Add support for reading files and splitting them by pages using `PageSplitter`. Add support to read more files with `VanillaReader`.
>   
> ➡️ See [**documentation**](https://andreshere00.github.io/Splitter_MR/examples/text/paged_splitter/).

### Features

- Add support to read a PDF by pages using `MarkItDownReader` without LLM.
- Add method to read `xlsx`, `pptx`, `docx` files using `VanillaReader`.
- Add method to read several image formats using `VanillaReader`.
- Add support to read excel and parquet files using different engines in `VanillaReader`.
- Add support to analyze content in several file types using AzureOpenAI and OpenAI models.

### Documentation

- Update documentation.
- Fix some hyperlinks in README.

# **v0.3.x**

## v0.3.3

### Features

- Add a method to convert variables to a `ReaderOutput` object.
- Add a `page_placeholder` attribute to the `ReaderOutput` object to distinguish when a file has been read by pages and which placeholder is.
- Add an splitter method which split by pages for supported documents: `PagedSplitter`. 

### Developer features

- Refactor the `VanillaReader` class to be more decoupled.

### Documentation

- Update examples in documentation server.

## v0.3.2

### Features

- Add `Pydantic` models to validate inputs and outputs for `BaseReader` and `BaseSplitter` objects.
- Refactor models to modularize into constants, pydantic models and prompts.

### Documentation

- Update `README.md` to handle notes and warnings.

## v0.3.1

### Features

- Add support to read and scan PDF by pages for all the readers, using the parameter `scan_pdf_images = True`. 
- Add support to use different placeholders for images in Vanilla and Docling Readers.
- Add support to split by pages for PDFs. 
- Add three different pipelines to DoclingReader to process the document as PageImages, using VLM to provide image captioning and regularly.
- Add three different pipelines to VanillaReader to process the document as PageImages, using VLM to provide image captioning and regularly.

### Bugs

- Change how the arguments are passed to every Reader to enhance robutsness.
- Add new test cases.

### Documentation

- Update examples.
- Change MkDocs server to support both light and dark modes.

## v0.3.0

> [!IMPORTANT]
> **Vision Language Model (VLM) support!**
>  
> You can now use vision-capable models (OpenAI Vision, Azure OpenAI Vision) to extract image descriptions and OCR text during file reading. Pass a VLM model to any Reader class via the `model` parameter. 
>   
> ➡️ See [**documentation**](https://andreshere00.github.io/Splitter_MR/api_reference/model/).

### Features

- Implement `AzureOpenAI` and `OpenAI` Vision Models to analyze graphical resources in PDF files.
- Add support to read PDF files to VanillaReader using `PDFPlumber`.

### Documentation

- Update examples.
- Add new examples to documentation.
- Add Reading methods with PDF documentation.
- Add information about implementing VLMs in your reading pipeline.
- Change file names on data folder to be more descriptive.
- Update `README.md` and `CHANGELOG`.

### Fixes

- Update tests.
- Update docstrings.
- Update `TokenSplitter` to raise Exceptions if no valid models are provided.
- Update `TokenSplitter` to take as default argument a valid tiktoken model.
- Change `HTMLTagSplitter` to take the headers if a table is provided.
- Change `HeaderSplitter` to preserve headers in chunks.

# **v0.2.x**

## v0.2.2

### Features

- Implement `TokenSplitter`: split text into chunks by number of tokens, using selectable tokenizers (`tiktoken`, `spacy`, `nltk`).
- `MarkItDownReader` now supports more file extensions: PDFs, audio, etc.

### Fixes

- `HTMLTagSplitter` does not correctly chunking the document as desired.
- Change `docstring` documentation.

### Documentation

- Updated Splitter strategies documentation to include `TokenSplitter`.
- Expanded example scripts and test scripts for end-to-end manual and automated verification of all Splitter strategies.
- New examples in documentation server for `HTMLTagSplitter`. 

### Developer Features

- Remove `Pipfile` and `Pipfile.lock`. 
- Update to [`poethepoet`](https://poethepoet.natn.io/index.html) as task runner tool 

### Fixes

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

# **v0.1.x**

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
- Add first splitters: `CharacterSplitter`, `RecursiveCharacterSplitter`, `WordSplitter`, `SentenceSplitter`, `ParagraphSplitter`, `HTMLTagSplitter`, `RecursiveJSONSplitter`, `HeaderSplitter`: 
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