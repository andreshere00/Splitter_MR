# **Example**: Splitting a Python Source File into Chunks with `CodeSplitter`

Suppose you have a Python code file and want to split it into chunks that respect function and class boundaries (rather than just splitting every N characters). The [**`CodeSplitter`**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#codesplitter) leverages [LangChain's RecursiveCharacterTextSplitter](https://python.langchain.com/docs/how_to/code_splitter/) to achieve this, making it ideal for preparing code for LLM ingestion, code review, or annotation.

![Programming languages](https://bairesdev.mo.cloudinary.net/blog/2020/10/top-programming-languages.png?tx=w_1920,q_auto)

---

## Step 1: Read the Python Source File

We will use the [**`VanillaReader`**](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#vanillareader) to load our code file. You can provide a local file path (or a URL if your implementation supports it).

!!! Note
    In case that you use [`MarkItDownReader`](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#markitdownreader) or [`DoclingReader`](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#doclingreader), save your files in `txt` format.


```python
from splitter_mr.reader import VanillaReader

reader = VanillaReader()
reader_output = reader.read(
    "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/code_example.py"
)
```


The [`reader_output`](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#splitter_mr.schema.models.ReaderOutput) is an object containing the raw code and its metadata:


```python
print(reader_output.model_dump_json(indent=4))
```

    {
        "text": "from langchain_text_splitters import Language, RecursiveCharacterTextSplitter\n\nfrom ...schema import ReaderOutput, SplitterOutput\nfrom ..base_splitter import BaseSplitter\n\n\ndef get_langchain_language(lang_str: str) -> Language:\n    \"\"\"\n    Map a string language name to Langchain Language enum.\n    Raises ValueError if not found.\n    \"\"\"\n    lookup = {lang.name.lower(): lang for lang in Language}\n    key = lang_str.lower()\n    if key not in lookup:\n        raise
    ...
    split_params={\"chunk_size\": chunk_size, \"language\": self.language},\n            metadata=metadata,\n        )\n        return output\n",
        "document_name": "code_example.py",
        "document_path": "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/code_example.py",
        "document_id": "fadd9a15-06ba-488a-8c9c-9fd09ebbe82c",
        "conversion_method": "txt",
        "reader_method": "vanilla",
        "ocr_method": null,
        "page_placeholder": null,
        "metadata": {}
    }



To see the code content:


```python
print(reader_output.text)
```

    from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
    
    from ...schema import ReaderOutput, SplitterOutput
    from ..base_splitter import BaseSplitter
    
    
    def get_langchain_language(lang_str: str) -> Language:
        """
        Map a string language name to Langchain Language enum.
        Raises ValueError if not found.
        """
        lookup = {lang.name.lower(): lang for lang in Language}
        key = lang_str.lower()
        if key not in lookup:
            raise ValueError(
                f"Unsuppor
    ...
    cument_name=reader_output.document_name,
                document_path=reader_output.document_path,
                document_id=reader_output.document_id,
                conversion_method=reader_output.conversion_method,
                reader_method=reader_output.reader_method,
                ocr_method=reader_output.ocr_method,
                split_method="code_splitter",
                split_params={"chunk_size": chunk_size, "language": self.language},
                metadata=metadata,
            )
            return output
    



---

## Step 2: Chunk the Code Using `CodeSplitter`

To split your code by language-aware logical units, instantiate the `CodeSplitter`, specifying the `chunk_size` (maximum number of characters per chunk) and `language` (e.g., `"python"`):


```python
from splitter_mr.splitter import CodeSplitter

splitter = CodeSplitter(chunk_size=1000, language="python")
splitter_output = splitter.split(reader_output)
```


The [`splitter_output`](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#splitter_mr.schema.models.ReaderOutput) contains the split code chunks:


```python
print(splitter_output)
```

    chunks=['from langchain_text_splitters import Language, RecursiveCharacterTextSplitter\n\nfrom ...schema import ReaderOutput, SplitterOutput\nfrom ..base_splitter import BaseSplitter\n\n\ndef get_langchain_language(lang_str: str) -> Language:\n    """\n    Map a string language name to Langchain Language enum.\n    Raises ValueError if not found.\n    """\n    lookup = {lang.name.lower(): lang for lang in Language}\n    key = lang_str.lower()\n    if key not in lookup:\n        raise ValueError(
    ...
    945-9485-e915b616319d', 'c2a4cdb9-1cea-40ff-8474-949de5cb3cbb', 'cf065bed-bf46-4984-b3ca-f38297737b56', 'ef15e01d-98ad-4112-a4fd-cef4c2179772'] document_name='code_example.py' document_path='https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/code_example.py' document_id='fadd9a15-06ba-488a-8c9c-9fd09ebbe82c' conversion_method='txt' reader_method='vanilla' ocr_method=None split_method='code_splitter' split_params={'chunk_size': 1000, 'language': 'python'} metadata={}



To inspect the split results, iterate over the chunks and print them:


```python
for idx, chunk in enumerate(splitter_output.chunks):
    print("=" * 40 + f" Chunk {idx} " + "=" * 40)
    print(chunk)
```

    ======================================== Chunk 0 ========================================
    from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
    
    from ...schema import ReaderOutput, SplitterOutput
    from ..base_splitter import BaseSplitter
    
    
    def get_langchain_language(lang_str: str) -> Language:
        """
        Map a string language name to Langchain Language enum.
        Raises ValueError if not found.
        """
        lookup = {lang.name.lower(): lang for lang in Language}
        key = l
    ...
    ocument_name=reader_output.document_name,
                document_path=reader_output.document_path,
                document_id=reader_output.document_id,
                conversion_method=reader_output.conversion_method,
                reader_method=reader_output.reader_method,
                ocr_method=reader_output.ocr_method,
                split_method="code_splitter",
                split_params={"chunk_size": chunk_size, "language": self.language},
                metadata=metadata,
            )
            return output



**And that's it!** You now have an efficient, language-aware way to chunk your code files for downstream tasks. 

Remember that you have plenty of programming languages available: **JavaScript, Go, Rust, Java**, etc. Currently, the available programming languages are:


```python
from typing import Set

SUPPORTED_PROGRAMMING_LANGUAGES: Set[str] = {
    "lua",
    "java",
    "ts",
    "tsx",
    "ps1",
    "psm1",
    "psd1",
    "ps1xml",
    "php",
    "php3",
    "php4",
    "php5",
    "phps",
    "phtml",
    "rs",
    "cs",
    "csx",
    "cob",
    "cbl",
    "hs",
    "scala",
    "swift",
    "tex",
    "rb",
    "erb",
    "kt",
    "kts",
    "go",
    "html",
    "htm",
    "rst",
    "ex",
    "exs",
    "md",
    "markdown",
    "proto",
    "sol",
    "c",
    "h",
    "cpp",
    "cc",
    "cxx",
    "c++",
    "hpp",
    "hh",
    "hxx",
    "js",
    "mjs",
    "py",
    "pyw",
    "pyc",
    "pyo",
    "pl",
    "pm",
}
```


!!! Note

    Remember that you can visit the [LangchainTextSplitter documentation](https://python.langchain.com/docs/how_to/code_splitter/) to see the up-to-date information about the available programming languages to split on.

## Complete Script

Here is a full example you can run directly:

```python
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import CodeSplitter

# Step 1: Read the code file
reader = VanillaReader()
reader_output = reader.read("https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/code_example.py")

print(reader_output.model_dump_json(indent=4))  # See metadata
print(reader_output.text)  # See raw code

# Step 2: Split code into logical chunks, max 1000 chars per chunk
splitter = CodeSplitter(chunk_size=1000, language="python")
splitter_output = splitter.split(reader_output)

print(splitter_output)  # Print the SplitterOutput object

# Step 3: Visualize code chunks
for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx} " + "="*40)
    print(chunk)
```


### References

[LangChain's RecursiveCharacterTextSplitter](https://python.langchain.com/docs/how_to/code_splitter/) 
