# **Example**: Splitting a Python Source File into Chunks with `CodeSplitter`

Suppose you have a Python code file and want to split it into chunks that respect function and class boundaries (rather than just splitting every N characters). The `CodeSplitter` leverages [LangChain's RecursiveCharacterTextSplitter](https://python.langchain.com/docs/how_to/code_splitter/) to achieve this, making it ideal for preparing code for LLM ingestion, code review, or annotation.

![Programming languages](https://bairesdev.mo.cloudinary.net/blog/2020/10/top-programming-languages.png?tx=w_1920,q_auto)

---

## Step 1: Read the Python Source File

We will use the `VanillaReader` to load our code file. You can provide a local file path (or a URL if your implementation supports it).

!!! Note
    In case that you use `MarkItDownReader` or `DoclingReader`, save your files in `txt` format.

```python
from splitter_mr.reader import VanillaReader

reader = VanillaReader()
reader_output = reader.read("https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/code_example.py")
```

The `reader_output` is an object containing the raw code and its metadata:

```python
print(reader_output)
```

Example output:

```python
ReaderOutput(
    text='from langchain_text_splitters import Language, RecursiveCharacterTextSplitter\n\nfrom ...',
    document_name='code_example.py',
    document_path='https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/code_example.py',
    document_id='8fdfd0c1-1b94-4eb0-86f4-c304ce8aa463',
    conversion_method='txt',
    reader_method='vanilla',
    ocr_method=None,
    metadata={}
)
```

To see the code content:

```python
print(reader_output.text)
```

```python
ReaderOutput(
    text='from langchain_text_splitters import Language, RecursiveCharacterTextSplitter\n\nfrom ...schema import ReaderOutput, SplitterOutput\nfrom ..base_splitter import BaseSplitter\n\n\ndef get_langchain_language(lang_str: str) -> Language:\n    """\n    Map a string language name to Langchain Language enum...', 
    chunk_id=['682dd838-f672-4337-b52a-b68b6a4cb0b1', 'a978d9be-cfe1-4a61-b73c-49842bdeca30', 'f390953f-d4b3-40b1-bd87-a9b1b7e634c9', 'f2bde214-5378-49e4-8d84-8832d46e2e26', '1cc57a4d-4618-4e53-bda1-977a343cbe9e', '58eb9713-320a-4a9c-924c-0ebce6b1a228'], 
    document_name='code_example.py', 
    document_path='https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/code_example.py', document_id='8fdfd0c1-1b94-4eb0-86f4-c304ce8aa463', conversion_method='txt', 
    reader_method='vanilla', 
    ocr_method=None, 
    split_method='code_splitter', 
    split_params={'chunk_size': 1000, 'language': 'python'}, 
    metadata={}
)
```

---

## Step 2: Chunk the Code Using `CodeSplitter`

To split your code by language-aware logical units, instantiate the `CodeSplitter`, specifying the `chunk_size` (maximum number of characters per chunk) and `language` (e.g., `"python"`):

```python
from splitter_mr.splitter import CodeSplitter

splitter = CodeSplitter(chunk_size=1000, language="python")
splitter_output = splitter.split(reader_output)
```

The `splitter_output` contains the split code chunks:

```python
print(splitter_output)
```

Example output:

```python
SplitterOutput(
    chunks=[
        'from langchain_text_splitters import Language, RecursiveCharacterTextSplitter\n\nfrom ...',
        'class CodeSplitter(BaseSplitter):\n    """\n    CodeSplitter recursively splits source code...',
        # ...
    ],
    chunk_id=[...],
    document_name='code_example.py',
    # ...
)
```

To inspect the split results, iterate over the chunks and print them:

```python
for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx} " + "="*40)
    print(chunk)
```

Example output:

```python
======================================== Chunk 0 ========================================
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from ...schema import ReaderOutput, SplitterOutput
from ..base_splitter import BaseSplitter

def get_langchain_language(lang_str: str) -> Language:
    ...
======================================== Chunk 1 ========================================
class CodeSplitter(BaseSplitter):
    """
    CodeSplitter recursively splits source code into programmatically meaningful chunks
    (functions, classes, methods, etc.) for the given programming language.

    Args:
        chunk_size (int): Maximum chunk size, in characters.
        language (str): Programming language (e.g., "python", "java", "kotlin", etc.)
    ...
```

**And that's it!** You now have an efficient, language-aware way to chunk your code files for downstream tasks. 

Remember that you have plenty of programming languages available: JavaScript, Go, Rust, Java, etc. Currently, the available programming languages are:

```python
LANGUAGES: str = {
    'lua',
    'java',
    'ts', 'tsx',
    'ps1', 'psm1', 'psd1', 'ps1xml',
    'php', 'php3', 'php4', 'php5', 'phps', 'phtml',
    'rs',
    'cs', 'csx',
    'cob', 'cbl',
    'hs',
    'scala',
    'swift',
    'tex',
    'rb', 'erb',
    'kt', 'kts',
    'go',
    'html', 'htm',
    'rst',
    'ex', 'exs',
    'md', 'markdown',
    'proto',
    'sol',
    'c', 'h',
    'cpp', 'cc', 'cxx', 'c++', 'hpp', 'hh', 'hxx',
    'js', 'mjs',
    'py', 'pyw', 'pyc', 'pyo',
    'pl', 'pm',
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

print(reader_output)  # See metadata
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

* [LangChain's RecursiveCharacterTextSplitter](https://python.langchain.com/docs/how_to/code_splitter/) 