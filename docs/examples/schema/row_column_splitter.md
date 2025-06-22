# **Example**: Splitting Tabular Data with `RowColumnSplitter`

Tabular files such as CSVs, TSVs, or Markdown tables are ubiquitous in business and data workflows, but can become too large for direct LLM ingestion, annotation, or analysis. **SplitterMR’s `RowColumnSplitter` provides flexible chunking for tabular data, letting you split tables by rows, columns, or character size—while preserving the structural integrity of each chunk.**

![Tabular data example](https://cdn.prod.website-files.com/5ec4696a9b6d337d51632638/66a13775bfe37ea3ea11bbdf_666b29d252001a163f5260a4_65af215b95aef7ce55d2a923_Screenshot%2525202024-01-22%252520at%2525209.13.25%2525E2%252580%2525AFPM.png)

## Step 1: Read the Tabular File

Let's use the `VanillaReader` to load a CSV file:

```python
from splitter_mr.reader import VanillaReader

file = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/invoices.csv"
reader = VanillaReader()
reader_output = reader.read(file)

# Print metadata and content
print(reader_output)
print(reader_output.text)
```

**Sample output:**

```python
ReaderOutput(
    text='id,name,amount,Remark\n1,"Johnson, Smith, and Jones Co.",345.33,Pays on time\n2,"Sam ""Mad Dog"" Smith",993.44,\n3,Barney & Company,0,"Great to work with and always pays with cash."\n4,Johnson\'s Automotive,2344,',
    document_name='invoices.csv',
    document_path='https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/invoices.csv',
    document_id='4eb99715-b11c-4a34-b0be-724ae6e34e4c',
    conversion_method='csv',
    reader_method='vanilla',
    ocr_method=None,
    metadata={}
)
```

The content file is extracted accessing to the `text` attribute:

```text
id,name,amount,Remark
1,"Johnson, Smith, and Jones Co.",345.33,Pays on time
2,"Sam ""Mad Dog"" Smith",993.44,
3,Barney & Company,0,"Great to work with
and always pays with cash."
4,Johnson's Automotive,2344,
```

Transformed into a markdown table will be:

| id   | name   | amount   | Remark   |
|------|--------|----------|----------|
|    1 | Johnson, Smith, and Jones Co. |   345.33 | Pays on time |
|    2 | Sam "Mad Dog" Smith |   993.44 |      nan |
|    3 | Barney & Company |        0 | Great to work with and always pays with cash. |
|    4 | Johnson's Automotive |     2344 |      nan |

## Step 2: Split the Table

### 2.1. **Split by Character Size (row-wise, preserving full rows)**

Split into chunks such that each chunk's markdown table representation stays under a character limit:

```python
from splitter_mr.splitter import RowColumnSplitter

splitter = RowColumnSplitter(chunk_size=200)
splitter_output = splitter.split(reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```

**Chunk 1:**

| id   | name   | amount   | Remark   |
|------|--------|----------|----------|
|    1 | Johnson, Smith, and Jones Co. |   345.33 | Pays on time |
|    2 | Sam "Mad Dog" Smith |   993.44 |      nan |

**Chunk 2:**

| id   | name   | amount   | Remark   |
|------|--------|----------|----------|
|    3 | Barney & Company |        0 | Great to work with and always pays with cash. |

**Chunk 3:**

| id   | name   | amount   | Remark   |
|------|--------|----------|----------|
|    4 | Johnson's Automotive |     2344 |      nan |

Each output chunk is a valid markdown table with the header and as many full rows as will fit the character size. 

!!! note
    No chunk will ever split a row or a column in half.

### 2.2. **Split by a Fixed Number of Rows**

Set `num_rows` to split the table into smaller tables, each with a fixed number of rows:

```python
splitter = RowColumnSplitter(num_rows=2)
splitter_output = splitter.split(reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```

The output will be:

**Chunk 1:**

|   id | name                          |   amount | Remark       |
|-----:|:------------------------------|---------:|:-------------|
|    1 | Johnson, Smith, and Jones Co. |   345.33 | Pays on time |
|    2 | Sam "Mad Dog" Smith           |   993.44 | nan          |

**Chunk 2:**

|   id | name                 |   amount | Remark                                        |
|-----:|:---------------------|---------:|:----------------------------------------------|
|    3 | Barney & Company     |        0 | Great to work with and always pays with cash. |
|    4 | Johnson's Automotive |     2344 | nan                                           |

### 2.3. **Split by a Fixed Number of Columns**

Set `num_cols` to split the table into column groups, each containing a fixed set of columns (e.g., for wide tables):

```python
splitter = RowColumnSplitter(num_cols=2)
splitter_output = splitter.split(reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```

The output will be a list with this format:

```python
======================================== Chunk 1 ========================================
[['id', 1, 2, 3, 4], ['name', 'Johnson, Smith, and Jones Co.', 'Sam "Mad Dog" Smith', 'Barney & Company', "Johnson's Automotive"]]

======================================== Chunk 2 ========================================
[['amount', 345.33, 993.44, 0.0, 2344.0], ['Remark', 'Pays on time', nan, 'Great to work with and always pays with cash.', nan]]
```

### 2.4. **Add Overlapping Rows/Columns**

Use `chunk_overlap` (int or float between 0 and 1) to specify how many rows or columns are repeated between consecutive chunks for context preservation:

```python
splitter = RowColumnSplitter(chunk_size=150, chunk_overlap=0.2)
splitter_output = splitter.split(reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```

The output is a table with an overlapping row column:

**Chunk 1:**

| id   | name   | amount   | Remark   |
|------|--------|----------|----------|
|    1 | Johnson, Smith, and Jones Co. |   345.33 | Pays on time |
|    2 | Sam "Mad Dog" Smith |   993.44 |      nan |
|    3 | Barney & Company |        0 | Great to work with and always pays with cash. |

**Chunk 2:**

| id   | name   | amount   | Remark   |
|------|--------|----------|----------|
|    3 | Barney & Company |        0 | Great to work with and always pays with cash. |
|    4 | Johnson's Automotive |     2344 |      nan |

!!! note
    `chunk_overlap` parameter can be used by rows or columns.

**And that's it!** In this example we have used a CSV files, but we can process other file formats. The compatible file extensions are: `csv`, `tsv`, `md`, `txt` and tabular `json`. Parquet files which are processed as JSON can be processed as well. 

!!! warning
    Setting both `num_rows` and `num_cols` will raise an error.
    If `chunk_overlap` is a float, it is interpreted as a percentage (e.g., `0.2` means 20% overlap).

### 3. Use cases

`RowColumnSplitter` is useful for the following use cases:

* For splitting large tabular datasets into LLM-friendly or context-aware chunks.
* For preserving row/column integrity in `csv`/`tsv`/`markdown` data.
* When you need easy chunking with overlap for annotation, document search, or analysis.

## Complete script

```python
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import RowColumnSplitter

file = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/invoices.csv"

reader = VanillaReader()
reader_output = reader.read(file)

# Visualize the ReaderOutput object
print(reader_output)

# Access to the text content
print(reader_output.text)

print("*"*20 + " Split by rows based on chunk size " + "*"*20)

splitter = RowColumnSplitter(chunk_size=200)
splitter_output = splitter.split(reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")

print("*"*20 + " Split by an specific number of rows " + "*"*20)

splitter = RowColumnSplitter(num_rows=2)
splitter_output = splitter.split(reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")

print("*"*20 + " Split by an specific number of columns " + "*"*20)

splitter = RowColumnSplitter(num_cols=2)
splitter_output = splitter.split(reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")

print("*"*20 + " Split with overlap " + "*"*20)

splitter = RowColumnSplitter(chunk_size=300, chunk_overlap=0.4)
splitter_output = splitter.split(reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```
