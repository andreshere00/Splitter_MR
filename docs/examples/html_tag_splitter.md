# Example: Splitting an HTML Table into Chunks with `HTMLTagSplitter`

As an example, we'll use a dataset of donuts in HTML table format (see [reference dataset](https://github.com/andreshere00/Splitter_MR/blob/main/data/test_2.html)).
The goal is to split the table into groups of rows so that each chunk contains as many `<tr>` elements as possible, while not exceeding a maximum number of characters per chunk.

---

## Step 1: Read the HTML Document

We will use the `VanillaReader` to load our HTML table.

```python
from splitter_mr.reader import VanillaReader

reader = VanillaReader()

# You can provide a local path or a URL to your HTML file
url = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/test_2.html"
reader_output = reader.read(url)
```

The `reader_output` object contains the raw HTML and metadata.

```python
print(reader_output)
```

Example output:

```python
ReaderOutput(
    text='<table border="1" cellpadding="4" cellspacing="0">\n  <thead>\n    <tr> ...',
    document_name='test_2.html',
    document_path='data/test_2.html',
    document_id='ae194c82-4ea6-465f-8d49-fc2a36214748',
    conversion_method='html',
    reader_method='vanilla',
    ocr_method=None,
    metadata={}
)
```

To see the HTML text:

```python
print(reader_output.text)
```

---

## Step 2: Chunk the HTML Table Using `HTMLTagSplitter`

To split the table into groups of rows, instantiate the `HTMLTagSplitter` with the desired tag (in this case, `"tr"` for table rows) and a chunk size in characters.

```python
from splitter_mr.splitter import HTMLTagSplitter

# Set chunk_size to the max number of characters you want per chunk
splitter = HTMLTagSplitter(chunk_size=400, tag="tr")
splitter_output = splitter.split(reader_output)
```

The `splitter_output` contains the chunks:

```python
print(splitter_output)
```

Example output:

```python
SplitterOutput(
    chunks=[
        '<html><body><tr> ... </tr><tr> ... </tr><tr> ... </tr></body></html>',
        '<html><body><tr> ... </tr><tr> ... </tr> ... </body></html>',
        ...
    ],
    chunk_id=[...],
    document_name='test_2.html',
    ...
)
```

To visualize each chunk, simply iterate through them:

```python
for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```

Sample output:

```
======================================== Chunk 1 ========================================
<html><body><tr>
<th>id</th>
<th>type</th>
<th>name</th>
<th>batter</th>
<th>topping</th>
</tr><tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>None</td></tr><tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>Glazed</td></tr><tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>Sugar</td></tr></body></html>
======================================== Chunk 2 ========================================
<html><body><tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>Powdered Sugar</td></tr><tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>Chocolate with Sprinkles</td></tr><tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>Chocolate</td></tr><tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>Maple</td></tr></body></html>
...
```

---

## Complete Script

Here is the full example you can use directly:

```python
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import HTMLTagSplitter

# Step 1: Read the HTML file
reader = VanillaReader()
url = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/test_2.html"  # Use your path or URL here
reader_output = reader.read(url)

print(reader_output)  # Visualize the ReaderOutput object
print(reader_output.text)  # See the HTML content

# Step 2: Split by group of <tr> tags, max 400 characters per chunk
splitter = HTMLTagSplitter(chunk_size=400, tag="tr")
splitter_output = splitter.split(reader_output)

print(splitter_output)  # Print the SplitterOutput object

# Step 3: Visualize each HTML chunk
for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```

---

!!! Note

    If you want to always include the table header in every chunk (for tables), you can enhance the splitter to prepend the `<thead>` content to each chunk.

---

**And that's it!** You can now flexibly chunk HTML tables for processing, annotation, or LLM ingestion.
