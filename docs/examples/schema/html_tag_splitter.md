# **Example**: Splitting an HTML Table into Chunks with `HTMLTagSplitter`

As an example, we will use a dataset of donuts in HTML table format (see [reference dataset](https://github.com/andreshere00/Splitter_MR/blob/main/https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sweet_list.html)).
The goal is to split the table into groups of rows so that each chunk contains as many `<tr>` elements as possible, while not exceeding a maximum number of characters per chunk.

![HTML Tag examples](https://www.tutorialspoint.com/html/images/html_basic_tags.jpg)

---

## Step 1: Read the HTML Document

We will use the `VanillaReader` to load our HTML table.

```python
from splitter_mr.reader import VanillaReader

reader = VanillaReader()

# You can provide a local path or a URL to your HTML file
url = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sweet_list.html"
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
    document_name='sweet_list.html',
    document_path='https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sweet_list.html',
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

```html
<table border="1" cellpadding="4" cellspacing="0">
    <thead>
      <tr>
        <th>id</th>
        <th>type</th>
        <th>name</th>
        <th>batter</th>
        <th>topping</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>None</td></tr>
      <tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>Glazed</td></tr>
      <tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>Sugar</td></tr>
      ...
      <tr><td>0006</td><td>filled</td><td>Filled</td><td>Regular</td><td>Chocolate</td></tr>
      <tr><td>0006</td><td>filled</td><td>Filled</td><td>Regular</td><td>Maple</td></tr>
    </tbody>
  </table>
```

This table can be interpretated in markdown format as:

|id|type|name|batter|topping|
|--- |--- |--- |--- |--- |
|0001|donut|Cake|Regular|None|
|0001|donut|Cake|Regular|Glazed|
|0001|donut|Cake|Regular|Sugar|
|...|...|...|...|...|
|0006|filled|Filled|Regular|Chocolate|
|0006|filled|Filled|Regular|Maple|


---

## Step 2: Chunk the HTML Table Using `HTMLTagSplitter`

To split the table into groups of rows, instantiate the `HTMLTagSplitter` with the desired tag (in this case, `"tr"` for table rows) and a chunk size in characters.

```python
from splitter_mr.splitter import HTMLTagSplitter

# Set chunk_size to the max number of characters you want per chunk
splitter = HTMLTagSplitter(chunk_size=400, tag="tr")
splitter_output = splitter.split(reader_output)
print(splitter_output)
```

The output is a `SplitterOutput` object:

```python
SplitterOutput(
    chunks=[
        '<html><body><thead>\n<tr>\n<th>id</th>\n<th>type</th>\n<th>name</th>\n<th>batter</th>\n<th>topping</th>\n</tr>\n</thead><tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>None</td></tr><tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>Glazed</td></tr><tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>Sugar</td></tr></body></html>', ... 
        '<html><body><thead>\n<tr>\n<th>id</th>\n<th>type</th>\n<th>name</th>\n<th>batter</th>\n<th>topping</th>\n</tr>\n</thead><tr><td>0006</td><td>filled</td><td>Filled</td><td>Regular</td><td>Powdered Sugar</td></tr><tr><td>0006</td><td>filled</td><td>Filled</td><td>Regular</td><td>Chocolate</td></tr><tr><td>0006</td><td>filled</td><td>Filled</td><td>Regular</td><td>Maple</td></tr></body></html>'
        ],
    chunk_id=[
        'fb0cc57f-866b-4a15-9369-e9ac7905c521', ..., 
        'b2ec81a9-044d-4a90-ac5b-705b77e7bcd5'], 
    document_name='sweet_list.html', 
    document_path='https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sweet_list.html', 
    document_id='0d9f18c6-31a8-4a5d-b7d1-16287233ba64', 
    conversion_method='html', 
    reader_method='vanilla', 
    ocr_method=None, 
    split_method='html_tag_splitter', 
    split_params={
        'chunk_size': 400, 
        'tag': 'tr'
        }, 
    metadata={}
    )
```

To visualize each chunk, simply iterate through them:

```python
for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```

And the output will be chunks with a valid HTML format:

```
======================================== Chunk 1 ========================================
<html><body><thead>
<tr>
<th>id</th>
<th>type</th>
<th>name</th>
<th>batter</th>
<th>topping</th>
</tr>
</thead><tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>None</td></tr><tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>Glazed</td></tr><tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>Sugar</td></tr></body></html>

======================================== Chunk 2 ========================================
<html><body><thead>
<tr>
<th>id</th>
<th>type</th>
<th>name</th>
<th>batter</th>
<th>topping</th>
</tr>
</thead><tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>Powdered Sugar</td></tr><tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>Chocolate with Sprinkles</td></tr><tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>Chocolate</td></tr></body></html>
...
```

In markdown format can be displayed as:

**Chunk 1:**

| id   | type  | name | batter  | topping |
|-------|-------|-------|---------|---------|
| 0001  | donut | Cake  | Regular | None    |
| 0001  | donut | Cake  | Regular | Glazed  |
| 0001  | donut | Cake  | Regular | Sugar   |

**Chunk 2:**

| id   | type  | name | batter  | topping |
|-------|-------|-------|---------|---------|
| 0001  | donut | Cake  | Regular | Powdered Sugar          |
| 0001  | donut | Cake  | Regular | Chocolate with Sprinkles|
| 0001  | donut | Cake  | Regular | Chocolate               |
| 0001  | donut | Cake  | Regular | Maple                   |

**And that's it!** You can now flexibly chunk HTML tables for processing, annotation, or LLM ingestion.

---

## Complete Script

Here is the full example you can use directly:

```python
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import HTMLTagSplitter

# Step 1: Read the HTML file
reader = VanillaReader()
url = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sweet_list.html"  # Use your path or URL here
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
