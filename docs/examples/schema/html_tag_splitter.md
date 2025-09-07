# **Example**: Splitting an HTML Table into Chunks with `HTMLTagSplitter`

As an example, we will use a dataset of donuts in HTML table format (see [reference dataset](https://github.com/andreshere00/Splitter_MR/blob/main/https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sweet_list.html)).
The goal is to split the table into groups of rows so that each chunk contains as many `<tr>` elements as possible, while not exceeding a maximum number of characters per chunk.

![HTML Tag examples](https://www.tutorialspoint.com/html/images/html_basic_tags.jpg)

---

## Step 1: Read the HTML Document

We will use the [`VanillaReader`](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#vanillareader) to load our HTML table.


```python
from splitter_mr.reader import VanillaReader

reader = VanillaReader()  # you can use the argument html_to_markdown to transform the table directly to markdown format.

# You can provide a local path or a URL to your HTML file
url = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sweet_list.html"
reader_output = reader.read(url)
```


The [`reader_output`](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#output-format) object contains the raw HTML and metadata.


```python
print(reader_output.model_dump_json(indent=4))
```

    {
        "text": "<table border=\"1\" cellpadding=\"4\" cellspacing=\"0\">\n    <thead>\n      <tr>\n        <th>id</th>\n        <th>type</th>\n        <th>name</th>\n        <th>batter</th>\n        <th>topping</th>\n      </tr>\n    </thead>\n    <tbody>\n      <tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>None</td></tr>\n      <tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>Glazed</td></tr>\n      <tr><td>0001</td><td>donut</td><td>Cake</td><td>Regular</td><td>
    ...
    td>Chocolate</td></tr>\n      <tr><td>0006</td><td>filled</td><td>Filled</td><td>Regular</td><td>Maple</td></tr>\n    </tbody>\n  </table>",
        "document_name": "sweet_list.html",
        "document_path": "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sweet_list.html",
        "document_id": "413c5f15-c7bc-407f-862e-9f392aa46c95",
        "conversion_method": "html",
        "reader_method": "vanilla",
        "ocr_method": null,
        "page_placeholder": null,
        "metadata": {}
    }



To see the HTML text:


```python
print(reader_output.text)
```

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
          <tr><td>0001
    ...
    d>Glazed</td></tr>
          <tr><td>0005</td><td>twist</td><td>Twist</td><td>Regular</td><td>Sugar</td></tr>
          <tr><td>0006</td><td>filled</td><td>Filled</td><td>Regular</td><td>Glazed</td></tr>
          <tr><td>0006</td><td>filled</td><td>Filled</td><td>Regular</td><td>Powdered Sugar</td></tr>
          <tr><td>0006</td><td>filled</td><td>Filled</td><td>Regular</td><td>Chocolate</td></tr>
          <tr><td>0006</td><td>filled</td><td>Filled</td><td>Regular</td><td>Maple</td></tr>
        </tbody>
      </table>



This table can be interpretated in markdown format as:

|id|type|name|batter|topping|
|--- |--- |--- |--- |--- |
|0001|donut|Cake|Regular|None|
|0001|donut|Cake|Regular|Glazed|
|0001|donut|Cake|Regular|Sugar|
|...|...|...|...|...|
|0006|filled|Filled|Regular|Chocolate|
|0006|filled|Filled|Regular|Maple|

Note that you can parse directly this table in VanillaReader using the keyword argument `html_to_markdown=True`. Refer to the [class documentation](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#vanillareader).


---

## Step 2: Chunk the HTML Table Using `HTMLTagSplitter`

To split the table into groups of rows, instantiate the [**`HTMLTagSplitter`**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#htmltagsplitter) with the desired tag (in this case, `"tr"` for table rows) and a chunk size in characters.


```python
from splitter_mr.splitter import HTMLTagSplitter

# Set chunk_size to the max number of characters you want per chunk
splitter = HTMLTagSplitter(chunk_size=400, tag="tr")
splitter_output = splitter.split(reader_output)
print(splitter_output.model_dump_json(indent=4))
```

    {
        "chunks": [
            "| id | type | name | batter | topping |\n| --- | --- | --- | --- | --- |\n| 0001 | donut | Cake | Regular | None |\n| 0001 | donut | Cake | Regular | Glazed |",
            "| id | type | name | batter | topping |\n| --- | --- | --- | --- | --- |\n| 0001 | donut | Cake | Regular | Sugar |\n| 0001 | donut | Cake | Regular | Powdered Sugar |",
            "| id | type | name | batter | topping |\n| --- | --- | --- | --- | --- |\n| 0001 | donut | Cake | Regular | Chocolate with S
    ...
       "document_name": "sweet_list.html",
        "document_path": "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sweet_list.html",
        "document_id": "413c5f15-c7bc-407f-862e-9f392aa46c95",
        "conversion_method": "html",
        "reader_method": "vanilla",
        "ocr_method": null,
        "split_method": "html_tag_splitter",
        "split_params": {
            "chunk_size": 400,
            "tag": "table",
            "batch": true,
            "to_markdown": true
        },
        "metadata": {}
    }



To visualize each chunk, simply iterate through them:


```python
for idx, chunk in enumerate(splitter_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")
```

    ======================================== Chunk 1 ========================================
    | id | type | name | batter | topping |
    | --- | --- | --- | --- | --- |
    | 0001 | donut | Cake | Regular | None |
    | 0001 | donut | Cake | Regular | Glazed |
    
    ======================================== Chunk 2 ========================================
    | id | type | name | batter | topping |
    | --- | --- | --- | --- | --- |
    | 0001 | donut | Cake | Regular | Sugar |
    | 0001 | donut | Cake | Regular | Powdered Sugar 
    ...
    d | Regular | Glazed |
    
    ======================================== Chunk 24 ========================================
    | id | type | name | batter | topping |
    | --- | --- | --- | --- | --- |
    | 0006 | filled | Filled | Regular | Powdered Sugar |
    | 0006 | filled | Filled | Regular | Chocolate |
    
    ======================================== Chunk 25 ========================================
    | id | type | name | batter | topping |
    | --- | --- | --- | --- | --- |
    | 0006 | filled | Filled | Regular | Maple |
    



By default the output will be chunks with a valid markdown format. In case that needed, you can directly give the results in HTML format using the argument `to_markdown=False`. Refer to the [class documentation](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#htmltagsplitter).

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

```python
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import HTMLTagSplitter

# Step 1: Read the HTML file
reader = VanillaReader()
url = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sweet_list.html"  # Use your path or URL here
reader_output = reader.read(url)

print(reader_output.model_dump_json(indent=4))  # Visualize the ReaderOutput object
print(reader_output.text)  # See the HTML content

# Step 2: Split by group of <tr> tags, max 400 characters per chunk
splitter = HTMLTagSplitter(chunk_size=400, tag="tr")
splitter_output = splitter.split(reader_output)

print(splitter_output.model_dump_json(indent=4))  # Print the SplitterOutput object

# Step 3: Visualize each HTML chunk
for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```
