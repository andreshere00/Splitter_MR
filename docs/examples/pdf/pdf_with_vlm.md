# **Example**: Reading files with Visual Language Models to Provide Image Annotations

!!! warning
This tutorial has been redone and it is **deprecated**. See new versions here:

```
1. [VanillaReader](./pdf_vanilla.md).
2. [DoclingReader](./pdf_docling.md).
3. [MarkItDownReader](./pdf_markitdown.md).
```

When reading a PDF file or other files which contain images, it can be useful to provide descriptive text alongside those images. Since images in a Markdown file are typically rendered by encoding them in base64 format, you may alternatively want to include a description of each image instead.

This is where **Visual Language Models (VLMs)** come in—to analyze and describe images automatically. In this tutorial, we'll show how to use these models with the library.

## Step 1: Load a Model

To extract image descriptions or perform OCR, instantiate any model that implements the [`BaseModel` interface](../../api_reference/model.md#basemodel) (vision variants inherit from it).

### Supported models (and when to use them)

| Model (docs)                                                                    | When to use                                       | Required environment variables                                                                                        |
| ------------------------------------------------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| [`OpenAIVisionModel`](../../api_reference/model.md#openaivisionmodel)           | You have an OpenAI API key and want OpenAI cloud. | `OPENAI_API_KEY` (optional: `OPENAI_MODEL`, defaults to `gpt-4o`)                                                     |
| [`AzureOpenAIVisionModel`](../../api_reference/model.md#azureopenaivisionmodel) | You use Azure OpenAI Service.                     | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`                |
| [`GrokVisionModel`](../../api_reference/model.md#grokvisionmodel)               | You have access to xAI Grok multimodal.           | `XAI_API_KEY` (optional: `XAI_MODEL`, default `grok-4`)                                                               |
| [`GeminiVisionModel`](../../api_reference/model.md#geminivisionmodel)           | You want Google’s Gemini vision models.           | `GEMINI_API_KEY` (also install extras: `pip install "splitter-mr[multimodal]"`)                                       |
| [`AnthropicVisionModel`](../../api_reference/model.md#anthropicvisionmodel)     | You have an Anthropic key (Claude Vision).        | `ANTHROPIC_API_KEY` (optional: `ANTHROPIC_MODEL`)                                                                     |
| [`HuggingFaceVisionModel`](../../api_reference/model.md#huggingfacevisionmodel) | You prefer local/open-source/offline inference.   | Install extras: `pip install "splitter-mr[multimodal]"` (optional: `HF_ACCESS_TOKEN` if the chosen model requires it) |

> **Note on HuggingFace models:** Not all HF models are supported (e.g., gated or uncommon architectures). A well-tested option is **SmolDocling**.

### Environment variables

<details>
  <summary><b>Show/hide environment variables needed for every provider</b></summary>

  <h4>OpenAI</h4> 

```txt
# OpenAI
OPENAI_API_KEY=<your-api-key>
# (optional) OPENAI_MODEL=gpt-4o
```

  <h4>Azure OpenAI</h4>

```txt
# Azure OpenAI
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_API_VERSION=<your-api-version>
AZURE_OPENAI_DEPLOYMENT=<your-model-name>
```

  <h4>xAI Grok</h4>

```txt
# xAI Grok
XAI_API_KEY=<your-api-key>
# (optional) XAI_MODEL=grok-4
```

  <h4>Google Gemini</h4>

```txt
# Google Gemini
GEMINI_API_KEY=<your-api-key>
# Also: pip install "splitter-mr[multimodal]"
```

  <h4>Anthropic (Claude Vision)</h4>

```txt
# Anthropic (Claude Vision)
ANTHROPIC_API_KEY=<your-api-key>
# (optional) ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

  <h4>Hugging Face (local/open-source)</h4>

```txt
# Hugging Face (optional, only if needed by the model)
HF_ACCESS_TOKEN=<your-hf-token>
# Also: pip install "splitter-mr[multimodal]"
```

</details>

### Instantiation examples

<details>
  <summary><b>Show/hide instantiation snippets for all providers</b></summary>

  <h4>OpenAI</h4>

```python
from splitter_mr.model import OpenAIVisionModel

# Reads OPENAI_API_KEY (and optional OPENAI_MODEL) from .env if present
model = OpenAIVisionModel()
# or pass explicitly:
# model = OpenAIVisionModel(api_key="...", model_name="gpt-4o")
```

  <h4>Azure OpenAI</h4>

```python
from splitter_mr.model import AzureOpenAIVisionModel

# Reads Azure vars from .env if present
model = AzureOpenAIVisionModel()
# or:
# model = AzureOpenAIVisionModel(
#     api_key="...",
#     azure_endpoint="https://<resource>.openai.azure.com/",
#     api_version="2024-02-15-preview",
#     azure_deployment="<your-deployment-name>",
# )
```

  <h4>xAI Grok</h4>

```python
from splitter_mr.model import GrokVisionModel

# Reads XAI_API_KEY (and optional XAI_MODEL) from .env
model = GrokVisionModel()
```

  <h4>Google Gemini</h4>

```python
from splitter_mr.model import GeminiVisionModel

# Requires GEMINI_API_KEY and the 'multimodal' extra installed
model = GeminiVisionModel()
```

  <h4>Anthropic (Claude Vision)</h4>

```python
from splitter_mr.model import AnthropicVisionModel

# Reads ANTHROPIC_API_KEY (and optional ANTHROPIC_MODEL) from .env
model = AnthropicVisionModel()
```

  <h4>Hugging Face (local/open-source)</h4>

```python
from splitter_mr.model import HuggingFaceVisionModel

# Token only if the model requires gating
model = HuggingFaceVisionModel()
```

</details>

## Step 2: Read the file using a VLM

All the implemented Readers support VLMs. To use these VLMs with the Readers, you only need to create the [`BaseReader`](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#basereader) classes with an object from [`BaseVisionModel`](https://andreshere00.github.io/Splitter_MR/api_reference/model/#basevisionmodel) as argument.

Firstly, we will use a [`VanillaReader`](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#vanillareader) class:

### Read a file using VanillaReader



```python
from splitter_mr.reader import VanillaReader
from splitter_mr.model import AzureOpenAIVisionModel

FILE_PATH = "data/pdfplumber_example.pdf"

model = AzureOpenAIVisionModel()
reader = VanillaReader(model=model)
reader_output = reader.read(file_path=FILE_PATH)

print(reader_output.text)
```

    <!-- page -->
    
    An example of a PDF file
    This is a PDF file
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam commodo egestas suscipit.
    Morbi sodales mi et lacus laoreet, eu molestie felis sodales. Aenean mattis gravida
    congue. Suspendisse bibendum malesuada volutpat. Nunc aliquam iaculis ex, sed
    sollicitudin lorem congue et. Pellentesque imperdiet ac sem ac imperdiet. Sed vel enim
    vitae orci scelerisque convallis quis ac purus.
    Cras sed neque vel justo auctor interdum a sit amet quam.
    ...
    disse potenti. Cras imperdiet enim vitae
    nunc elementum, non commodo ligula pretium. Vestibulum placerat nec tortor eu
    dapibus. Nullam et ipsum tortor. Nulla imperdiet enim velit, commodo facilisis elit
    tempus quis. Cras in interdum augue.
    
    <!-- image -->
    *Caption: A mysterious figure in a hoodie with glowing, round lenses, evoking a blend of futuristic technology and anonymity.*
    
    | It seems like | This is a table | But I am not sure |
    | --- | --- | --- |
    | About this | What do you think | ? |
    


!!! warning

    If you dont have the file locally, it is possible that instead of loading the content of the file, it will show only the document path. In order to avoid this behavior, please, use a correct file path on the file to be read. 


In this case we have read a PDF with an image at the end of the file. When reading the file and priting the content, we can see that the image has been described by the VLM.


When using a `VanillaReader` class, the image is highlighted with a `> **Caption**:` placeholder by default. But the prompt can be changed using the keyword argument `prompt`. For example, you can say that you want the Caption to be signalised as a comment `<!--- Caption: >:`


```python
from splitter_mr.reader import VanillaReader

PROMPT: str = "Describe the resource in a concise way: e.g., <!---- Caption: Image shows ...!--->:"

reader = VanillaReader(model=model)
reader_output = reader.read(file_path=FILE_PATH, prompt=PROMPT)

print(reader_output.text)
```

    <!-- page -->
    
    An example of a PDF file
    This is a PDF file
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam commodo egestas suscipit.
    Morbi sodales mi et lacus laoreet, eu molestie felis sodales. Aenean mattis gravida
    congue. Suspendisse bibendum malesuada volutpat. Nunc aliquam iaculis ex, sed
    sollicitudin lorem congue et. Pellentesque imperdiet ac sem ac imperdiet. Sed vel enim
    vitae orci scelerisque convallis quis ac purus.
    Cras sed neque vel justo auctor interdum a sit amet quam.
    ...
    or eu
    dapibus. Nullam et ipsum tortor. Nulla imperdiet enim velit, commodo facilisis elit
    tempus quis. Cras in interdum augue.
    
    <!-- image -->
    <!---- Caption: Image shows a figure wearing a teal hoodie, with their hands on their head and a black face featuring glowing circular eyes, set against a dark background. The figure's expression conveys a sense of surprise or shock. ---!>
    
    | It seems like | This is a table | But I am not sure |
    | --- | --- | --- |
    | About this | What do you think | ? |
    



### Read a file using MarkItDownReader

In this case, we will read an image file to provide a complete description. So, you simply instantiate the object and pass a model which inherits from a `BaseVisionModel` object.


```python
from splitter_mr.reader import MarkItDownReader

FILE_PATH = "data/chameleon.jpg"

md = MarkItDownReader(model=model)
md_reader_output = md.read(file_path=FILE_PATH, prompt=PROMPT)

print(md_reader_output.text)
```

    <!-- page -->
    
    # Description:
    <!---- Caption: Image shows a vibrant, colorful lizard peering out from a pink and orange floral background, showcasing its bright features and intricate details against a soft, blurred setting. ---!>
    



Original image is:

![Chameleon](https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/chameleon.jpg)


As we can see, `MarkItDownReader` provides a very complete but verbose description of the files that you provide. In addition, it is not capable to analyze the image contents inside a PDF. In contrast, you should provide the image separatedly. 

!!! warning
    You can NOT modify the prompt of the VLM in this method.

### Read the file using DoclingReader

The same process can be applied to DoclingReader. This time, we will analyze an invoice. So, the code is the following:


```python
from splitter_mr.reader import DoclingReader

FILE_PATH = "data/sunny_farm.pdf"

docling = DoclingReader(model=model)
docling_output = docling.read(file_path=FILE_PATH)

print(docling_output.text)
```

    <!-- image -->
    *Caption: A decorative golden banner, ideal for adding a classic touch to titles or announcements.*
    
    <!-- image -->
    *Caption: A vibrant logo for Sunny Farm, showcasing fresh produce from Victoria, Australia, with a sun emblem symbolizing freshness and quality.*
    
    ## 123 Somewhere St, Melbourne VIC 3000 (03) 1234 5678
    
    ## Denny Gunawan
    
    221 Queen St Melbourne VIC 3000
    
    $39.60
    
    Invoice Number: #20130304
    
    | Organic Items   | Price/kg   |   Quantity(kg) | Subtotal   |
    |----------------
    ...
     image -->
    *Caption: A bold and expressive typography design conveying gratitude, perfect for expressing appreciation and thanks.*
    
    * Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam sodales dapibus fermentum. Nunc adipiscing, magna sed scelerisque cursus, erat lectus dapibus urna, sed facilisis leo dui et ipsum.
    
    Subtotal
    
    $36.00
    
    GST (10%)
    
    $3.60
    
    Total
    
    $39.60
    
    <!-- image -->
    *Caption: A decorative brown banner, perfect for adding a rustic touch to announcements or displays.*



The result is pretty similar to the observed PDF (https://raw.githubusercontent.com/andreshere00/Splitter_MR/blob/main/data/sunny_farm.pdf)


As the same way as `VanillaReader`, you can change the prompt to provide larger descriptions or whatever you want to. For example:


```python
file = "data/sunny_farm.pdf"

docling = DoclingReader(model=model)
docling_output = docling.read(file, prompt="Provide a long description")

print(docling_output.text)
```

    <!-- image -->
    The image presents a decorative, elongated banner that has a rustic yet elegant appearance. The banner is designed in a subtle shade of gold, reminiscent of aged parchment or well-worn fabric. Its surface features a natural texture, giving it an organic and artisanal quality. The edges of the banner are slightly frayed, suggesting it has been hand-crafted, adding a touch of vintage charm to its overall aesthetic. 
    
    The banner curls gracefully at both ends, creating a sense of move
    ...
    could also serve as a visual focal point in designs, drawing the viewer’s eye and conveying messages of nostalgia, warmth, and the beauty of tradition.
    
    In summary, this exquisite ribbon captures the essence of storytelling and personal connection, making it a timeless element perfect for a myriad of creative applications. Its versatile design and beautiful color palette enable it to complement various themes and occasions, ensuring that it remains a cherished item in any decorative collection.


## Conclusion

Although all three methods can read files from various sources, they differ significantly in how VLM analysis is implemented:

* **`VanillaReader`** extracts graphical files from the input and uses a VLM to provide descriptions for these resources. Currently, it is only compatible with PDFs, and the VLM analysis and PDF reading logic are separated. It is the most scalable method for reading files, as it performs a call for every graphical resource in your PDF. However, this can become expensive for documents with a large number of images.

* **`MarkItDownReader`** can only transform images into Markdown descriptions. Supported image formats include `png`, `jpg`, `jpeg`, and `svg`. It cannot provide hybrid methods for reading PDFs with image annotations. While it is fast and cost-effective, it can only process one file at a time and is limited to OpenAI models.

* **`DoclingReader`** can read any file you provide using VLMs. If given a PDF, it reads the entire document with the VLM; the same applies to images and other graphical resources. However, it does not distinguish between text and image content, as the analysis is multimodal. As a result, in some cases, it cannot provide specific descriptions for images but instead analyzes the whole document.

Using one or another method depends on your needs!

In case that you want more information about available Models, visit [Developer guide](https://andreshere00.github.io/Splitter_MR/api_reference/model/). **Thank you for reading!**

## Complete script

```python
from splitter_mr.model import AzureOpenAIVisionModel
from splitter_mr.reader import DoclingReader, MarkItDownReader, VanillaReader

# Define the model
model = AzureOpenAIVisionModel()

# Readers

## Vanilla Reader

FILE_PATH = "data/pdfplumber_example.pdf"

reader = VanillaReader(model = model)
reader_output = reader.read(file_path = FILE_PATH)

print(reader_output.text)

PROMPT: str = "Describe the resource in a concise way: e.g., <!---- Caption: Image shows ...!--->:"

reader_output_with_dif_prompt = reader.read(
    FILE_PATH, 
    prompt = PROMPT
)

print(reader_output_with_dif_prompt.text)

## MarkItDown Reader

FILE_PATH = "data/chameleon.jpg"

md = MarkItDownReader(model = model)
md_reader_output = md.read(file_path = FILE_PATH)

print(md_reader_output.text)

## Docling Reader

FILE_PATH = "data/sunny_farm.pdf"

docling = DoclingReader(model = model)
docling_output = docling.read(
    file_path = FILE_PATH, 
    prompt = "Provide a long description"
)

print(docling_output.text)
```
