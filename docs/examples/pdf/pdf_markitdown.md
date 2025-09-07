# **Example:** Reading PDF Documents with Images using MarkItDownReader

<p style="text-align:center;">
<img src="https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/markitdown_reader_button.svg#only-light" alt="MarkItDownReader logo">
<img src="https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/markitdown_reader_button_white.svg#only-dark" alt="MarkItDownReader logo">
</p>

As we have seen in previous examples, reading a PDF is not a simple task. In this case, we will see how to read a PDF using the **MarkItDown** framework, and connect this library to Visual Language Models (VLMs) to extract text or get annotations from images.

## How to connect a VLM to MarkItDownReader

For this example, we will use the same document as the [previous tutorial](https://github.com/andreshere00/Splitter_MR/blob/main/data/sample_pdf.pdf).

To extract image descriptions or perform OCR, instantiate any model that implements the [`BaseModel` interface](https://andreshere00.github.io/Splitter_MR/api_reference/model/#basemodel) (vision variants inherit from it) and pass it into the [`MarkItDownReader`](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#markitdownreader). Swapping providers only changes the model constructor; your Reader usage remains the same.

### Supported models (and when to use them)

| Model (docs)                                                                                                       | When to use                                       | Required environment variables                                                                                        |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| [`OpenAIVisionModel`](https://andreshere00.github.io/Splitter_MR/api_reference/model/#openaivisionmodel)           | You have an OpenAI API key and want OpenAI cloud. | `OPENAI_API_KEY` (optional: `OPENAI_MODEL`, defaults to `gpt-4o`)                                                     |
| [`AzureOpenAIVisionModel`](https://andreshere00.github.io/Splitter_MR/api_reference/model/#azureopenaivisionmodel) | You use Azure OpenAI Service.                     | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`                |
| [`GrokVisionModel`](https://andreshere00.github.io/Splitter_MR/api_reference/model/#grokvisionmodel)               | You have access to xAI Grok multimodal.           | `XAI_API_KEY` (optional: `XAI_MODEL`, default `grok-4`)                                                               |
| [`GeminiVisionModel`](https://andreshere00.github.io/Splitter_MR/api_reference/model/#geminivisionmodel)           | You want Google’s Gemini vision models.           | `GEMINI_API_KEY` (also install extras: `pip install "splitter-mr[multimodal]"`)                                       |
| [`AnthropicVisionModel`](https://andreshere00.github.io/Splitter_MR/api_reference/model/#anthropicvisionmodel)     | You have an Anthropic key (Claude Vision).        | `ANTHROPIC_API_KEY` (optional: `ANTHROPIC_MODEL`)                                                                     |
| [`HuggingFaceVisionModel`](https://andreshere00.github.io/Splitter_MR/api_reference/model/#huggingfacevisionmodel) | You prefer local/open-source/offline inference.   | Install extras: `pip install "splitter-mr[multimodal]"` (optional: `HF_ACCESS_TOKEN` if the chosen model requires it) |

> **Note on HuggingFace models:** Not all HF models are supported (e.g., gated or uncommon architectures). A well-tested option is **SmolDocling**.

### Environment variables

Create a `.env` file alongside your Python script:

<details>
  <summary><strong>Show/hide environment variables needed for every provider</strong></summary>

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
  <summary><strong>Show/hide instantiation snippets for all providers</strong></summary>

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



```python
from splitter_mr.model import AzureOpenAIVisionModel
from splitter_mr.reader import MarkItDownReader

file = "data/sample_pdf.pdf"
model = AzureOpenAIVisionModel()
```


Then, you can simply pass the model that you have instantiated to the Reader class:


```python
reader = MarkItDownReader(model=model)
output = reader.read(file)
```


This returns a `ReaderOutput` object with all document text and extracted image descriptions via the vision model. You can access metadata like `output.conversion_method`, `output.reader_method`, `output.ocr_method`, etc.

To retrieve the text content, you can simply access to the `text` attribute:


```python
print(output.text)
```

    <!-- page -->
    
    # Description:
    # A sample PDF
    
    Converting PDF files to other formats, such as Markdown, is a surprisingly complex task due to the nature of the PDF format itself. PDF (Portable Document Format) was designed primarily for preserving the visual layout of documents, making them look the same across different devices and platforms. However, this design goal introduces several challenges when trying to extract and convert the underlying content into a more flexible, structured format l
    ...
    simple on the surface, converting PDFs to formats like Markdown involves a series of technical and interpretive challenges. Effective conversion tools must blend text extraction, document analysis, and sometimes machine learning techniques (such as OCR or structure recognition) to produce usable, readable, and faithful Markdown output. As a result, perfect conversion is rarely possible, and manual review and cleanup are often required.
    
    ![Hummingbird](https://example.com/hummingbird-image)
    ```
    



With the by-default method, you obtain the text extracted from the PDF as it is shown. This method scan the PDF pages as images and process them using a VLM. The result will be a markdown text with all the images detected in every page. Every page is highlighted with a markdown comment as a placeholder: `<!-- page -->`. 

## Experimenting with some keyword arguments

In case that needed, you can pass use other keyword arguments to process the PDFs.

For example, you can customize how to process the images by the VLM using the parameter prompt. For example, in case that you only need an excerpt or a brief description for every page, you can use the following prompt:

```python
output = reader.read(
    file, 
    scan_pdf_pages = True, 
    prompt = "Return only a short description for these pages"
)
```


In case that needed, it could be interesting split the PDF pages using another placeholder. You can configure that using the `page_placeholder` parameter:


```python
output = reader.read(
    file,
    scan_pdf_pages=True,
    prompt="Return only a short description for these pages",
    page_placeholder="## PAGE",
)
print(output.text)
```

    ## PAGE
    
    # Description:
    The document discusses the challenges of converting PDF files to more flexible formats like Markdown due to the inherent characteristics of PDFs. It highlights two main issues: the lack of structural information, which complicates the extraction of organized content, and the variability in PDF content types, leading to inconsistent results during conversion.
    
    ## PAGE
    
    # Description:
    1. **Preservation of Formatting**: Discusses the limitations of Markdown in replicating co
    ...
    d special characters in PDFs, focusing on the risks of data loss and corruption during extraction.
    
    ## PAGE
    
    # Description:
    The page summarizes the challenges of converting PDFs to Markdown formats, emphasizing the need for advanced tools that integrate text extraction, document analysis, and machine learning. It concludes that perfect conversion is difficult and often requires manual review and adjustments. Additionally, a table of team members with their roles and contact emails is included.
    



In comparison, `MarkItDownReader` offers a faster conversion than Docling but with less options to be configured. In that sense, we cannot obtain directly the `base64` images from every image detected in our documents, or write image placeholders easily (despite we can do it using a prompt). In addition, you will always get a `# Description` placeholder every time you use a VLM for extraction and captioning in this Reader. 

As conclusion, using this reader with a VLM can be useful for those use cases when we need to efficiently extract the text from a PDF. In case that you need the highest reliability or customization, it is not the most suitable option.

## Complete script

```python
import os

from splitter_mr.model import AzureOpenAIVisionModel
from splitter_mr.reader import MarkItDownReader
from dotenv import load_dotenv

load_dotenv()

file = "data/sample_pdf.pdf"
model = AzureOpenAIVisionModel()
# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "markitdown_output")
os.makedirs(output_dir, exist_ok=True)

def save_markdown(output, filename_base):
    """
    Saves the ReaderOutput.text attribute to a markdown file in the markitdown_output directory.

    Args:
        output (ReaderOutput): The result object returned from DoclingReader.read().
        filename_base (str): A descriptive base name for the file (e.g., 'vlm', 'scan_pages').
    """
    filename = f"{filename_base}.md"
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(output.text)
    print(f"Saved: {file_path}")

markitdown_reader = MarkItDownReader(model = model)
markitdown_output = markitdown_reader.read(file)
save_markdown(markitdown_output, "vlm")

markitdown_output = markitdown_reader.read(file, scan_pdf_pages = True, prompt = "Return only a short description for these pages", page_placeholder = "## PAGE")
save_markdown(markitdown_output, "custom_vlm")

markitdown_reader = MarkItDownReader()
markitdown_output = markitdown_reader.read(file)
save_markdown(markitdown_output, "no_vlm")
```


!!! note
    For more on available options, see the [**MarkItDownReader class documentation**](../../api_reference/reader.md#markitdownreader).
