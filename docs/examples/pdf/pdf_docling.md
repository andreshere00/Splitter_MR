# **Example:** Read PDF documents with images using Docling Reader

<p style="text-align:center;">
<img src="https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/docling_reader_button.svg#only-light" alt="DoclingReader logo">
<img src="https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/docling_reader_button_white.svg#only-dark" alt="DoclingReader logo">
</p>

As we have seen in previous examples, reading a PDF is not a simple task. In this case, we will see how to read a PDF using the Docling framework, and connect this library into Visual Language Models to extract text or get annotations from images.

## Connecting to a VLM to extract text and analyze images

For this example, we will use the same document as the [previous tutorial](https://github.com/andreshere00/Splitter_MR/blob/main/data/sample_pdf.pdf).

To use a VLM to read images and get annotations, instantiate any model that implements the [`BaseModel` interface](https://andreshere00.github.io/Splitter_MR/api_reference/model/#basemodel) (vision variants inherit from it) and pass it into the [`VanillaReader`](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#vanillareader). Swapping providers only changes the model constructor; your Reader usage remains the same.

### Supported models (and when to use them)

| Model (docs)                                                                                                       | When to use                                       | Required environment variables                                                                                        |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| [`OpenAIVisionModel`](https://andreshere00.github.io/Splitter_MR/api_reference/model/#openaivisionmodel)           | You have an OpenAI API key and want OpenAI cloud. | `OPENAI_API_KEY` (optional: `OPENAI_MODEL`, defaults to `gpt-5.6-luna`)                                                     |
| [`AzureOpenAIVisionModel`](https://andreshere00.github.io/Splitter_MR/api_reference/model/#azureopenaivisionmodel) | You use Azure OpenAI Service.                     | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`                |
| [`GrokVisionModel`](https://andreshere00.github.io/Splitter_MR/api_reference/model/#grokvisionmodel)               | You have access to xAI Grok multimodal.           | `XAI_API_KEY` (optional: `XAI_MODEL`, default `grok-4.5`)                                                               |
| [`GeminiVisionModel`](https://andreshere00.github.io/Splitter_MR/api_reference/model/#geminivisionmodel)           | You want Google’s Gemini vision models.           | `GEMINI_API_KEY` (also install extras: `pip install "splitter-mr[multimodal]"`)                                       |
| [`AnthropicVisionModel`](https://andreshere00.github.io/Splitter_MR/api_reference/model/#anthropicvisionmodel)     | You have an Anthropic key (Claude Vision).        | `ANTHROPIC_API_KEY` (optional: `ANTHROPIC_MODEL`)                                                                     |
| [`HuggingFaceVisionModel`](https://andreshere00.github.io/Splitter_MR/api_reference/model/#huggingfacevisionmodel) | You prefer local/open-source/offline inference.   | Install extras: `pip install "splitter-mr[multimodal]"` (optional: `HF_ACCESS_TOKEN` if the chosen model requires it) |

> **Note on HuggingFace models:** Not all HF models are supported (e.g., gated or uncommon architectures). A well-tested option is **Granite Docling**.

### Environment variables

Create a `.env` file alongside your Python script:

<details>
  <summary><strong>Show/hide environment variables needed for every provider</strong></summary>

  <h4>OpenAI</h4> 
```txt
# OpenAI
OPENAI_API_KEY=<your-api-key>
# (optional) OPENAI_MODEL=gpt-5.6-luna
```

  <h4>OpenRouter</h4>

```txt
# OpenRouter
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_API_VERSION=<your-api-version>
AZURE_OPENAI_DEPLOYMENT=<your-model-name>
```

  <h4>xAI Grok</h4>

```txt
# xAI Grok
XAI_API_KEY=<your-api-key>
# (optional) XAI_MODEL=grok-4.5
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
# (optional) ANTHROPIC_MODEL=claude-haiku-4-5
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
# model = OpenAIVisionModel(api_key="...", model_name="gpt-5.6-luna")
```

  <h4>OpenRouter</h4>

```python
from splitter_mr.model import OpenRouterVisionModel

# Reads OPENROUTER_API_KEY (and optional OPENROUTER_MODEL) from .env if present
model = OpenRouterVisionModel()
# or pass explicitly:
# model = OpenRouterVisionModel(api_key="...", model_name="openai/gpt-5.6-luna")
```

  <h4>Azure OpenAI</h4>

```python
from splitter_mr.model import AzureOpenAIVisionModel

model = AzureOpenAIVisionModel()
# or:
# model = AzureOpenAIVisionModel(
#     api_key="...",
#     azure_endpoint="https://<resource>.openai.azure.com/",
#     api_version="2024-02-15-preview",
#     azure_deployment="gpt-5.6-luna",
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
from splitter_mr.model import OpenRouterVisionModel
from splitter_mr.reader import DoclingReader
from dotenv import load_dotenv
import os

load_dotenv()

ROOT_PATH: str = os.getenv("ROOT_PATH") or "."
load_dotenv(os.path.join(ROOT_PATH, ".env"))
FILE_PATH: str = f"{ROOT_PATH}/data/sample_pdf.pdf"

model = OpenRouterVisionModel()
```


Then, use the `read` method of this object and read a file as always. Once detected that the file is PDF, it will return a ReaderOutput object containing the extracted text.


```python
# 1. Read PDF using a Visual Language Model

print("=" * 80 + " DoclingReader with VLM " + "=" * 80)
docling_reader = DoclingReader(model=model)
docling_output = docling_reader.read(FILE_PATH)

# Get Docling ReaderOutput
print(docling_output.model_dump_json(indent=4))
```

    2026-08-30 18:58:35,615 - INFO - detected formats: [<InputFormat.PDF: 'pdf'>]


    ================================================================================ DoclingReader with VLM ================================================================================


    2026-08-30 18:58:37,043 - INFO - Going to convert document batch...


    2026-08-30 18:58:37,044 - INFO - Initializing pipeline for StandardPdfPipeline with options hash 4086a820d51c119ab30bfe51e86a0aeb


    2026-08-30 18:58:37,077 - INFO - Loading plugin 'docling_defaults'


    2026-08-30 18:58:37,096 - INFO - Registered ocr engines: ['easyocr', 'ocrmac', 'rapidocr', 'tesserocr', 'tesseract']


    2026-08-30 18:58:38,013 - INFO - Accelerator device: 'mps'


    2026-08-30 18:58:41,133 - INFO - Accelerator device: 'mps'


    2026-08-30 18:58:42,708 - INFO - Accelerator device: 'mps'


    2026-08-30 18:58:43,329 - INFO - Loading plugin 'docling_defaults'


    2026-08-30 18:58:43,346 - INFO - Registered picture descriptions: ['vlm', 'api']


    2026-08-30 18:58:43,347 - INFO - Processing document sample_pdf.pdf


    2026-08-30 18:58:54,078 - INFO - Finished converting document sample_pdf.pdf in 18.46 sec.


    2026-08-30 18:58:55,153 - INFO - HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"


    2026-08-30 18:58:57,443 - INFO - HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"


    {
        "text": "## A sample PDF\n\nConverting PDF files to other formats, such as Markdown, is a surprisingly complex task due to the nature of the PDF format itself . PDF (Portable Document Format) was designed primarily for preserving the visual layout of documents, making them look the same across different devices and platforms. However, this design goal introduces several challenges when trying to extract and convert the underlying content into a more flexible, structured format like Markdow
    ...
    lice@example.com |\n| Bob Johnson | Designer     | bob@example.com   |\n| Carol White | Project Lead | carol@example.com |",
        "document_name": "sample_pdf.pdf",
        "document_path": "/Users/aherencia/Documents/Projects/Professional/Splitter_MR/data/sample_pdf.pdf",
        "document_id": "322fe0a8-abec-4a73-aef9-c0063f84cc83",
        "conversion_method": "markdown",
        "reader_method": "docling",
        "ocr_method": "openai/gpt-5.6-luna",
        "page_placeholder": "<!-- page -->",
        "metadata": {}
    }



As we can see, the PDF contents along with some metadata information such as the `conversion_method`, `reader_method` or the `ocr_method` have been retrieved. To get the PDF contents, you can simply access to the `text` attribute as always:


```python
# Get text attribute from Docling Reader
print(docling_output.text)
```

    ## A sample PDF
    
    Converting PDF files to other formats, such as Markdown, is a surprisingly complex task due to the nature of the PDF format itself . PDF (Portable Document Format) was designed primarily for preserving the visual layout of documents, making them look the same across different devices and platforms. However, this design goal introduces several challenges when trying to extract and convert the underlying content into a more flexible, structured format like Markdown.
    
    <!-- image --
    ...
    nversion is rarely possible, and manual review and cleanup are often required.
    
    <!-- image -->
    *Caption: A vibrant hummingbird hovers beside delicate orange flowers, its iridescent turquoise feathers and outstretched wings captured in motion.*
    
    | Name        | Role         | Email             |
    |-------------|--------------|-------------------|
    | Alice Smith | Developer    | alice@example.com |
    | Bob Johnson | Designer     | bob@example.com   |
    | Carol White | Project Lead | carol@example.com |



As seen, all the images have been described using a caption. 

## Experimenting with some keyword arguments

In case that you have additional requirements to describe these images, you can provide a prompt via a `prompt` argument:


```python
docling_output = docling_reader.read(
    FILE_PATH, prompt="Describe the image briefly in Spanish."
)

print(docling_output.text)
```

    2026-08-30 18:58:57,960 - INFO - detected formats: [<InputFormat.PDF: 'pdf'>]


    2026-08-30 18:58:57,963 - INFO - Going to convert document batch...


    2026-08-30 18:58:57,963 - INFO - Initializing pipeline for StandardPdfPipeline with options hash 4086a820d51c119ab30bfe51e86a0aeb


    2026-08-30 18:58:57,963 - INFO - Accelerator device: 'mps'


    2026-08-30 18:59:00,226 - INFO - Accelerator device: 'mps'


    2026-08-30 18:59:01,416 - INFO - Accelerator device: 'mps'


    2026-08-30 18:59:01,977 - INFO - Processing document sample_pdf.pdf


    2026-08-30 18:59:03,684 - INFO - Finished converting document sample_pdf.pdf in 5.73 sec.


    2026-08-30 18:59:04,407 - INFO - HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"


    2026-08-30 18:59:05,595 - INFO - HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"


    ## A sample PDF
    
    Converting PDF files to other formats, such as Markdown, is a surprisingly complex task due to the nature of the PDF format itself . PDF (Portable Document Format) was designed primarily for preserving the visual layout of documents, making them look the same across different devices and platforms. However, this design goal introduces several challenges when trying to extract and convert the underlying content into a more flexible, structured format like Markdown.
    
    <!-- image --
    ...
    , and faithful Markdown output. As a result, perfect conversion is rarely possible, and manual review and cleanup are often required.
    
    <!-- image -->
    Un colibrí de colores brillantes vuela junto a unas flores naranjas, con las alas extendidas.
    
    | Name        | Role         | Email             |
    |-------------|--------------|-------------------|
    | Alice Smith | Developer    | alice@example.com |
    | Bob Johnson | Designer     | bob@example.com   |
    | Carol White | Project Lead | carol@example.com |



You can read the PDF scanning the pages as images and extracting its content. To do so, enable the option `scan_pdf_pages = True`. In case that you want to change the placeholder, you can do it passing the keyword argument `placeholder = <your desired placeholder>`.

Finally, it could be interesting extract the markdown text with the images as embedded content. In that case, activate the option `show_base64_images`. In that case, it is not necessary to pass the model to the Reader class.


```python
docling_reader = DoclingReader()
docling_output = docling_reader.read(FILE_PATH, show_base64_images=True)

print(docling_output.text)
```

    2026-08-30 18:59:06,500 - INFO - detected formats: [<InputFormat.PDF: 'pdf'>]


    2026-08-30 18:59:06,504 - INFO - Going to convert document batch...


    2026-08-30 18:59:06,504 - INFO - Initializing pipeline for StandardPdfPipeline with options hash e3309ea8218dc3b978b4932281c99b2a


    2026-08-30 18:59:06,505 - INFO - Accelerator device: 'mps'


    2026-08-30 18:59:08,706 - INFO - Accelerator device: 'mps'


    2026-08-30 18:59:09,671 - INFO - Accelerator device: 'mps'


    2026-08-30 18:59:09,917 - INFO - Processing document sample_pdf.pdf


    2026-08-30 18:59:11,399 - INFO - Finished converting document sample_pdf.pdf in 4.90 sec.


    ## A sample PDF
    
    Converting PDF files to other formats, such as Markdown, is a surprisingly complex task due to the nature of the PDF format itself . PDF (Portable Document Format) was designed primarily for preserving the visual layout of documents, making them look the same across different devices and platforms. However, this design goal introduces several challenges when trying to extract and convert the underlying content into a more flexible, structured format like Markdown.
    
    ![Image](data
    ...
    VmVpiJZTp+1FIM0Uj2JxZpUwxbOP3ZfuHh0YI7cx7xtQDKvKiAXbMIavwfuiik4LgTqaLMqHYx1YHJk6k9LmH4vC7CemDpsDy27Z29RwpQOi7w2JF1gkqMOWWH8LHq0ChPIsQzgKjXKO3JnLu49DQD1e2HGdS49mKyOq/W+W/sVFF/OISV+PBnUgGhCAC2SmpXYyJZ6KFKv7gsXW7v3P7/AOfCIRvcDUjcAAAAAElFTkSuQmCC)
    
    | Name        | Role         | Email             |
    |-------------|--------------|-------------------|
    | Alice Smith | Developer    | alice@example.com |
    | Bob Johnson | Designer     | bob@example.com   |
    | Carol White | Project Lead | carol@example.com |



Of course, remember that the use of a VLM is not mandatory, and you can read the PDF obtaining most of the information.

## Complete script

```python
from splitter_mr.model import AzureOpenAIVisionModel
from splitter_mr.reader import DoclingReader
import os

from dotenv import load_dotenv

load_dotenv()

ROOT_PATH: str = os.getenv("ROOT_PATH") or "."
load_dotenv(os.path.join(ROOT_PATH, ".env"))
FILE_PATH: str = f"{ROOT_PATH}/data/sample_pdf.pdf"


model = OpenRouterVisionModel()
docling_reader = DoclingReader(model = model)

# 1. Read PDF using a Visual Language Model

docling_output = docling_reader.read(FILE_PATH)
print(docling_output.model_dump_json(indent=4))  # Get Docling ReaderOutput
print(docling_output.text)  # Get text attribute from Docling Reader

# 2. Describe the images using a custom prompt

docling_output = docling_reader.read(FILE_PATH, prompt = "Describe the image briefly in Spanish.")
print(docling_output.text)

# 3. Scan PDF pages 

docling_output = docling_reader.read(FILE_PATH, scan_pdf_pages = True)
print(docling_output.text)

# 4. Extract images as embedded content

docling_reader = DoclingReader()
docling_output = docling_reader.read(FILE_PATH, show_base64_images = True)
print(docling_output.text)
```


!!! note
    For more on available options, see the [**DoclingReader class documentation**](../../api_reference/reader.md#doclingreader).
