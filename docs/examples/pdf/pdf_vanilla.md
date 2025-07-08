# **Example:** Read PDF documents with images using Vanilla Reader

In this tutorial we will see how to read a PDF using our custom component, which is based on **PDFPlumber**. Then, we will connect this reader component into Visual Language Models to extract text or get annotations from images inside the PDF. In addition, we will explore which options we have to analyse and extract the content of the PDF in a custom, fast and a comprehensive way. Let's dive in.

!!! note
    Remember that you can access to the complete documentation of this Reader Component in the [**Developer Guide**](../../api_reference/reader.md#vanillareader).

## How to connect a VLM to MarkItDownReader

For this tutorial, we will use the same data as the first tutorial. [**Consult reference here**](https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sample_pdf.pdf).

Currently, two models are supported, both from OpenAI: the regular client, **OpenAI** and the available deployments from **Azure**. Hence, you can instantiate wherever you want to your project, or create a new one using as reference the [BaseModel abstract class](../../api_reference/model.md#basemodel).

Before instantiating the model, you should provide connection parameters. These connections parameters are loaded from environment variables (you can save them in a `.env` in the root of the project or script that you will execute). Consult these snippets:

<details> <summary><b>Environment variables definition</b></summary>
    
    <h3>For <code>OpenAI</code>:</h3>

    ```txt
    OPENAI_API_KEY=<your-api-key>
    ```

    <h3>For <code>AzureOpenAI</code>:</h3>

    ```txt
    AZURE_OPENAI_API_KEY=<your-api-key>
    AZURE_OPENAI_ENDPOINT=<your-endpoint>
    AZURE_OPENAI_API_VERSION=<your-api-version>
    AZURE_OPENAI_DEPLOYMENT=<your-model-name>
    ```
</details>

After that, you can explicitly declare the connection parameters as follows:

<details> <summary><code>OpenAI</code> and <code>AzureOpenAI</code> <b>implementation example</b></summary>

    <h3>For <code>OpenAI</code></h3>

    ```python
    import os
    from splitter_mr.model import OpenAIVisionModel

    api_key = os.getenv("OPENAI_API_KEY")

    model = OpenAIVisionModel(api_key=api_key)
    ```

    <h3>For <code>AzureOpenAI</code></h3>

    ```python
    import os
    from splitter_mr.model import AzureOpenAIVisionModel

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    model = AzureOpenAIVisionModel(
        azure_api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        azure_deployment=azure_deployment
    )
    ```
</details>

Or, alternatively, if you have saved the environment variables as indicated above, you can simply instantiate the model without explicit parameters. For this tutorial, we will use an `AzureOpenAI` deployment.

```python
from splitter_mr.model import AzureOpenAIVisionModel

model = AzureOpenAIVisionModel()
```

Then, use the Reader component and insert the model as parameter:

```python
from splitter_mr.reader import VanillaReader

reader = VanillaReader(model = model)
```

Then, you can read the file. The result will be an object from the type `ReaderOutput`, which is a dictionary containing some metadata about the file. To get the content, you can access to the `text` attribute:

```python
file = "data/sample_pdf.pdf"

output = reader.read(file_path = file)
print(output.text)
```

As observed, all the images have been described by the LLM:

```md

```

## Experimenting with some keyword arguments

Suppose that you need to simply get the base64 images from the file. Then, you can use the option `show_base64_images` to get those images:

```python
reader = VanillaReader()
output = reader.read(file_path = file, show_base64_images = True)
print(output.text)
```

In addition, you can modify how the image and page placeholders are generated with the options `image_placeholder` and `page_placeholder`:

```python
reader = VanillaReader()
output = reader.read(file_path = file, image_placeholder = "## Image", page_placeholder = "## Page")
print(output.text)
```

But one of the most important features is to scan the PDF as PageImages, to analyse every page with a VLM to extract the content. In order to do that, you can simply activate the option `scan_pdf_pages`. 

```python
reader = VanillaReader(model = model)
output = reader.read(file_path = file, scan_pdf_pages = True)
print(output.text)
```

Remember that you can always customize the prompt to get one or other results using the parameter `prompt`:

```python
reader = VanillaReader(model = model)
output = reader.read(file_path = file, prompt = "Extract the content of this resource in html format")
print(output.text)
```

To sum up, we can see that `VanillaReader` is a good option to extract rapidly and efficiently the text content for a PDF file. Remember that you can customize how the extraction is performed. But remember to consult other reading options in the [Developer guide](../../api_reference/reader.md) or [other tutorials](../examples.md).

Thank you so much for reading :).

## Complete script

```python
import os
from splitter_mr.reader import VanillaReader
from splitter_mr.model import AzureOpenAIVisionModel

file = "data/sample_pdf.pdf"
output_dir = "tmp/vanilla_output"
os.makedirs(output_dir, exist_ok=True)

model = AzureOpenAIVisionModel()

# 1. Default with model
reader = VanillaReader(model=model)
output = reader.read(file_path=file)
with open(os.path.join(output_dir, "output_with_model.txt"), "w", encoding="utf-8") as f:
    f.write(output.text)

# 2. Default without model, with base64 images shown
reader = VanillaReader()
output = reader.read(file_path=file, show_base64_images=True)
with open(os.path.join(output_dir, "output_with_base64_images.txt"), "w", encoding="utf-8") as f:
    f.write(output.text)

# 3. Default without model, with placeholders
reader = VanillaReader()
output = reader.read(file_path=file, image_placeholder="## Image", page_placeholder="## Page")
with open(os.path.join(output_dir, "output_with_placeholders.txt"), "w", encoding="utf-8") as f:
    f.write(output.text)

# 4. With model, scan PDF pages
reader = VanillaReader(model=model)
output = reader.read(file_path=file, scan_pdf_pages=True)
with open(os.path.join(output_dir, "output_scan_pdf_pages.txt"), "w", encoding="utf-8") as f:
    f.write(output.text)

# 5. With model, custom prompt
reader = VanillaReader(model=model)
output = reader.read(file_path=file, prompt="Extract the content of this resource in html format")
with open(os.path.join(output_dir, "output_html_prompt.txt"), "w", encoding="utf-8") as f:
    f.write(output.text)
```