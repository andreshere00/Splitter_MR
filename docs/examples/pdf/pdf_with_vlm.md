# **Example**: Reading files with Visual Language Models to Provide Image Annotations

When reading a PDF file or other files which contains images, it can be useful to provide descriptive text alongside those images. Since images in a Markdown file are typically rendered by encoding them in base64 format, you may alternatively want to include a description of each image instead. 

This is where **Visual Language Models (VLMs)** come in—to analyze and describe images automatically. In this tutorial, we'll show how to use these models with the library.

## Step 1: Load a Model

Currently, two models are supported: one from OpenAI and one from an Azure OpenAI deployment. After choosing a model, you simply need to instantiate the `BaseModel` class, which implements one of these VLMs.

Before that, you should provide some environment variables (these variables should be saved in a `.env` file in the directory where the Python script will be executed):

<details> <summary>Environment variables definition</summary>
    
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

So, the models can be loaded as follows:

<details> <summary><code>OpenAI</code> and <code>AzureOpenAI</code> implementation example</summary>

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

## Step 2: Read the file using a VLM

All the implemented Readers support VLMs. To use these VLMs with the Readers, you only need to create the `BaseReader` classes with an object from `BaseModel` as argument. Firstly, we will use a `VanillaReader` class:

### Read a file using VanillaReader

```python
from splitter_mr.reader import VanillaReader

file = "data/pdfplumber_example.pdf"

reader = VanillaReader(model = model)
reader_output = reader.read(file)

print(reader_output.text)
```

In this case we have read a PDF with an image at the end of the file. When reading the file and priting the content, we can see that the image has been described by the VLM:

```md
---
## Page 1
---

An example of a PDF file
This is a PDF file
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam commodo egestas suscipit.
Morbi sodales mi et lacus laoreet, eu molestie felis sodales. Aenean mattis gravida
congue. Suspendisse bibendum malesuada volutpat. Nunc aliquam iaculis ex, sed
sollicitudin lorem congue et. Pellentesque imperdiet ac sem ac imperdiet. Sed vel enim
vitae orci scelerisque convallis quis ac purus.
Cras sed neque vel justo auctor interdum a sit amet quam. Curabitur rhoncus, ligula a
lacinia euismod, mi nunc vestibulum erat, vitae laoreet neque lorem quis mi. Phasellus
eu nunc in orci sagittis faucibus. Donec eget luctus sem, sit amet viverra neque.
Curabitur pulvinar velit rhoncus mauris sodales, vitae bibendum augue vestibulum.
Mauris porta, enim ut pellentesque bibendum, augue dui finibus nulla, et laoreet magna
nisi eu magna. Mauris sit amet semper leo, vitae malesuada turpis. Nunc arcu felis,
consequat in congue at, iaculis at ligula. Suspendisse potenti. Cras imperdiet enim vitae
nunc elementum, non commodo ligula pretium. Vestibulum placerat nec tortor eu
dapibus. Nullam et ipsum tortor. Nulla imperdiet enim velit, commodo facilisis elit
tempus quis. Cras in interdum augue.

> **Caption:** A mysterious figure in a glowing-eyed hoodie stands against a dark backdrop, blending elements of intrigue and futurism.

| It seems like | This is a table | But I am not sure |
| --- | --- | --- |
| About this | What do you think | ? |
```

When using a `VanillaReader` class, the image is highlighted with a `> **Caption**:` placeholder by default. But the prompt can be changed using the keyword argument `prompt`. For example, you can say that you want the Caption to be signalised as a comment `<!--- Caption: >:`

```python
from splitter_mr.reader import VanillaReader

file = "data/pdfplumber_example.pdf"

reader = VanillaReader(model = model)
reader_output = reader.read(file, prompt = "Describe the resource in a concise way: e.g., <!--- Caption: Image shows ... {DESCRIPTION}>:")

print(reader_output.text)
```

```md
---
## Page 1
---

An example of a PDF file
This is a PDF file
...
Cras in interdum augue.

<!---- Caption: Image shows a person wearing a black mask and a light blue hoodie, with glowing round eyes, striking a mysterious pose against a dark background.!--->

| It seems like | This is a table | But I am not sure |
| --- | --- | --- |
| About this | What do you think | ? |
```

### Read a file using MarkItDownReader

In this case, we will read an image file to provide a complete description. So, you simply instantiate the object and pass a model which inherits from a `BaseModel` object.

```python
from splitter_mr.reader import MarkItDownReader

file = 

md = MarkItDownReader(model = model)
md_reader_output = md.read(file)

print(md_reader_output.text)
```

Original image is:

![Chameleon](https://raw.githubusercontent.com/andreshere00/Splitter_MR/blob/main/data/chameleon.jpg)

And the description is:

```md
# Description:
In this captivating close-up, we see a vibrant lizard peering curiously from the embrace of a blooming flower. The intricate details of its scales shimmer in a range of mesmerizing colors, from vivid turquoise to rich reds. Its large, expressive eyes reflect a sense of wonder as it engages with its blooming surroundings, seemingly unbothered by the colorful petals that cradle it. The soft, blurred background, awash in dreamy hues of pink and yellow, contrasts beautifully with the lizard’s eye-catching palette, enhancing the ethereal quality of the composition. This image invites us to appreciate the delicate balance of nature, showcasing how creatures from different realms come together in splendor, creating a harmonious moment in the natural world. With every glance, one can’t help but marvel at the beauty of biodiversity and the intriguing interactions that occur in hidden corners of our environment.
```

As we can see, `MarkItDownReader` provides a very complete but verbose description of the files that you provide. In addition, it is not capable to analyze the image contents inside a PDF. In contrast, you should provide the image separatedly. 

!!! warning
    You can NOT modify the prompt of the VLM in this method.

### Read the file using DoclingReader

The same process can be applied to DoclingReader. This time, we will analyze an invoice. So, the code is the following:

```python
file = "data/sunny_farm.pdf"

docling = DoclingReader(model = model)
docling_output = docling.read(file)

print(docling_output.text)
```

The result is pretty similar to the observed PDF (https://raw.githubusercontent.com/andreshere00/Splitter_MR/blob/main/data/sunny_farm.pdf)

```md
# Sunny Farm Invoice

Attention To: Denny Gunawan 221 Queen St Melbourne VIC 3000  

Total Amount:  $39.60  

Invoice Number:  #20130304  

### Organic Items

| Item       | Price/kg   |   Quantity (kg) | Subtotal   |
|------------|------------|-----------------|------------|
| Apple      | $5.00      |               1 | $5.00      |
| Orange     | $1.99      |               2 | $3.98      |
| Watermelon | $1.69      |               3 | $5.07      |
| Mango      | $9.56      |               2 | $19.12     |
| Peach      | $2.99      |               1 | $2.99      |

Subtotal:  $36.00 GST (10%):  $3.60 Total:  $39.60  

Thank You! Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam sodales dapibus fermentum. Nunc adipiscing, magna sed scelerisque cursus, erat lectus dapibus urna, sed facilisis leo dui et ipsum.
```

As the same way as `VanillaReader`, you can change the prompt to provide larger descriptions or whatever you want to. For example:

```python
file = "data/sunny_farm.pdf"

docling = DoclingReader(model = model)
docling_output = docling.read(file, prompt = "Provide a long description")

print(docling_output.text)
```

```md
The document is an invoice from "Sunny Farm," a fresh produce supplier located in Victoria, Australia. The top section of the invoice features the company's logo, which includes a sun rising over a field, symbolizing freshness and agriculture. Below the logo, the company's address is listed as "123 Somewhere St, Melbourne VIC 3000," along with a contact phone number, "(03) 1234 5678."

The invoice is addressed to a customer named Denny Gunawan, whose address is provided as "221 Queen St, Melbourne VIC 3000." The total amount due is prominently displayed in a large font, indicating a total of $39.60.

The invoice includes an invoice number, "#20130304," and is organized into a table that details the organic items purchased. The table has several columns: "Organic Items," "Price/kg," "Quantity (kg)," and "Subtotal." 

The items listed are as follows:

[&lt;RawText children='Apple'&gt;]

[&lt;RawText children='Orange'&gt;]

[&lt;RawText children='Watermelon'&gt;]

[&lt;RawText children='Mango'&gt;]

[&lt;RawText children='Peach'&gt;]

The subtotal for all items is calculated to be $36.00. Below the itemized list, the document specifies a GST (Goods and Services Tax) of 10%, which amounts to $3.60. The total amount due, which combines the subtotal and the GST, is again stated as $39.60.

At the bottom of the invoice, there is a "THANK YOU" message, indicating appreciation for the customer's business. Additionally, there is a note marked with an asterisk that contains placeholder text, suggesting that it could be replaced with specific terms and conditions or additional information relevant to the transaction.

Overall, the invoice is well-structured, providing clear information about the purchase, pricing, and total amount due, while also conveying a friendly and professional tone.
```

Here’s a corrected and slightly clarified version:

---

## Conclusion

Although all three methods can read files from various sources, they differ significantly in how VLM analysis is implemented:

* **`VanillaReader`** extracts graphical files from the input and uses a VLM to provide descriptions for these resources. Currently, it is only compatible with PDFs, and the VLM analysis and PDF reading logic are separated. It is the most scalable method for reading files, as it performs a call for every graphical resource in your PDF. However, this can become expensive for documents with a large number of images.

* **`MarkItDownReader`** can only transform images into Markdown descriptions. Supported image formats include `png`, `jpg`, `jpeg`, and `svg`. It cannot provide hybrid methods for reading PDFs with image annotations. While it is fast and cost-effective, it can only process one file at a time and is limited to OpenAI models.

* **`DoclingReader`** can read any file you provide using VLMs. If given a PDF, it reads the entire document with the VLM; the same applies to images and other graphical resources. However, it does not distinguish between text and image content, as the analysis is multimodal. As a result, in some cases, it cannot provide specific descriptions for images but instead analyzes the whole document.

Again, using one or another method depends on your needs!

In case that you want more information about available Models, visit [Developer guide](../../api_reference/model.md). **Thank you for reading!**

## Complete script

```python
from markitdown import MarkItDown
from openai import AzureOpenAI

from splitter_mr.model import AzureOpenAIVisionModel
from splitter_mr.reader import DoclingReader, MarkItDownReader, VanillaReader

# Define the model
model = AzureOpenAIVisionModel()

# Readers

## Vanilla Reader

file = "data/pdfplumber_example.pdf"

reader = VanillaReader(model = model)
reader_output = reader.read(file)

print(reader_output.text)

reader_output_with_dif_prompt = reader.read(file, prompt = "Describe the resource in a concise way: e.g., <!---- Caption: Image shows ...!--->:")

print(reader_output_with_dif_prompt.text)

## MarkItDown Reader

file = "data/chameleon.jpg"

md = MarkItDownReader(model = model)
md_reader_output = md.read(file)

print(md_reader_output.text)

## Docling Reader

file = "data/sunny_farm.pdf"

docling = DoclingReader(model = model)
docling_output = docling.read(file, prompt = "Provide a long description")

print(docling_output.text)
```