# **Vision Models**

Reading documents like Word, PDF, or PowerPoint can sometimes be complicated if they contain images. To avoid this problem, **you can use visual language models (VLMs), which are capable of recognizing images and extracting descriptions from them**. In this prospectus, a model module has been developed, the implementation of which is based on the [**BaseVisionModel**](#basevisionmodel) class. It is presented below.

## Using `analyze_content`

Every concrete vision model implements **`analyze_content`** to send a text prompt and a base64-encoded image (no `data:` prefix) to the provider and return the model’s text response. Pass the instance to a [**Reader**](./reader.md) as `model=...`, or call it directly when you already have encoded image bytes.

```python
import base64

from splitter_mr.model import OpenRouterVisionModel

with open("diagram.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

model = OpenRouterVisionModel()  # OPENROUTER_API_KEY; optional OPENROUTER_MODEL
description = model.analyze_content(
    file=image_b64,
    prompt="Describe this image in one paragraph.",
    file_ext="png",
)
print(description)
```

Install the multimodal extra when the provider uses optional SDK dependencies: `pip install 'splitter-mr[multimodal]'`.

## Which model should I use?

The choice of model depends on your cloud provider, available API keys, and desired level of integration.
All models inherit from [**BaseVisionModel**](#basevisionmodel) and implement **`analyze_content`** for image prompts and extraction.

| Model                                                 | When to use                                         | Requirements                                                                                                   | Features                                                                                                     |
| ----------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| [**OpenRouterVisionModel**](#openroutervisionmodel)   | **Recommended:** one API key for many vision models via OpenRouter | `OPENROUTER_API_KEY` (optional: `OPENROUTER_MODEL`, defaults to `"openai/gpt-5.6-luna"`; optional attribution: `OPENROUTER_SITE_URL`, `OPENROUTER_APP_NAME`) | OpenAI SDK with OpenRouter base URL; any vision-capable model slug; data-URI image input |
| [**OpenAIVisionModel**](#openaivisionmodel)           | If you have an OpenAI API key and want OpenAI cloud | `OPENAI_API_KEY` (optional: `OPENAI_MODEL`, defaults to `"gpt-5.6-luna"`)                                           | Simple setup; standard OpenAI chat API                                                                       |
| [**AzureOpenAIVisionModel**](#azureopenaivisionmodel) | For Azure OpenAI Services users                     | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` (default deployment name: `gpt-5.6-luna`), `AZURE_OPENAI_API_VERSION`         | Integrates with Azure; enterprise controls                                                                   |
| [**GrokVisionModel**](#grokvisionmodel)               | If you have access to xAI’s Grok multimodal model   | `XAI_API_KEY` (optional: `XAI_MODEL`, defaults to `"grok-4.5"`)                                                  | Supports data-URIs; optional image quality                                                                   |
| [**GeminiVisionModel**](#geminivisionmodel)           | If you want Google’s Gemini Vision models           | `GEMINI_API_KEY` + **Multimodal extra:** `pip install 'splitter-mr[multimodal]'`                                 | Google Gemini API, multi-modal, high-quality extraction                                                      |
| [**HuggingFaceVisionModel**](#huggingfacevisionmodel) | Local/open-source/offline inference                 | **Multimodal extra**: `pip install 'splitter-mr[multimodal]'` (optional: `HF_ACCESS_TOKEN`, for required models) | Runs locally, uses HF `AutoProcessor` + chat templates                                                       |
| [**AnthropicVisionModel**](#anthropicvisionmodel)     | If you have an Anthropic key and want Claude Vision | `ANTHROPIC_API_KEY` (optional: `ANTHROPIC_MODEL`, defaults to `"claude-haiku-4-5"`)                    | Uses OpenAI SDK with Anthropic base URL; data-URI (base64) image input; OpenAI-compatible `chat.completions` |
| [**BaseVisionModel**](#basevisionmodel)               | Abstract base, not used directly                    | –                                                                                                              | Template to build your own adapters                                                                          |


## Models

### BaseVisionModel

::: src.splitter_mr.model.base_model
    handler: python
    options:
      extra:
        members_order: source

### OpenAIVisionModel

![OpenAIVisionModel logo](../assets/openai_vision_model_button.svg#gh-light-mode-only)
![OpenAIVisionModel logo](../assets/openai_vision_model_button_white.svg#gh-dark-mode-only)

::: src.splitter_mr.model.models.openai_model
    handler: python
    options:
      extra:
        members_order: source

### AzureOpenAIVisionModel

![OpenAIVisionModel logo](../assets/azure_openai_vision_model_button.svg#gh-light-mode-only)
![OpenAIVisionModel logo](../assets/azure_openai_vision_model_button_white.svg#gh-dark-mode-only)

::: src.splitter_mr.model.models.azure_openai_model
    handler: python
    options:
      extra:
        members_order: source

### GrokVisionModel

![GrokVisionModel logo](../assets/grok_vision_model_button.svg#gh-light-mode-only)
![GrokVisionModel logo](../assets/grok_vision_model_button_white.svg#gh-dark-mode-only)

::: src.splitter_mr.model.models.grok_model
    handler: python
    options:
      extra:
        members_order: source

### GeminiVisionModel

![GeminiVisionModel logo](../assets/gemini_vision_model_button.svg#gh-light-mode-only)
![GeminiVisionModel logo](../assets/gemini_vision_model_button_white.svg#gh-dark-mode-only)

::: src.splitter_mr.model.models.gemini_model
    handler: python
    options:
      extra:
        members_order: source

### AnthropicVisionModel

![AnthropicVisionModel logo](../assets/anthropic_vision_model_button.svg#gh-light-mode-only)
![AnthropicVisionModel logo](../assets/anthropic_vision_model_button_white.svg#gh-dark-mode-only)

::: src.splitter_mr.model.models.anthropic_model
    handler: python
    options:
      extra:
        members_order: source

### OpenRouterVisionModel

![OpenRouterVisionModel logo](../assets/openrouter_vision_model_button.svg#gh-light-mode-only)
![OpenRouterVisionModel logo](../assets/openrouter_vision_model_button_white.svg#gh-dark-mode-only)

::: src.splitter_mr.model.models.openrouter_model
    handler: python
    options:
      extra:
        members_order: source

### HuggingFaceVisionModel

!!! warning

    **`HuggingFaceVisionModel` can NOT currently support all the models available in HuggingFace**. 
    
    For example, closed models (e.g., [Microsoft Florence 2 large](https://huggingface.co/microsoft/Florence-2-large)) or models which uses uncommon architectures ([NanoNets](https://huggingface.co/nanonets/Nanonets-OCR-s)). We strongly recommend to use [Granite Docling](https://huggingface.co/ibm-granite/granite-docling-258M), since it has been exhaustively tested.

![HuggingFaceVisionModel logo](../assets/huggingface_vision_model_button.svg#gh-light-mode-only)
![HuggingFaceVisionModel logo](../assets/huggingface_vision_model_button_white.svg#gh-dark-mode-only)

::: src.splitter_mr.model.models.huggingface_model
    handler: python
    options:
      extra:
        members_order: source