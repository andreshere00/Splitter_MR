# Visual Models

Reading documents like Word, PDF, or PowerPoint can sometimes be complicated if they contain images. To avoid this problem, **you can use visual language models (VLMs), which are capable of recognizing images and extracting descriptions from them**. In this prospectus, a model module has been developed, the implementation of which is based on the `BaseModel` class. It is presented below.

## Which model should I use?

The choice of model depends on your cloud provider, available API keys, and desired level of integration.
All models inherit from `BaseModel` and provide the same interface for extracting text and descriptions from images.

| Model                    | When to use                                                    | Requirements                               | Features                                     |
| ------------------------ | -------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------- |
| `OpenAIVisionModel`      | Use if you have an OpenAI API key and want to use OpenAI cloud | OpenAI account & API key                   | No Azure setup, easy to get started.         |
| `AzureOpenAIVisionModel` | Use if your organization uses Azure OpenAI Services            | Azure OpenAI deployment, API key, endpoint | Integration with Azure, enterprise security. |
| `BaseModel`              | Abstract base, not used directly                               | –                                          | Use as a template for building your own.     |


## Models

### BaseModel

::: src.splitter_mr.model.base_model
    handler: python
    options:
      members_order: source

### OpenAIVisionModel

<img alt="OpenAI logo" src = "https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenAI_Logo.svg" width = 50%>

::: src.splitter_mr.model.models.openai_model
    handler: python
    options:
      members_order: source

### AzureOpenAIVisionModel

<img alt="OpenAI logo" src = "https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenAI_Logo.svg" width = 50%>

::: src.splitter_mr.model.models.azure_openai_model
    handler: python
    options:
      members_order: source