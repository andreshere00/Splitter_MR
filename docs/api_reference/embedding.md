# **Embedding Models**

## Overview

Encoder models are the engines which produce *embeddings*. These embeddings are distributed and vectorized representations of a text. These embeddings allows to capture relationships between semantic units (commonly words, but can be sentences, or even multimodal content such as images).  

These embeddings can be used in a variety of tasks, such as:

- Measuring how relevant a word is within a text.  
- Comparing the similarity between two pieces of text.  
- Power searching, clustering, and recommendation systems building.  

![Example of an embedding representation](../assets/vectorization.png)

**SplitterMR** takes advantage of these models in [**`SemanticSplitter`**](./splitter.md#semanticsplitter). These representations are used to break text into chunks based on *meaning*, not just size. Sentences with similar context end up together, regardless of length or position.

## Which embedder should I use?

All embedders inherit from [**BaseEmbedding**](#baseembedding) and expose the same interface for generating embeddings. Choose based on your cloud provider, credentials, and compliance needs.

| Model                                             | When to use                                                                 | Requirements                                                                                                        | Features                                                                                                            |
| ------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| [**OpenAIEmbedding**](#openaiembedding)           | You have an OpenAI API key and want to use OpenAI’s hosted embeddings       | `OPENAI_API_KEY`                                                                                                    | Production-ready text embeddings; simple setup; broad ecosystem/tooling support.                                    |
| [**AzureOpenAIEmbedding**](#azureopenaiembedding) | Your organization uses Azure OpenAI Services                                | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`                                          | Enterprise controls, Azure compliance & data residency; integrates with Azure identity.                             |
| [**GeminiEmbedding**](#geminiembedding)           | You want Google’s Gemini text embeddings                                    | `GEMINI_API_KEY` + **Multimodal extra**: `pip install 'splitter-mr[multimodal]'`                                      | Google Gemini API; modern, high-quality text embeddings.                                                            |
| [**AnthropicEmbeddings**](#anthropicembedding)   | You want embeddings aligned with Anthropic guidance (via Voyage AI)         | `VOYAGE_API_KEY` + **Multimodal extra**: `pip install 'splitter-mr[multimodal]'`                                      | Voyage AI embeddings (general, code, finance, law, multimodal); supports `input_type` for query/document asymmetry. |
| [**HuggingFaceEmbedding**](#huggingfaceembedding) | Prefer local/open-source models (Sentence-Transformers); offline capability | **Multimodal extra**: `pip install 'splitter-mr[multimodal]'` (optional: `HF_ACCESS_TOKEN`, only for required models) | No API key; huge model zoo; CPU/GPU/MPS; optional L2 normalization for cosine similarity.                           |
| [**BaseEmbedding**](#baseembedding)               | Abstract base, not used directly                                            | –                                                                                                                   | Implement to plug in a custom or self-hosted embedder.                                                              |

!!! note

    In case that you want to bring your own embedding provider, you can easily implement the class using [**`BaseEmbedding`**](#baseembedding).

## Embedders

### BaseEmbedding

::: src.splitter_mr.embedding.base_embedding
    handler: python
    options:
      members_order: source

### OpenAIEmbedding

![OpenAIEmbedding logo](../assets/openai_embedding_model_button.svg#gh-light-mode-only)
![OpenAIEmbedding logo](../assets/openai_embedding_model_button_white.svg#gh-dark-mode-only)

::: src.splitter_mr.embedding.embeddings.openai_embedding
    handler: python
    options:
      members_order: source

### AzureOpenAIEmbedding

![AzureOpenAIEmbedding logo](../assets/azure_openai_embedding_model_button.svg#gh-light-mode-only)
![AzureOpenAIEmbedding logo](../assets/azure_openai_embedding_model_button_white.svg#gh-dark-mode-only)

::: src.splitter_mr.embedding.embeddings.azure_openai_embedding
    handler: python
    options:
      members_order: source

### GeminiEmbedding

![GeminiEmbedding logo](../assets/gemini_embedding_model_button.svg#gh-light-mode-only)
![GeminiEmbedding logo](../assets/gemini_embedding_model_button_white.svg#gh-dark-mode-only)

::: src.splitter_mr.embedding.embeddings.gemini_embedding
    handler: python
    options:
      members_order: source

### AnthropicEmbedding

![AnthropicEmbedding logo](../assets/anthropic_embedding_model_button.svg#gh-light-mode-only)
![AnthropicEmbedding logo](../assets/anthropic_embedding_model_button_white.svg#gh-dark-mode-only)

::: src.splitter_mr.embedding.embeddings.anthropic_embedding
    handler: python
    options:
      members_order: source

### HuggingFaceEmbedding

!!! warning

    Currently, only models compatible with `sentence-transformers` library are available. 

![HuggingFaceEmbedding logo](../assets/huggingface_embedding_model_button.svg#gh-light-mode-only)
![HuggingFaceEmbedding logo](../assets/huggingface_embedding_model_button_white.svg#gh-dark-mode-only)

::: src.splitter_mr.embedding.embeddings.huggingface_embedding
    handler: python
    options:
      members_order: source
