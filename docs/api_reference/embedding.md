# Embedding Models

Encoder models are the engines that produce *embeddings* — vectorized representations of your input (see the image below). These embeddings capture mathematical relationships between semantic units (like words, sentences, or even images).  

Why does this matter? Because once you have embeddings, you can:  
- Measure how relevant a word is within a text.  
- Compare the similarity between two pieces of text.  
- Power search, clustering, and recommendation systems.  

![Example of an embedding representation](../assets/vectorization.png)

**SplitterMR** takes advantage of these models to break text into chunks based on *meaning*, not just size. Sentences with similar context end up together, regardless of length or position. This approach is called `SemanticSplitter` — perfect when you want your chunks to *make sense* rather than just follow arbitrary size limits.

Below is the list of embedding models you can use out-of-the-box.  
And if you want to bring your own, simply implement `BaseEmbedding` and plug it in.

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

![AzureOpenAIEmbedding logo](../assets/azure_openai_embedding_button.svg#gh-light-mode-only)
![AzureOpenAIEmbedding logo](../assets/azure_openai_embedding_button_white.svg#gh-dark-mode-only)

::: src.splitter_mr.embedding.embeddings.azure_openai_embedding
    handler: python
    options:
      members_order: source
