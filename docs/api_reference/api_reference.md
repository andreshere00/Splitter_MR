# **Developer guide**

Welcome to the **SplitterMR Python API** reference.

![SplitterMR architecture diagram](../assets/splitter_mr_architecture_diagram.svg#gh-light-mode-only)

![SplitterMR architecture diagram](../assets/splitter_mr_architecture_diagram_white.svg#gh-dark-mode-only)

## **Documentation**

### [Vision Model component](./model.md)

Extend reader capabilities using VLMs (Visual Language Models) to analyze visual content from your documents. All vision providers implement [**`BaseVisionModel.analyze_content`**](model.md#basevisionmodel) to run a prompt against a base64-encoded image. For a single OpenRouter key and many upstream models, see [**`OpenRouterVisionModel`**](model.md#openroutervisionmodel).

### [Reader component](./reader.md)

Use different reading methods to process your files before splitting them.

### [Splitter component](./splitter.md)

Implement several splitting strategies based on the type of document and use case.

### [Embedding component](./embedding.md)

Implement encoder models from different providers to perform semantic splitting. Embedders expose [**`BaseEmbedding.embed_text`**](embedding.md#baseembedding) and [**`embed_documents`**](embedding.md#baseembedding) to vectorize strings for similarity-based chunking. For OpenRouter-hosted embedding slugs, see [**`OpenRouterEmbedding`**](embedding.md#openrouterembedding).

### [MCP and REST server](./server.md)

Expose the same read, split, and read-and-split pipeline through Swagger-documented REST endpoints and typed MCP tools. Install `splitter-mr[mcp]` and run `splitter-mr-mcp`.
