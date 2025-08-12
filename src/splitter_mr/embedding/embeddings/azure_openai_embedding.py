import os
from typing import Any, List, Optional

import tiktoken
from openai import AzureOpenAI

from ...schema import OPENAI_EMBEDDING_MAX_TOKENS
from ..base_embedding import BaseEmbedding


class AzureOpenAIEmbedding(BaseEmbedding):
    """
    Encoder provider using Azure OpenAI Embeddings.

    You must provision a deployment for an embeddings model (e.g., "text-embedding-3-large")
    and pass its deployment name via `azure_deployment` or env var.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name: Unused for Azure (deployment name is what matters). Kept to
                respect BaseEmbedding signature. If provided and `azure_deployment`
                is None, we'll use this value as the deployment name.
            api_key: If None, reads from AZURE_OPENAI_API_KEY.
            azure_endpoint: If None, reads from AZURE_OPENAI_ENDPOINT.
            azure_deployment: The *deployment name* for your embeddings model. If None,
                reads from AZURE_OPENAI_DEPLOYMENT, or falls back to `model_name`.
            api_version: If None, reads from AZURE_OPENAI_API_VERSION
                (defaults to "2025-04-14-preview").
        """
        if api_key is None:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Azure OpenAI API key not provided or 'AZURE_OPENAI_API_KEY' env var is not set."
                )

        if azure_endpoint is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint:
                raise ValueError(
                    "Azure endpoint not provided or 'AZURE_OPENAI_ENDPOINT' env var is not set."
                )

        if azure_deployment is None:
            # prefer explicit deployment env var; otherwise allow model_name to stand in
            azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or model_name
            if not azure_deployment:
                raise ValueError(
                    "Azure deployment name not provided. Set 'azure_deployment', "
                    "'AZURE_OPENAI_DEPLOYMENT', or pass `model_name`."
                )

        if api_version is None:
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-14-preview")

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
        )
        self.model_name = azure_deployment

    def get_client(self) -> AzureOpenAI:
        """
        Return the initialized Azure OpenAI client instance.

        Returns:
            AzureOpenAI: The API client configured with provided credentials and
            endpoint details.
        """
        return self.client

    def _validate_token_length(self, text: str) -> None:
        """Validate that the text does not exceed the model's token limit.

        Args:
            text: The input text to check.

        Raises:
            ValueError: If the text exceeds `OPENAI_EMBEDDING_MAX_TOKENS` tokens.
        """
        enc = tiktoken.encoding_for_model(self.model_name)
        tokens = enc.encode(text)
        if len(tokens) > OPENAI_EMBEDDING_MAX_TOKENS:
            raise ValueError(
                f"Input text exceeds maximum allowed length of {OPENAI_EMBEDDING_MAX_TOKENS} tokens."
            )

    def embed_text(self, text: str, **parameters: Any) -> List[float]:
        """
        Compute an embedding for a single text using Azure OpenAI.

        Args:
            text: The text to embed.
            **parameters: Additional args forwarded to `client.embeddings.create(...)`.

        Returns:
            List[float]: Embedding vector.

        Raises:
            ValueError: If `text` is empty or exceeds max token length.

        Example:
            ```python
            from splitter_mr.embedding import AzureOpenAIEmbedding

            embedder = AzureOpenAIEmbedding(model_name="text-embedding-3-large")
            embedding = embedder.embed_text("hello world")
            print(embedding)
            ```
        """
        if not text:
            raise ValueError("`text` must be a non-empty string.")

        self._validate_token_length(text)

        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
            **parameters,
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str], **parameters: Any) -> List[List[float]]:
        """
        Batch embeddings using a single Azure OpenAI API call.

        Args:
            texts: List of input strings to embed.
            **parameters: AzureOpenAI embeddings API-specific parameters.

        Returns:
            List of embedding vectors, one per input string.

        Raises:
            ValueError:
            - If `texts` is empty or any element is empty
            - If an input exceeds `OPENAI_EMBEDDING_MAX_TOKENS` tokens.
        """
        if not texts:
            raise ValueError("`texts` must be a non-empty list of strings.")
        if any(not isinstance(t, str) or not t for t in texts):
            raise ValueError("All items in `texts` must be non-empty strings.")

        # Per-item token validation
        enc = tiktoken.encoding_for_model(self.model_name)
        for t in texts:
            if len(enc.encode(t)) > OPENAI_EMBEDDING_MAX_TOKENS:
                raise ValueError(
                    "An input exceeds the maximum allowed length of "
                    f"{OPENAI_EMBEDDING_MAX_TOKENS} tokens."
                )

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            **parameters,
        )
        # Ensure order is preserved
        return [d.embedding for d in response.data]
