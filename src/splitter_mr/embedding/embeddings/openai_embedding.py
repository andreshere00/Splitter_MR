import os
from typing import Any, List, Optional

import tiktoken
from openai import OpenAI

from ...schema import OPENAI_EMBEDDING_MAX_TOKENS
from ..base_embedding import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """
    Encoder provider using OpenAI's embeddings API.

    Example:
        ```python
        from splitter_mr.embedding import OpenAIEmbedding

        embedder = OpenAIEmbedding(model_name="text-embedding-3-large")
        embedding = embedder.embed_text("hello world")
        print(embedding)
        ```
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name: OpenAI embedding model name, e.g. "text-embedding-3-large".
            api_key: If None, reads from OPENAI_API_KEY.
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not provided or 'OPENAI_API_KEY' env var is not set."
                )
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def get_client(self) -> OpenAI:
        """
        Return the initialized OpenAI client instance.

        Returns:
            OpenAI: The API client configured with provided credentials.
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
        Compute an embedding for a single text.

        Args:
            text: The text to embed.
            **parameters: Additional args forwarded to `client.embeddings.create(...)`.

        Returns:
            List[float]: Embedding vector.

        Raises:
            ValueError: If `text` is empty or exceeds max token length.

        Example:
            ```python
            from splitter_mr.embedding import OpenAIEmbedding

            embedder = OpenAIEmbedding(model_name="text-embedding-3-large")
            embedding = embedder.embed_text("hello world")
            print(embedding)
            ```
        """
        if not text:
            raise ValueError("`text` must be a non-empty string.")

        self._validate_token_length(text)

        response = self.client.embeddings.create(
            input=text,
            model=self.model_name,
            **parameters,
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str], **parameters: Any) -> List[List[float]]:
        """
        Batch embeddings using a single OpenAI embeddings call.

        Args:
            texts: List of input strings to embed.
            **parameters: OpenAI embeddings API-specific parameters.

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

        enc = tiktoken.encoding_for_model(self.model_name)
        for t in texts:
            if len(enc.encode(t)) > OPENAI_EMBEDDING_MAX_TOKENS:
                raise ValueError(
                    f"An input exceeds the maximum allowed length of {OPENAI_EMBEDDING_MAX_TOKENS} tokens."
                )

        resp = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
            **parameters,
        )
        return [d.embedding for d in resp.data]
