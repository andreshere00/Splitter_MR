import os
from typing import Any, List, Optional

import tiktoken
from openai import OpenAI

from ...schema import OPENAI_EMBEDDING_MAX_TOKENS, OPENAI_EMBEDDING_MODEL_FALLBACK
from ..base_embedding import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """
    Encoder provider using OpenAI's embeddings API.

    This class wraps OpenAI's embeddings endpoint, providing convenience
    methods for both single-text and batch embeddings. It also adds token
    counting and validation to avoid exceeding model limits.

    Example:
        ```python
        from splitter_mr.embedding import OpenAIEmbedding

        embedder = OpenAIEmbedding(model_name="text-embedding-3-large")
        vector = embedder.embed_text("hello world")
        print(vector)
        ```
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the OpenAI embeddings provider.

        Args:
            model_name (str):
                The OpenAI embedding model name (e.g., `"text-embedding-3-large"`).
            api_key (Optional[str]):
                API key for OpenAI. If not provided, reads from the
                `OPENAI_API_KEY` environment variable.
            tokenizer_name (Optional[str]):
                Optional explicit tokenizer name for `tiktoken`. If provided,
                this overrides automatic model-to-tokenizer mapping.

        Raises:
            ValueError: If the API key is not provided or the `OPENAI_API_KEY` environment variable is not set.
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not provided or 'OPENAI_API_KEY' env var is not set."
                )
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self._tokenizer_name = tokenizer_name

    def get_client(self) -> OpenAI:
        """
        Get the configured OpenAI client.

        Returns:
            OpenAI: The OpenAI API client instance.
        """
        return self.client

    def _get_encoder(self):
        """
        Retrieve the `tiktoken` encoder for the configured model.

        If a `tokenizer_name` is explicitly provided, it is used. Otherwise,
        attempts to use `tiktoken.encoding_for_model`. If that fails, falls
        back to the default tokenizer defined by `OPENAI_EMBEDDING_MODEL_FALLBACK`.

        Returns:
            tiktoken.Encoding: The encoding object for tokenizing text.

        Raises:
            ValueError: If neither the model-specific nor fallback encoder
            can be loaded.
        """
        if self._tokenizer_name:
            return tiktoken.get_encoding(self._tokenizer_name)
        try:
            return tiktoken.encoding_for_model(self.model_name)
        except Exception:
            return tiktoken.get_encoding(OPENAI_EMBEDDING_MODEL_FALLBACK)

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text (str): The text to tokenize.

        Returns:
            int: Number of tokens.
        """
        encoder = self._get_encoder()
        return len(encoder.encode(text))

    def _validate_token_length(self, text: str) -> None:
        """
        Ensure the text does not exceed the model's token limit.

        Args:
            text (str): The text to check.

        Raises:
            ValueError: If the token count exceeds `OPENAI_EMBEDDING_MAX_TOKENS`.
        """
        if self._count_tokens(text) > OPENAI_EMBEDDING_MAX_TOKENS:
            raise ValueError(
                f"Input text exceeds maximum allowed length of {OPENAI_EMBEDDING_MAX_TOKENS} tokens."
            )

    def embed_text(self, text: str, **parameters: Any) -> List[float]:
        """
        Compute an embedding vector for a single text string.

        Args:
            text (str):
                The text to embed. Must be non-empty and within the model's
                token limit.
            **parameters:
                Additional keyword arguments forwarded to
                `client.embeddings.create(...)`.

        Returns:
            List[float]: The computed embedding vector.

        Raises:
            ValueError: If `text` is empty or exceeds the token limit.
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
        Compute embeddings for multiple texts in one API call.

        Args:
            texts (List[str]):
                List of text strings to embed. All must be non-empty and within
                the model's token limit.
            **parameters:
                Additional keyword arguments forwarded to
                `client.embeddings.create(...)`.

        Returns:
            A list of embedding vectors, one per input string.

        Raises:
            ValueError:
                - If `texts` is empty.
                - If any text is empty or not a string.
                - If any text exceeds the token limit.
        """
        if not texts:
            raise ValueError("`texts` must be a non-empty list of strings.")
        if any(not isinstance(t, str) or not t for t in texts):
            raise ValueError("All items in `texts` must be non-empty strings.")

        encoder = self._get_encoder()
        for t in texts:
            if len(encoder.encode(t)) > OPENAI_EMBEDDING_MAX_TOKENS:
                raise ValueError(
                    f"An input exceeds the maximum allowed length of {OPENAI_EMBEDDING_MAX_TOKENS} tokens."
                )

        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
            **parameters,
        )
        return [data.embedding for data in response.data]
