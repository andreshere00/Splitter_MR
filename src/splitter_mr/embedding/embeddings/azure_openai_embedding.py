import os
from typing import Any, List, Optional

import tiktoken
from openai import AzureOpenAI

from ...schema import OPENAI_EMBEDDING_MAX_TOKENS, OPENAI_EMBEDDING_MODEL_FALLBACK
from ..base_embedding import BaseEmbedding


class AzureOpenAIEmbedding(BaseEmbedding):
    """
    Encoder provider using Azure OpenAI Embeddings.

    This class wraps Azure OpenAI's embeddings API, handling both authentication
    and tokenization. It supports both direct embedding calls for a single text
    (`embed_text`) and batch embedding calls (`embed_documents`).

    Azure deployments use *deployment names* (e.g., `my-embedding-deployment`)
    instead of OpenAI's standard model names. Since `tiktoken` may not be able to
    map a deployment name to a tokenizer automatically, this class implements
    a fallback mechanism to use a known encoding (e.g., `cl100k_base`) when necessary.

    Example:
        ```python
        from splitter_mr.embedding import AzureOpenAIEmbedding

        embedder = AzureOpenAIEmbedding(
            azure_deployment="text-embedding-3-large",
            api_key="...",
            azure_endpoint="https://my-azure-endpoint.openai.azure.com/"
        )
        vector = embedder.embed_text("Hello world")
        ```
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the Azure OpenAI Embedding provider.

        Args:
            model_name (Optional[str]):
                OpenAI model name (unused for Azure, but kept for API parity).
                If `azure_deployment` is not provided, this will be used as the
                deployment name.
            api_key (Optional[str]):
                API key for Azure OpenAI. If not provided, it will be read from
                the environment variable `AZURE_OPENAI_API_KEY`.
            azure_endpoint (Optional[str]):
                The base endpoint for the Azure OpenAI service. If not provided,
                it will be read from `AZURE_OPENAI_ENDPOINT`.
            azure_deployment (Optional[str]):
                Deployment name for the embeddings model in Azure OpenAI. If not
                provided, it will be read from `AZURE_OPENAI_DEPLOYMENT` or
                fallback to `model_name`.
            api_version (Optional[str]):
                Azure API version string. Defaults to `"2025-04-14-preview"`.
                If not provided, it will be read from `AZURE_OPENAI_API_VERSION`.
            tokenizer_name (Optional[str]):
                Optional explicit tokenizer name for `tiktoken` (e.g.,
                `"cl100k_base"`). If provided, it overrides the automatic mapping.

        Raises:
            ValueError: If any required parameter is missing or it is not found in environment variables.
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
        self._tokenizer_name = tokenizer_name

    def get_client(self) -> AzureOpenAI:
        """
        Get the underlying Azure OpenAI client.

        Returns:
            AzureOpenAI: The configured Azure OpenAI API client.
        """
        return self.client

    def _get_encoder(self):
        """
        Retrieve the `tiktoken` encoder for this deployment.

        This method ensures compatibility with Azure's deployment names, which
        may not be directly recognized by `tiktoken`. If the user has explicitly
        provided a tokenizer name, that is used. Otherwise, the method first
        tries to look up the encoding via `tiktoken.encoding_for_model` using the
        deployment name. If that fails, it falls back to the default encoding
        defined by `OPENAI_EMBEDDING_MODEL_FALLBACK`.

        Returns:
            tiktoken.Encoding: A tokenizer encoding object.

        Raises:
            ValueError: If `tiktoken` fails to load the fallback encoding.
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

        Uses the encoder retrieved from `_get_encoder()` to tokenize the input
        and returns the length of the resulting token list.

        Args:
            text (str): The text to tokenize.

        Returns:
            int: Number of tokens in the input text.
        """
        encoder = self._get_encoder()
        return len(encoder.encode(text))

    def _validate_token_length(self, text: str) -> None:
        """
        Ensure the input text does not exceed the model's maximum token limit.

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
                Additional parameters to forward to the Azure OpenAI embeddings API.

        Returns:
            List[float]: The computed embedding vector.

        Raises:
            ValueError: If `text` is empty or exceeds the token limit.
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
        Compute embeddings for multiple texts in a single API call.

        Args:
            texts (List[str]):
                List of text strings to embed. All items must be non-empty strings
                within the token limit.
            **parameters:
                Additional parameters to forward to the Azure OpenAI embeddings API.

        Returns:
            A list of embedding vectors, one per input text.

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
            model=self.model_name,
            input=texts,
            **parameters,
        )
        return [data.embedding for data in response.data]
