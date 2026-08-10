import os
from typing import Any, Dict, List, Optional

import tiktoken
from openai import OpenAI

from ...schema import (
    DEFAULT_OPENROUTER_EMBEDDING_MODEL,
    DEFAULT_OPENROUTER_ENTRYPOINT,
    OPENAI_EMBEDDING_MAX_TOKENS,
    OPENAI_EMBEDDING_MODEL_FALLBACK,
)
from ..base_embedding import BaseEmbedding


class OpenRouterEmbedding(BaseEmbedding):
    """
    Embedding provider using OpenRouter's OpenAI-compatible embeddings API.

    Routes requests through OpenRouter so any supported embedding model slug
    can be used with a single API key and endpoint.
    """

    def __init__(
        self,
        model_name: str = os.getenv(
            "OPENROUTER_EMBEDDING_MODEL", DEFAULT_OPENROUTER_EMBEDDING_MODEL
        ),
        api_key: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        site_url: Optional[str] = os.getenv("OPENROUTER_SITE_URL"),
        app_name: Optional[str] = os.getenv("OPENROUTER_APP_NAME"),
    ) -> None:
        """
        Initialize the OpenRouter embeddings provider.

        Args:
            model_name (str): OpenRouter embedding model slug
                (e.g. ``openai/text-embedding-3-large``).
            api_key (Optional[str]): OpenRouter API key. Uses ``OPENROUTER_API_KEY``
                if not provided.
            tokenizer_name (Optional[str]): Optional explicit tokenizer for tiktoken.
            site_url (Optional[str]): Optional site URL for OpenRouter attribution.
            app_name (Optional[str]): Optional app name for OpenRouter attribution.

        Raises:
            ValueError: If the API key is not provided or found in the environment.
        """
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenRouter API key not provided and 'OPENROUTER_API_KEY' "
                    "env var is not set."
                )

        default_headers: Dict[str, str] = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-OpenRouter-Title"] = app_name

        self.client = OpenAI(
            api_key=api_key,
            base_url=DEFAULT_OPENROUTER_ENTRYPOINT,
            default_headers=default_headers or None,
        )
        self.model_name = model_name
        self._tokenizer_name = tokenizer_name

    def get_client(self) -> OpenAI:
        """Get the configured OpenRouter OpenAI-compatible client."""
        return self.client

    def _get_encoder(self):
        if self._tokenizer_name:
            return tiktoken.get_encoding(self._tokenizer_name)
        try:
            return tiktoken.encoding_for_model(self.model_name)
        except Exception:
            return tiktoken.get_encoding(OPENAI_EMBEDDING_MODEL_FALLBACK)

    def _count_tokens(self, text: str) -> int:
        encoder = self._get_encoder()
        return len(encoder.encode(text))

    def _validate_token_length(self, text: str) -> None:
        if self._count_tokens(text) > OPENAI_EMBEDDING_MAX_TOKENS:
            raise ValueError(
                f"Input text exceeds maximum allowed length of "
                f"{OPENAI_EMBEDDING_MAX_TOKENS} tokens."
            )

    def embed_text(self, text: str, **parameters: Any) -> List[float]:
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
        if not texts:
            raise ValueError("`texts` must be a non-empty list of strings.")
        if any(not isinstance(t, str) or not t for t in texts):
            raise ValueError("All items in `texts` must be non-empty strings.")

        encoder = self._get_encoder()
        for t in texts:
            if len(encoder.encode(t)) > OPENAI_EMBEDDING_MAX_TOKENS:
                raise ValueError(
                    f"An input exceeds the maximum allowed length of "
                    f"{OPENAI_EMBEDDING_MAX_TOKENS} tokens."
                )

        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
            **parameters,
        )
        return [data.embedding for data in response.data]
