from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModel(ABC):
    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def get_client(self) -> Any:
        pass

    @abstractmethod
    def extract_text(
        self, file_b64: str, prompt: str, **parameters: Dict[str, Any]
    ) -> str:
        """
        Extracts text from the provided image (base64-encoded string) using the prompt.
        """
        pass

    @abstractmethod
    def analyze_resource(
        self, file_b64: str, context: str, prompt: str, **parameters: Dict[str, Any]
    ) -> str:
        """
        Analyzes the image (base64-encoded string) based on content, prompt, and context.
        """
        pass
