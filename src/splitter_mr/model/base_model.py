from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModel(ABC):
    @abstractmethod
    def __init__(self, model_name) -> Any:
        pass

    @abstractmethod
    def get_client(self) -> Any:
        pass

    @abstractmethod
    def extract_text(
        self, prompt: str, file: Optional[bytes], **parameters: Dict[str, Any]
    ) -> str:
        """
        Extracts text from the provided image (base64-encoded string) using the prompt.
        """
        pass
