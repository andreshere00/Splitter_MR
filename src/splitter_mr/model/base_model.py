from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """
        Runs Image processing tasks on the data.
        """
        pass
