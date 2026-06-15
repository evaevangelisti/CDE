from abc import ABC, abstractmethod
from typing import Any


class Generator(ABC):
    """
    Base generator.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        options: dict[str, Any] = {},
    ) -> str:
        """
        Generate a continuation.

        Args:
            prompt (str): Prompt.
            options (dict[str, Any]): Generation options.
        """
        pass
