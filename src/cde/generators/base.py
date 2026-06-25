from abc import ABC, abstractmethod
from typing import Any


class Generator(ABC):
    """
    Base generator.
    """

    def __init__(
        self,
        model: str,
    ):
        self._model: str = model

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        options: dict[str, Any] = {},
    ) -> list[str]:
        """
        Generate text based on the given prompts and options.

        Args:
            prompt (list[str]): Prompts.
            options (dict[str, Any]): Generation options.

        Returns:
            list[str]: Generated text for each prompt.
        """
        pass
